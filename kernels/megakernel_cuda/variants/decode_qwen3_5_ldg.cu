/**
 * Qwen3.5-0.8B fused decode -- persistent multi-block kernel (ported from Qwen3 megakernel).
 *
 * Hybrid architecture: 18 linear attention + 6 full attention layers.
 * One kernel launch per token: embed -> 24x(layer) -> final norm -> LM head.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include "sm_profiler.h"

// sm-profiler event IDs
#define SM_PROF_EMBEDDING    0
#define SM_PROF_QKV_PROJ     1
#define SM_PROF_QK_NORM_ROPE 2
#define SM_PROF_ATTN_COMPUTE 3
#define SM_PROF_ATTN_PREFETCH 4
#define SM_PROF_O_PROJ_MLP   5
#define SM_PROF_FINAL_NORM   6
#define SM_PROF_GRID_SYNC    7
#define SM_PROF_NUM_EVENTS   8

// =============================================================================
// Configuration & model constants for Qwen3.5-0.8B
// =============================================================================

constexpr int WARP_SIZE = 32;

constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3584;

// Full attention dims
constexpr int NUM_Q_HEADS = 8;
constexpr int NUM_KV_HEADS = 2;
constexpr int HEAD_DIM = 256;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;      // 2048
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM;    // 512
constexpr int ROTARY_DIM = 64;

// Linear attention dims
constexpr int NUM_K_HEADS = 16;
constexpr int NUM_V_HEADS = 16;
constexpr int HEAD_K_DIM = 128;
constexpr int HEAD_V_DIM = 128;
constexpr int KEY_DIM = NUM_K_HEADS * HEAD_K_DIM;     // 2048
constexpr int VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM;   // 2048
constexpr int CONV_DIM = KEY_DIM * 2 + VALUE_DIM;     // 6144
constexpr int CONV_KERNEL_SIZE = 4;

// Grid
constexpr int LDG_BLOCK_SIZE = 256;
constexpr int LDG_NUM_WARPS = LDG_BLOCK_SIZE / WARP_SIZE; // 8
constexpr float LDG_RMS_EPS = 1e-6f;

// Shared memory bank conflict padding: 1 pad per 8 elements (stride 9)
#define SMEM_PAD_IDX(i) ((i) + (i) / 8)
#define SMEM_PAD_SIZE(n) ((n) + (n) / 8)

// LM head
constexpr int LDG_LM_NUM_BLOCKS = 1184;
constexpr int LDG_LM_BLOCK_SIZE = 256;
constexpr int LDG_VOCAB_SIZE = 248320;

// =============================================================================
// Unified layer weight struct
// =============================================================================

struct LayerWeights {
    int64_t layer_type;  // 0 = full_attention, 1 = linear_attention

    // Common
    const __nv_bfloat16 *input_norm;
    const __nv_bfloat16 *post_norm;
    const __nv_bfloat16 *gate_proj;
    const __nv_bfloat16 *up_proj;
    const __nv_bfloat16 *down_proj;

    // Full attention
    const __nv_bfloat16 *full_q_proj;
    const __nv_bfloat16 *full_k_proj;
    const __nv_bfloat16 *full_v_proj;
    const __nv_bfloat16 *full_q_norm;
    const __nv_bfloat16 *full_k_norm;
    const __nv_bfloat16 *full_o_proj;

    // Linear attention
    const __nv_bfloat16 *linear_in_proj_qkv;
    const __nv_bfloat16 *linear_conv1d_weight;
    const __nv_bfloat16 *linear_in_proj_z;
    const __nv_bfloat16 *linear_in_proj_b;
    const __nv_bfloat16 *linear_in_proj_a;
    const float *linear_dt_bias;
    const float *linear_a_log;
    const __nv_bfloat16 *linear_norm_weight;
    const __nv_bfloat16 *linear_out_proj;
};

// =============================================================================
// Atomic barrier for persistent kernel
// =============================================================================

struct AtomicGridSync {
    unsigned int *counter;
    unsigned int *generation;
    unsigned int nblocks;
    unsigned int local_gen;

    __device__ void sync() {
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int my_gen = local_gen;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1) {
                *counter = 0;
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                atomicAdd(generation, 1);
            } else {
                volatile unsigned int *vgen = (volatile unsigned int *)generation;
                while (*vgen <= my_gen) {}
            }
            local_gen = my_gen + 1;
        }
        __syncthreads();
    }
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float ldg_warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float ldg_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus(float x) {
    return logf(1.0f + expf(x));
}

// =============================================================================
// Phase 1: RMSNorm + Projection
// =============================================================================

__device__ void ldg_norm_and_proj(
    AtomicGridSync &grid,
    const __nv_bfloat16 *__restrict__ hidden_in,
    const LayerWeights &__restrict__ w,
    float *__restrict__ g_activations,
    float *__restrict__ g_residual,
    float *__restrict__ g_q,       // full: q+gate; linear: conv_in
    float *__restrict__ g_k,       // full: k;      linear: k
    float *__restrict__ g_v,       // full: v;      linear: v
    float *__restrict__ g_z,       // linear only: z
    float *__restrict__ g_b,       // linear only: b
    float *__restrict__ g_a,       // linear only: a
    uint64_t *__restrict__ profiler_buffer,
    bool prof_on)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // RMSNorm into shared memory (all blocks redundant compute to avoid grid.sync)
    __shared__ float s_normalized[SMEM_PAD_SIZE(HIDDEN_SIZE)];
    {
        __shared__ float smem_reduce[LDG_NUM_WARPS];
        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = __bfloat162float(__ldg(hidden_in + i));
            s_normalized[SMEM_PAD_IDX(i)] = v;
            local_sum_sq += v * v;
        }
        local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[0];

        // Distributed residual write (raw hidden state, BEFORE normalization)
        {
            int res_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
            int res_start = block_id * res_per_block;
            int res_end = min(res_start + res_per_block, HIDDEN_SIZE);
            for (int i = res_start + threadIdx.x; i < res_end; i += LDG_BLOCK_SIZE)
                g_residual[i] = s_normalized[SMEM_PAD_IDX(i)];
        }

        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float wt = __bfloat162float(__ldg(w.input_norm + i));
            s_normalized[SMEM_PAD_IDX(i)] = s_normalized[SMEM_PAD_IDX(i)] * rstd * (1.0f + wt);
        }
        __syncthreads();
    }

    if (w.layer_type == 0) {
        // Full attention: QKV projection
        constexpr int TOTAL_ROWS = Q_SIZE * 2 + KV_SIZE + KV_SIZE; // q+gate, k, v
        int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
        int row_start = block_id * rows_per_block;
        int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

        for (int m_base = row_start; m_base < row_end; m_base += LDG_NUM_WARPS) {
            int m = m_base + warp_id;
            if (m < row_end) {
                const __nv_bfloat16 *weight_row;
                float *output_ptr;
                if (m < Q_SIZE * 2) {
                    weight_row = w.full_q_proj + m * HIDDEN_SIZE;
                    output_ptr = g_q + m;
                } else if (m < Q_SIZE * 2 + KV_SIZE) {
                    weight_row = w.full_k_proj + (m - Q_SIZE * 2) * HIDDEN_SIZE;
                    output_ptr = g_k + (m - Q_SIZE * 2);
                } else {
                    weight_row = w.full_v_proj + (m - Q_SIZE * 2 - KV_SIZE) * HIDDEN_SIZE;
                    output_ptr = g_v + (m - Q_SIZE * 2 - KV_SIZE);
                }
                float sum = 0.0f;
#pragma unroll 4
                for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                    uint4 w_u4 = __ldg(reinterpret_cast<const uint4 *>(weight_row + k));
                    __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);
                    int pk = SMEM_PAD_IDX(k);
                    sum += __bfloat162float(w_ptr[0]) * s_normalized[pk + 0] +
                           __bfloat162float(w_ptr[1]) * s_normalized[pk + 1] +
                           __bfloat162float(w_ptr[2]) * s_normalized[pk + 2] +
                           __bfloat162float(w_ptr[3]) * s_normalized[pk + 3] +
                           __bfloat162float(w_ptr[4]) * s_normalized[pk + 4] +
                           __bfloat162float(w_ptr[5]) * s_normalized[pk + 5] +
                           __bfloat162float(w_ptr[6]) * s_normalized[pk + 6] +
                           __bfloat162float(w_ptr[7]) * s_normalized[pk + 7];
                }
                sum = ldg_warp_reduce_sum(sum);
                if (lane_id == 0) *output_ptr = sum;
            }
        }
    } else {
        // Linear attention: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
        // in_proj_qkv -> g_q (size CONV_DIM)
        // in_proj_z   -> g_z (size VALUE_DIM)
        // in_proj_b   -> g_b (size NUM_V_HEADS)
        // in_proj_a   -> g_a (size NUM_V_HEADS)
        constexpr int TOTAL_ROWS = CONV_DIM + VALUE_DIM + NUM_V_HEADS + NUM_V_HEADS;
        int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
        int row_start = block_id * rows_per_block;
        int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

        for (int m_base = row_start; m_base < row_end; m_base += LDG_NUM_WARPS) {
            int m = m_base + warp_id;
            if (m < row_end) {
                const __nv_bfloat16 *weight_row;
                float *output_ptr;
                if (m < CONV_DIM) {
                    weight_row = w.linear_in_proj_qkv + m * HIDDEN_SIZE;
                    output_ptr = g_q + m;
                } else if (m < CONV_DIM + VALUE_DIM) {
                    weight_row = w.linear_in_proj_z + (m - CONV_DIM) * HIDDEN_SIZE;
                    output_ptr = g_z + (m - CONV_DIM);
                } else if (m < CONV_DIM + VALUE_DIM + NUM_V_HEADS) {
                    weight_row = w.linear_in_proj_b + (m - CONV_DIM - VALUE_DIM) * HIDDEN_SIZE;
                    output_ptr = g_b + (m - CONV_DIM - VALUE_DIM);
                } else {
                    weight_row = w.linear_in_proj_a + (m - CONV_DIM - VALUE_DIM - NUM_V_HEADS) * HIDDEN_SIZE;
                    output_ptr = g_a + (m - CONV_DIM - VALUE_DIM - NUM_V_HEADS);
                }
                float sum = 0.0f;
#pragma unroll 4
                for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                    uint4 w_u4 = __ldg(reinterpret_cast<const uint4 *>(weight_row + k));
                    __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);
                    int pk = SMEM_PAD_IDX(k);
                    sum += __bfloat162float(w_ptr[0]) * s_normalized[pk + 0] +
                           __bfloat162float(w_ptr[1]) * s_normalized[pk + 1] +
                           __bfloat162float(w_ptr[2]) * s_normalized[pk + 2] +
                           __bfloat162float(w_ptr[3]) * s_normalized[pk + 3] +
                           __bfloat162float(w_ptr[4]) * s_normalized[pk + 4] +
                           __bfloat162float(w_ptr[5]) * s_normalized[pk + 5] +
                           __bfloat162float(w_ptr[6]) * s_normalized[pk + 6] +
                           __bfloat162float(w_ptr[7]) * s_normalized[pk + 7];
                }
                sum = ldg_warp_reduce_sum(sum);
                if (lane_id == 0) *output_ptr = sum;
            }
        }
    }

    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

// =============================================================================
// Phase 2: Full -- QK Norm + RoPE + KV Cache Write
//        Linear -- Causal Conv1d + L2 Norm + Q scaling
// =============================================================================

__device__ void ldg_prep_full(
    AtomicGridSync &grid,
    float *__restrict__ q,
    float *__restrict__ k,
    float *__restrict__ v,
    const __nv_bfloat16 *__restrict__ q_norm_weight,
    const __nv_bfloat16 *__restrict__ k_norm_weight,
    const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table,
    __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ v_cache,
    int position,
    int max_seq_len,
    unsigned int *__restrict__ kv_flag,
    int full_layer_idx,
    uint64_t *__restrict__ profiler_buffer,
    bool prof_on)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16 *cos_pos = cos_table + position * ROTARY_DIM;
    const __nv_bfloat16 *sin_pos = sin_table + position * ROTARY_DIM;

    // Q heads: norm per head + extract gate + RoPE on first ROTARY_DIM
    int q_heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int q_head_start = block_id * q_heads_per_block;
    int q_head_end = min(q_head_start + q_heads_per_block, NUM_Q_HEADS);

    for (int h = q_head_start + warp_id; h < q_head_end; h += LDG_NUM_WARPS) {
        float *q_head = q + h * HEAD_DIM * 2;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
            sum_sq += q_head[i] * q_head[i];
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float q_local[HEAD_DIM / WARP_SIZE];
#pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
            q_local[j] = q_head[i] * scale * (1.0f + __bfloat162float(__ldg(q_norm_weight + i)));

        // RoPE (partial: only first ROTARY_DIM elements)
#pragma unroll
        for (int i = lane_id, j = 0; i < ROTARY_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));
            int pair_offset = (i < ROTARY_DIM / 2) ? ROTARY_DIM / 2 : -ROTARY_DIM / 2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);
            if (i < ROTARY_DIM / 2)
                q_head[i] = q_local[j] * cos_v - pair_v * sin_v;
            else
                q_head[i] = pair_v * sin_v + q_local[j] * cos_v;
        }
        // Write non-RoPE dimensions back (already normalized)
        for (int i = lane_id + ROTARY_DIM; i < HEAD_DIM; i += WARP_SIZE)
            q_head[i] = q_local[(i - lane_id) / WARP_SIZE];
    }

    // K heads + cache write
    int k_heads_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
    int k_head_start = block_id * k_heads_per_block;
    int k_head_end = min(k_head_start + k_heads_per_block, NUM_KV_HEADS);

    for (int h = k_head_start + warp_id; h < k_head_end; h += LDG_NUM_WARPS) {
        float *k_head = k + h * HEAD_DIM;
        const float *v_head = v + h * HEAD_DIM;
        __nv_bfloat16 *k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __nv_bfloat16 *v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
            sum_sq += k_head[i] * k_head[i];
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float k_local[HEAD_DIM / WARP_SIZE];
#pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
            k_local[j] = k_head[i] * scale * (1.0f + __bfloat162float(__ldg(k_norm_weight + i)));

#pragma unroll
        for (int i = lane_id, j = 0; i < ROTARY_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));
            int pair_offset = (i < ROTARY_DIM / 2) ? ROTARY_DIM / 2 : -ROTARY_DIM / 2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);
            float k_final;
            if (i < ROTARY_DIM / 2)
                k_final = k_local[j] * cos_v - pair_v * sin_v;
            else
                k_final = pair_v * sin_v + k_local[j] * cos_v;
            k_head[i] = k_final;
            k_cache_head[i] = __float2bfloat16(k_final);
            v_cache_head[i] = __float2bfloat16(v_head[i]);
        }
        // Write non-RoPE dimensions of k/v to cache (k must be normalized)
        for (int i = lane_id + ROTARY_DIM; i < HEAD_DIM; i += WARP_SIZE) {
            int j = (i - lane_id) / WARP_SIZE;
            k_head[i] = k_local[j];
            k_cache_head[i] = __float2bfloat16(k_local[j]);
            v_cache_head[i] = __float2bfloat16(v_head[i]);
        }
    }

    // kv_flag partial barrier
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    const int ATTN_BLOCKS = NUM_Q_HEADS;
    if (block_id < ATTN_BLOCKS) {
        __syncthreads();
        if (threadIdx.x == 0) {
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            atomicAdd(kv_flag, 1);
            unsigned int target = (unsigned int)(ATTN_BLOCKS * (full_layer_idx + 1));
            volatile unsigned int *vf = (volatile unsigned int *)kv_flag;
            while (*vf < target) {}
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
        }
        __syncthreads();
    }
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

__device__ void ldg_prep_linear(
    AtomicGridSync &grid,
    float *__restrict__ g_conv_in,   // size CONV_DIM
    float *__restrict__ g_q,         // size KEY_DIM
    float *__restrict__ g_k,         // size KEY_DIM
    float *__restrict__ g_v,         // size VALUE_DIM
    float *__restrict__ g_b,         // size NUM_V_HEADS
    float *__restrict__ g_a,         // size NUM_V_HEADS
    float *__restrict__ g_z,         // size VALUE_DIM
    const __nv_bfloat16 *__restrict__ conv1d_weight,
    const float *__restrict__ dt_bias,
    const float *__restrict__ a_log,
    float *__restrict__ conv_state,  // [CONV_DIM, CONV_KERNEL_SIZE-1]
    uint64_t *__restrict__ profiler_buffer,
    bool prof_on)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int tid = threadIdx.x;

    // Causal conv1d update
    for (int c = block_id * LDG_BLOCK_SIZE + tid; c < CONV_DIM; c += num_blocks * LDG_BLOCK_SIZE) {
        float mixed[CONV_KERNEL_SIZE];
        for (int k = 0; k < CONV_KERNEL_SIZE - 1; k++)
            mixed[k] = conv_state[c * (CONV_KERNEL_SIZE - 1) + k];
        mixed[CONV_KERNEL_SIZE - 1] = g_conv_in[c];
        float out = 0.0f;
        for (int k = 0; k < CONV_KERNEL_SIZE; k++)
            out += mixed[k] * __bfloat162float(conv1d_weight[c * CONV_KERNEL_SIZE + k]);
        g_conv_in[c] = ldg_silu(out);
        for (int k = 0; k < CONV_KERNEL_SIZE - 1; k++)
            conv_state[c * (CONV_KERNEL_SIZE - 1) + k] = mixed[k + 1];
    }

    // Split conv_out into q, k, v
    for (int i = block_id * LDG_BLOCK_SIZE + tid; i < KEY_DIM; i += num_blocks * LDG_BLOCK_SIZE) {
        g_q[i] = g_conv_in[i];
        g_k[i] = g_conv_in[KEY_DIM + i];
    }
    for (int i = block_id * LDG_BLOCK_SIZE + tid; i < VALUE_DIM; i += num_blocks * LDG_BLOCK_SIZE)
        g_v[i] = g_conv_in[KEY_DIM * 2 + i];

    // beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)
    for (int h = block_id * LDG_BLOCK_SIZE + tid; h < NUM_V_HEADS; h += num_blocks * LDG_BLOCK_SIZE) {
        g_b[h] = sigmoid(g_b[h]);
        g_a[h] = -expf(a_log[h]) * softplus(g_a[h] + dt_bias[h]);
    }

    // L2 norm on q and k per head (cross-warp reduction required)
    __shared__ float smem_reduce[LDG_NUM_WARPS];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    for (int h = 0; h < NUM_V_HEADS; h++) {
        float *q_head = g_q + h * HEAD_K_DIM;
        float sum_sq = 0.0f;
        for (int i = tid; i < HEAD_K_DIM; i += LDG_BLOCK_SIZE) sum_sq += q_head[i] * q_head[i];
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            float total = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            total = ldg_warp_reduce_sum(total);
            if (lane_id == 0) smem_reduce[0] = total;
        }
        __syncthreads();
        float rnorm = rsqrtf(smem_reduce[0] + 1e-6f);
        for (int i = tid; i < HEAD_K_DIM; i += LDG_BLOCK_SIZE) q_head[i] *= rnorm;
    }
    for (int h = 0; h < NUM_V_HEADS; h++) {
        float *k_head = g_k + h * HEAD_K_DIM;
        float sum_sq = 0.0f;
        for (int i = tid; i < HEAD_K_DIM; i += LDG_BLOCK_SIZE) sum_sq += k_head[i] * k_head[i];
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            float total = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            total = ldg_warp_reduce_sum(total);
            if (lane_id == 0) smem_reduce[0] = total;
        }
        __syncthreads();
        float rnorm = rsqrtf(smem_reduce[0] + 1e-6f);
        for (int i = tid; i < HEAD_K_DIM; i += LDG_BLOCK_SIZE) k_head[i] *= rnorm;
    }

    // Scale q by 1/sqrt(head_k_dim) (triton_recurrent_norm_gated default behavior)
    float q_scale = 1.0f / sqrtf(float(HEAD_K_DIM));
    for (int h = 0; h < NUM_V_HEADS; h++) {
        float *q_head = g_q + h * HEAD_K_DIM;
        for (int i = tid; i < HEAD_K_DIM; i += LDG_BLOCK_SIZE) q_head[i] *= q_scale;
    }

    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

// =============================================================================
// Phase 3: Full -- Flash-decode Attention
//        Linear -- Recurrent state update + output compute
// =============================================================================

__device__ void ldg_attention_full(
    AtomicGridSync &grid,
    const float *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_cache,
    const __nv_bfloat16 *__restrict__ v_cache,
    float *__restrict__ attn_out,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    unsigned int *__restrict__ attn_flag,
    int full_layer_idx,
    uint64_t *__restrict__ profiler_buffer,
    bool prof_on)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const int ATTN_BLOCKS = NUM_Q_HEADS;

    // Non-attention blocks wait on attn_flag
    if (block_id >= ATTN_BLOCKS) {
        sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
        if (threadIdx.x == 0) {
            unsigned int target = (unsigned int)(ATTN_BLOCKS * (full_layer_idx + 1));
            volatile unsigned int *vf = (volatile unsigned int *)attn_flag;
            while (*vf < target) {}
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
        }
        __syncthreads();
        sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
        return;
    }

    sm_profiler_event_start(profiler_buffer, SM_PROF_ATTN_COMPUTE, prof_on);

    __shared__ float s_max_score[LDG_NUM_WARPS];
    __shared__ float s_sum_exp[LDG_NUM_WARPS];
    __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

    int heads_per_block = (NUM_Q_HEADS + ATTN_BLOCKS - 1) / ATTN_BLOCKS;
    int head_start = block_id * heads_per_block;
    int head_end = min(head_start + heads_per_block, NUM_Q_HEADS);

    for (int qh = head_start; qh < head_end; qh++) {
        int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
        // q has interleaved [query, gate] per head. We only use query part for attention.
        const float *q_head = q + qh * HEAD_DIM * 2;
        float *out_head = attn_out + qh * HEAD_DIM;

        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float out_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        int q_idx = lane_id * 8;
        float q_local[8];
        q_local[0] = q_head[q_idx + 0];
        q_local[1] = q_head[q_idx + 1];
        q_local[2] = q_head[q_idx + 2];
        q_local[3] = q_head[q_idx + 3];
        q_local[4] = q_head[q_idx + 4];
        q_local[5] = q_head[q_idx + 5];
        q_local[6] = q_head[q_idx + 6];
        q_local[7] = q_head[q_idx + 7];

        for (int pos = warp_id; pos < cache_len; pos += LDG_NUM_WARPS) {
            const __nv_bfloat16 *k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16 *v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            float score = 0.0f;
            uint4 k_u4 = __ldg(reinterpret_cast<const uint4 *>(k_pos + q_idx));
            __nv_bfloat16 *k_ptr = reinterpret_cast<__nv_bfloat16 *>(&k_u4);
            score += q_local[0] * __bfloat162float(k_ptr[0]) +
                     q_local[1] * __bfloat162float(k_ptr[1]) +
                     q_local[2] * __bfloat162float(k_ptr[2]) +
                     q_local[3] * __bfloat162float(k_ptr[3]) +
                     q_local[4] * __bfloat162float(k_ptr[4]) +
                     q_local[5] * __bfloat162float(k_ptr[5]) +
                     q_local[6] * __bfloat162float(k_ptr[6]) +
                     q_local[7] * __bfloat162float(k_ptr[7]);
            score = ldg_warp_reduce_sum(score) * attn_scale;
            score = __shfl_sync(0xffffffff, score, 0);

            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            sum_exp = sum_exp * exp_diff + expf(score - max_score);
            float weight = expf(score - max_score);

            uint4 v_u4 = __ldg(reinterpret_cast<const uint4 *>(v_pos + q_idx));
            __nv_bfloat16 *v_ptr = reinterpret_cast<__nv_bfloat16 *>(&v_u4);
            out_acc[0] = out_acc[0] * exp_diff + weight * __bfloat162float(v_ptr[0]);
            out_acc[1] = out_acc[1] * exp_diff + weight * __bfloat162float(v_ptr[1]);
            out_acc[2] = out_acc[2] * exp_diff + weight * __bfloat162float(v_ptr[2]);
            out_acc[3] = out_acc[3] * exp_diff + weight * __bfloat162float(v_ptr[3]);
            out_acc[4] = out_acc[4] * exp_diff + weight * __bfloat162float(v_ptr[4]);
            out_acc[5] = out_acc[5] * exp_diff + weight * __bfloat162float(v_ptr[5]);
            out_acc[6] = out_acc[6] * exp_diff + weight * __bfloat162float(v_ptr[6]);
            out_acc[7] = out_acc[7] * exp_diff + weight * __bfloat162float(v_ptr[7]);
        }

        if (lane_id == 0) {
            s_max_score[warp_id] = max_score;
            s_sum_exp[warp_id] = sum_exp;
        }
        int out_base = lane_id * 8;
        s_out_acc[warp_id][out_base + 0] = out_acc[0];
        s_out_acc[warp_id][out_base + 1] = out_acc[1];
        s_out_acc[warp_id][out_base + 2] = out_acc[2];
        s_out_acc[warp_id][out_base + 3] = out_acc[3];
        s_out_acc[warp_id][out_base + 4] = out_acc[4];
        s_out_acc[warp_id][out_base + 5] = out_acc[5];
        s_out_acc[warp_id][out_base + 6] = out_acc[6];
        s_out_acc[warp_id][out_base + 7] = out_acc[7];
        __syncthreads();

        if (warp_id == 0) {
            float global_max = s_max_score[0];
            for (int w = 1; w < LDG_NUM_WARPS; w++)
                if (s_max_score[w] > -INFINITY)
                    global_max = fmaxf(global_max, s_max_score[w]);

            float total_sum_exp = 0.0f;
            float final_out[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for (int w = 0; w < LDG_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {
                    float sc = expf(s_max_score[w] - global_max);
                    total_sum_exp += s_sum_exp[w] * sc;
                    int base = lane_id * 8;
                    final_out[0] += s_out_acc[w][base + 0] * sc;
                    final_out[1] += s_out_acc[w][base + 1] * sc;
                    final_out[2] += s_out_acc[w][base + 2] * sc;
                    final_out[3] += s_out_acc[w][base + 3] * sc;
                    final_out[4] += s_out_acc[w][base + 4] * sc;
                    final_out[5] += s_out_acc[w][base + 5] * sc;
                    final_out[6] += s_out_acc[w][base + 6] * sc;
                    final_out[7] += s_out_acc[w][base + 7] * sc;
                }
            }
            int base = lane_id * 8;
            out_head[base + 0] = final_out[0] / total_sum_exp;
            out_head[base + 1] = final_out[1] / total_sum_exp;
            out_head[base + 2] = final_out[2] / total_sum_exp;
            out_head[base + 3] = final_out[3] / total_sum_exp;
            out_head[base + 4] = final_out[4] / total_sum_exp;
            out_head[base + 5] = final_out[5] / total_sum_exp;
            out_head[base + 6] = final_out[6] / total_sum_exp;
            out_head[base + 7] = final_out[7] / total_sum_exp;
        }
        __syncthreads();
    }

    sm_profiler_event_end(profiler_buffer, SM_PROF_ATTN_COMPUTE, prof_on);
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    if (threadIdx.x == 0)
        atomicAdd(attn_flag, 1);
    if (threadIdx.x == 0) {
        unsigned int target = (unsigned int)(ATTN_BLOCKS * (full_layer_idx + 1));
        volatile unsigned int *vf = (volatile unsigned int *)attn_flag;
        while (*vf < target) {}
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    __syncthreads();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

__device__ void ldg_attention_linear(
    AtomicGridSync &grid,
    float *__restrict__ g_q,
    float *__restrict__ g_k,
    float *__restrict__ g_v,
    float *__restrict__ g_b,
    float *__restrict__ g_a_decay,  // exp(g) per head
    float *__restrict__ attn_out,
    float *__restrict__ recurrent_state,
    uint64_t *__restrict__ profiler_buffer,
    bool prof_on)
{
    int tid = threadIdx.x;
    int num_blocks = gridDim.x;

    // Step 1: decay recurrent state: S *= exp(g)
    for (int idx = blockIdx.x * LDG_BLOCK_SIZE + tid;
         idx < NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM;
         idx += num_blocks * LDG_BLOCK_SIZE) {
        int h = idx / (HEAD_K_DIM * HEAD_V_DIM);
        int rem = idx % (HEAD_K_DIM * HEAD_V_DIM);
        float *S = recurrent_state + h * HEAD_K_DIM * HEAD_V_DIM;
        S[rem] *= expf(g_a_decay[h]);
    }

    // Grid sync so all blocks finish decay before computing delta
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

    // Step 2: compute delta = (v - S @ k) * beta for all heads
    for (int j = blockIdx.x * LDG_BLOCK_SIZE + tid;
         j < NUM_V_HEADS * HEAD_V_DIM;
         j += num_blocks * LDG_BLOCK_SIZE) {
        int h = j / HEAD_V_DIM;
        int jj = j % HEAD_V_DIM;
        float *S = recurrent_state + h * HEAD_K_DIM * HEAD_V_DIM;
        float *k_head = g_k + h * HEAD_K_DIM;
        float *v_head = g_v + h * HEAD_V_DIM;
        float kv_mem = 0.0f;
        for (int i = 0; i < HEAD_K_DIM; i++)
            kv_mem += S[i * HEAD_V_DIM + jj] * k_head[i];
        v_head[jj] = (v_head[jj] - kv_mem) * g_b[h];
    }

    // Step 3: update S += k * delta (outer product)
    for (int idx = blockIdx.x * LDG_BLOCK_SIZE + tid;
         idx < NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM;
         idx += num_blocks * LDG_BLOCK_SIZE) {
        int h = idx / (HEAD_K_DIM * HEAD_V_DIM);
        int rem = idx % (HEAD_K_DIM * HEAD_V_DIM);
        int i = rem / HEAD_V_DIM;
        int j = rem % HEAD_V_DIM;
        float *S = recurrent_state + h * HEAD_K_DIM * HEAD_V_DIM;
        float *k_head = g_k + h * HEAD_K_DIM;
        float *v_head = g_v + h * HEAD_V_DIM;
        S[rem] += k_head[i] * v_head[j];
    }

    // Step 4: compute output = S @ q
    for (int j = blockIdx.x * LDG_BLOCK_SIZE + tid;
         j < NUM_V_HEADS * HEAD_V_DIM;
         j += num_blocks * LDG_BLOCK_SIZE) {
        int h = j / HEAD_V_DIM;
        int jj = j % HEAD_V_DIM;
        float *S = recurrent_state + h * HEAD_K_DIM * HEAD_V_DIM;
        float *q_head = g_q + h * HEAD_K_DIM;
        float *out_head = attn_out + h * HEAD_V_DIM;
        float sum = 0.0f;
        for (int i = 0; i < HEAD_K_DIM; i++)
            sum += S[i * HEAD_V_DIM + jj] * q_head[i];
        out_head[jj] = sum;
    }

    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

// =============================================================================
// Phase 4: Out Proj + Post Norm + MLP
// =============================================================================

__device__ void ldg_out_mlp(
    AtomicGridSync &grid,
    const LayerWeights &__restrict__ w,
    float *__restrict__ g_q,           // full: contains gate in second half
    float *__restrict__ g_z,           // linear: z gate
    float *__restrict__ attn_out,      // full: Q_SIZE; linear: VALUE_DIM
    float *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_mlp_intermediate,
    __nv_bfloat16 *__restrict__ hidden_out,
    uint64_t *__restrict__ profiler_buffer,
    bool prof_on)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // -------------------------------------------------------------------------
    // Attention-type-specific output projection
    // -------------------------------------------------------------------------
    if (w.layer_type == 0) {
        // Full attention: gated O-proj
        // Apply sigmoid(gate) elementwise to attn_out
        // gate is in g_q's second half per head: g_q[h*HEAD_DIM*2 + HEAD_DIM + i]
        for (int h = block_id; h < NUM_Q_HEADS; h += num_blocks) {
            for (int i = threadIdx.x; i < HEAD_DIM; i += LDG_BLOCK_SIZE) {
                float gate_val = sigmoid(g_q[h * HEAD_DIM * 2 + HEAD_DIM + i]);
                attn_out[h * HEAD_DIM + i] *= gate_val;
            }
        }
        __syncthreads();

        // Cache gated attn_out in shared memory for O-proj
        __shared__ float s_attn[SMEM_PAD_SIZE(Q_SIZE)];
        for (int i = threadIdx.x; i < Q_SIZE; i += LDG_BLOCK_SIZE)
            s_attn[SMEM_PAD_IDX(i)] = attn_out[i];
        __syncthreads();

        // O-proj: distributed matvec [HIDDEN_SIZE, Q_SIZE]
        int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
        int hid_start = block_id * hid_per_block;
        int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

        for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
            int m = m_base + warp_id;
            if (m < hid_end) {
                const __nv_bfloat16 *o_row = w.full_o_proj + m * Q_SIZE;
                float sum = 0.0f;
#pragma unroll 4
                for (int k = lane_id * 8; k < Q_SIZE; k += WARP_SIZE * 8) {
                    uint4 w_u4 = __ldg(reinterpret_cast<const uint4 *>(o_row + k));
                    __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);
                    int pk = SMEM_PAD_IDX(k);
                    sum += __bfloat162float(w_ptr[0]) * s_attn[pk + 0] +
                           __bfloat162float(w_ptr[1]) * s_attn[pk + 1] +
                           __bfloat162float(w_ptr[2]) * s_attn[pk + 2] +
                           __bfloat162float(w_ptr[3]) * s_attn[pk + 3] +
                           __bfloat162float(w_ptr[4]) * s_attn[pk + 4] +
                           __bfloat162float(w_ptr[5]) * s_attn[pk + 5] +
                           __bfloat162float(w_ptr[6]) * s_attn[pk + 6] +
                           __bfloat162float(w_ptr[7]) * s_attn[pk + 7];
                }
                sum = ldg_warp_reduce_sum(sum);
                if (lane_id == 0)
                    g_activations[m] = sum + g_residual[m];
            }
        }
    } else {
        // Linear attention: RMSNormGated then out_proj
        __shared__ float smem_reduce[LDG_NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        // RMSNorm per head on attn_out, then multiply by silu(g_z)
        for (int h = block_id; h < NUM_V_HEADS; h += num_blocks) {
            float *head_out = attn_out + h * HEAD_V_DIM;
            float *head_z = g_z + h * HEAD_V_DIM;

            float sum_sq = 0.0f;
            for (int i = threadIdx.x; i < HEAD_V_DIM; i += LDG_BLOCK_SIZE)
                sum_sq += head_out[i] * head_out[i];
            sum_sq = ldg_warp_reduce_sum(sum_sq);
            if (lane_id == 0) smem_reduce[warp_id] = sum_sq;
            __syncthreads();
            if (warp_id == 0) {
                float total = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
                total = ldg_warp_reduce_sum(total);
                if (lane_id == 0) smem_reduce[0] = total;
            }
            __syncthreads();
            float rstd = rsqrtf(smem_reduce[0] / float(HEAD_V_DIM) + LDG_RMS_EPS);

            for (int i = threadIdx.x; i < HEAD_V_DIM; i += LDG_BLOCK_SIZE) {
                float norm_w = __bfloat162float(__ldg(w.linear_norm_weight + i));
                head_out[i] = head_out[i] * rstd * norm_w * ldg_silu(head_z[i]);
            }
        }
        __syncthreads();

        // Cache gated/normed attn_out in shared memory for out_proj
        __shared__ float s_attn[SMEM_PAD_SIZE(VALUE_DIM)];
        for (int i = threadIdx.x; i < VALUE_DIM; i += LDG_BLOCK_SIZE)
            s_attn[SMEM_PAD_IDX(i)] = attn_out[i];
        __syncthreads();

        // Out-proj: distributed matvec [HIDDEN_SIZE, VALUE_DIM]
        int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
        int hid_start = block_id * hid_per_block;
        int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

        for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
            int m = m_base + warp_id;
            if (m < hid_end) {
                const __nv_bfloat16 *out_row = w.linear_out_proj + m * VALUE_DIM;
                float sum = 0.0f;
#pragma unroll 4
                for (int k = lane_id * 8; k < VALUE_DIM; k += WARP_SIZE * 8) {
                    uint4 w_u4 = __ldg(reinterpret_cast<const uint4 *>(out_row + k));
                    __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);
                    int pk = SMEM_PAD_IDX(k);
                    sum += __bfloat162float(w_ptr[0]) * s_attn[pk + 0] +
                           __bfloat162float(w_ptr[1]) * s_attn[pk + 1] +
                           __bfloat162float(w_ptr[2]) * s_attn[pk + 2] +
                           __bfloat162float(w_ptr[3]) * s_attn[pk + 3] +
                           __bfloat162float(w_ptr[4]) * s_attn[pk + 4] +
                           __bfloat162float(w_ptr[5]) * s_attn[pk + 5] +
                           __bfloat162float(w_ptr[6]) * s_attn[pk + 6] +
                           __bfloat162float(w_ptr[7]) * s_attn[pk + 7];
                }
                sum = ldg_warp_reduce_sum(sum);
                if (lane_id == 0)
                    g_activations[m] = sum + g_residual[m];
            }
        }
    }

    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

    // -------------------------------------------------------------------------
    // Common: Post-attention RMSNorm (all blocks redundant into shared mem)
    // -------------------------------------------------------------------------
    __shared__ float s_post_normalized[SMEM_PAD_SIZE(HIDDEN_SIZE)];
    {
        __shared__ float smem_reduce[LDG_NUM_WARPS];
        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = g_activations[i];
            s_post_normalized[SMEM_PAD_IDX(i)] = v;
            local_sum_sq += v * v;
        }
        local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0)
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float wt = __bfloat162float(__ldg(w.post_norm + i));
            s_post_normalized[SMEM_PAD_IDX(i)] = s_post_normalized[SMEM_PAD_IDX(i)] * rstd * (1.0f + wt);
        }
        __syncthreads();
    }

    // Distributed residual update
    {
        int res_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
        int res_start = block_id * res_per_block;
        int res_end = min(res_start + res_per_block, HIDDEN_SIZE);
        for (int i = res_start + threadIdx.x; i < res_end; i += LDG_BLOCK_SIZE)
            g_residual[i] = g_activations[i];
    }

    // -------------------------------------------------------------------------
    // MLP: Gate + Up + SiLU (all blocks participate)
    // -------------------------------------------------------------------------
    {
        int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
        int int_start = block_id * int_per_block;
        int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

        for (int m_base = int_start; m_base < int_end; m_base += LDG_NUM_WARPS) {
            int m = m_base + warp_id;
            if (m < int_end) {
                const __nv_bfloat16 *gate_row = w.gate_proj + m * HIDDEN_SIZE;
                const __nv_bfloat16 *up_row = w.up_proj + m * HIDDEN_SIZE;

                float gate_sum = 0.0f, up_sum = 0.0f;
#pragma unroll 4
                for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                    uint4 g_u4 = __ldg(reinterpret_cast<const uint4 *>(gate_row + k));
                    uint4 u_u4 = __ldg(reinterpret_cast<const uint4 *>(up_row + k));
                    __nv_bfloat16 *g_ptr = reinterpret_cast<__nv_bfloat16 *>(&g_u4);
                    __nv_bfloat16 *u_ptr = reinterpret_cast<__nv_bfloat16 *>(&u_u4);
                    int pk = SMEM_PAD_IDX(k);
                    float s0 = s_post_normalized[pk + 0], s1 = s_post_normalized[pk + 1],
                          s2 = s_post_normalized[pk + 2], s3 = s_post_normalized[pk + 3],
                          s4 = s_post_normalized[pk + 4], s5 = s_post_normalized[pk + 5],
                          s6 = s_post_normalized[pk + 6], s7 = s_post_normalized[pk + 7];
                    gate_sum += __bfloat162float(g_ptr[0]) * s0 +
                                __bfloat162float(g_ptr[1]) * s1 +
                                __bfloat162float(g_ptr[2]) * s2 +
                                __bfloat162float(g_ptr[3]) * s3 +
                                __bfloat162float(g_ptr[4]) * s4 +
                                __bfloat162float(g_ptr[5]) * s5 +
                                __bfloat162float(g_ptr[6]) * s6 +
                                __bfloat162float(g_ptr[7]) * s7;
                    up_sum += __bfloat162float(u_ptr[0]) * s0 +
                              __bfloat162float(u_ptr[1]) * s1 +
                              __bfloat162float(u_ptr[2]) * s2 +
                              __bfloat162float(u_ptr[3]) * s3 +
                              __bfloat162float(u_ptr[4]) * s4 +
                              __bfloat162float(u_ptr[5]) * s5 +
                              __bfloat162float(u_ptr[6]) * s6 +
                              __bfloat162float(u_ptr[7]) * s7;
                }
                gate_sum = ldg_warp_reduce_sum(gate_sum);
                up_sum = ldg_warp_reduce_sum(up_sum);
                if (lane_id == 0)
                    g_mlp_intermediate[m] = ldg_silu(gate_sum) * up_sum;
            }
        }
    }

    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

    // -------------------------------------------------------------------------
    // Down projection + residual (read mlp_intermediate from global memory)
    // -------------------------------------------------------------------------
    {
        int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
        int hid_start = block_id * hid_per_block;
        int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

        for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
            int m = m_base + warp_id;
            if (m < hid_end) {
                const __nv_bfloat16 *down_row = w.down_proj + m * INTERMEDIATE_SIZE;
                float sum = 0.0f;
#pragma unroll 4
                for (int k = lane_id * 8; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 8) {
                    uint4 d_u4 = __ldg(reinterpret_cast<const uint4 *>(down_row + k));
                    __nv_bfloat16 *d_ptr = reinterpret_cast<__nv_bfloat16 *>(&d_u4);
                    sum += __bfloat162float(d_ptr[0]) * g_mlp_intermediate[k + 0] +
                           __bfloat162float(d_ptr[1]) * g_mlp_intermediate[k + 1] +
                           __bfloat162float(d_ptr[2]) * g_mlp_intermediate[k + 2] +
                           __bfloat162float(d_ptr[3]) * g_mlp_intermediate[k + 3] +
                           __bfloat162float(d_ptr[4]) * g_mlp_intermediate[k + 4] +
                           __bfloat162float(d_ptr[5]) * g_mlp_intermediate[k + 5] +
                           __bfloat162float(d_ptr[6]) * g_mlp_intermediate[k + 6] +
                           __bfloat162float(d_ptr[7]) * g_mlp_intermediate[k + 7];
                }
                sum = ldg_warp_reduce_sum(sum);
                if (lane_id == 0)
                    hidden_out[m] = __float2bfloat16(sum + g_residual[m]);
            }
        }
    }

    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

// =============================================================================
// Main persistent decode kernel (all layers fused)
// =============================================================================

__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1)
    ldg_decode_kernel(
        int input_token_id,
        const __nv_bfloat16 *__restrict__ embed_weight,
        const LayerWeights *__restrict__ layer_weights,
        const __nv_bfloat16 *__restrict__ final_norm_weight,
        const __nv_bfloat16 *__restrict__ cos_table,
        const __nv_bfloat16 *__restrict__ sin_table,
        __nv_bfloat16 *__restrict__ k_cache,
        __nv_bfloat16 *__restrict__ v_cache,
        float *__restrict__ conv_state,
        float *__restrict__ recurrent_state,
        __nv_bfloat16 *__restrict__ hidden_buffer,
        float *__restrict__ g_activations,
        float *__restrict__ g_residual,
        float *__restrict__ g_q,
        float *__restrict__ g_k,
        float *__restrict__ g_v,
        float *__restrict__ g_z,
        float *__restrict__ g_b,
        float *__restrict__ g_a,
        float *__restrict__ g_attn_out,
        float *__restrict__ g_mlp_intermediate,
        float *__restrict__ g_normalized,
        const int *__restrict__ full_layer_idx,
        const int *__restrict__ linear_layer_idx,
        int num_layers,
        int position,
        int cache_len,
        int max_seq_len,
        float attn_scale,
        uint64_t *__restrict__ profiler_buffer,
        unsigned int *__restrict__ barrier_counter,
        unsigned int *__restrict__ barrier_sense,
        unsigned int *__restrict__ kv_flag,
        unsigned int *__restrict__ attn_flag)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    bool prof_on = (profiler_buffer != nullptr);

    // Bootstrap: reset flags and synchronize all blocks via atomics
    if (block_id == 0 && threadIdx.x == 0) {
        atomicExch(kv_flag, 0u);
        atomicExch(attn_flag, 0u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        unsigned int arrived = atomicAdd(barrier_counter, 1);
        if (arrived == (unsigned int)gridDim.x - 1) {
            *barrier_counter = 0;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            atomicAdd(barrier_sense, 1);
        } else {
            volatile unsigned int *vs = (volatile unsigned int *)barrier_sense;
            while (*vs == 0) {}
        }
    }
    __syncthreads();
    AtomicGridSync grid{barrier_counter, barrier_sense, (unsigned int)gridDim.x, 1};

    // Embedding lookup
    sm_profiler_event_start(profiler_buffer, SM_PROF_EMBEDDING, prof_on);
    const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    for (int i = block_id * LDG_BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += num_blocks * LDG_BLOCK_SIZE)
        hidden_buffer[i] = __ldg(embed_row + i);
    sm_profiler_event_end(profiler_buffer, SM_PROF_EMBEDDING, prof_on);
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

    int kv_cache_head_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;
    int conv_state_stride = CONV_DIM * (CONV_KERNEL_SIZE - 1);
    int recurrent_state_stride = NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const LayerWeights &w = layer_weights[layer];

        sm_profiler_event_start(profiler_buffer, SM_PROF_QKV_PROJ, prof_on);
        ldg_norm_and_proj(grid, hidden_buffer, w,
                          g_activations, g_residual,
                          g_q, g_k, g_v, g_z, g_b, g_a,
                          profiler_buffer, prof_on);
        sm_profiler_event_end(profiler_buffer, SM_PROF_QKV_PROJ, prof_on);

        if (w.layer_type == 0) {
            int full_idx = full_layer_idx[layer];
            __nv_bfloat16 *layer_k = k_cache + full_idx * kv_cache_head_stride;
            __nv_bfloat16 *layer_v = v_cache + full_idx * kv_cache_head_stride;

            sm_profiler_event_start(profiler_buffer, SM_PROF_QK_NORM_ROPE, prof_on);
            ldg_prep_full(grid, g_q, g_k, g_v,
                          w.full_q_norm, w.full_k_norm,
                          cos_table, sin_table,
                          layer_k, layer_v,
                          position, max_seq_len,
                          kv_flag, full_idx,
                          profiler_buffer, prof_on);
            sm_profiler_event_end(profiler_buffer, SM_PROF_QK_NORM_ROPE, prof_on);

            ldg_attention_full(grid, g_q, layer_k, layer_v, g_attn_out,
                               cache_len, max_seq_len, attn_scale,
                               attn_flag, full_idx,
                               profiler_buffer, prof_on);
        } else {
            int linear_idx = linear_layer_idx[layer];
            float *layer_conv = conv_state + linear_idx * conv_state_stride;
            float *layer_recurrent = recurrent_state + linear_idx * recurrent_state_stride;

            ldg_prep_linear(grid, g_q, g_q, g_k, g_v, g_b, g_a, g_z,
                            w.linear_conv1d_weight,
                            w.linear_dt_bias, w.linear_a_log,
                            layer_conv,
                            profiler_buffer, prof_on);

            ldg_attention_linear(grid, g_q, g_k, g_v, g_b, g_a,
                                 g_attn_out, layer_recurrent,
                                 profiler_buffer, prof_on);
        }

        // Sync all blocks before Phase 4 (ensure attn_out is fully written)
        sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
        grid.sync();
        sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

        sm_profiler_event_start(profiler_buffer, SM_PROF_O_PROJ_MLP, prof_on);
        ldg_out_mlp(grid, w, g_q, g_z, g_attn_out,
                    g_residual, g_activations, g_mlp_intermediate,
                    hidden_buffer,
                    profiler_buffer, prof_on);
        sm_profiler_event_end(profiler_buffer, SM_PROF_O_PROJ_MLP, prof_on);
    }

    // Final RMSNorm (block 0 only)
    sm_profiler_event_start(profiler_buffer, SM_PROF_FINAL_NORM, prof_on);
    if (block_id == 0) {
        __shared__ float smem_reduce[LDG_NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = __bfloat162float(hidden_buffer[i]);
            g_activations[i] = v;
            local_sum_sq += v * v;
        }
        local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
        if (lane_id == 0)
            smem_reduce[warp_id] = local_sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0)
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * (1.0f + wt);
        }
    }
    sm_profiler_event_end(profiler_buffer, SM_PROF_FINAL_NORM, prof_on);
}

// =============================================================================
// LM Head -- full logits (for correctness testing)
// =============================================================================

__global__ void ldg_lm_head_logits(
    const float *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ weight,
    float *__restrict__ logits)
{
    __shared__ float s_hidden[HIDDEN_SIZE];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE)
        s_hidden[i] = hidden[i];
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

    for (int m = row_start + warp_id; m < row_end; m += LDG_LM_BLOCK_SIZE / WARP_SIZE) {
        const __nv_bfloat16 *w_row = weight + m * HIDDEN_SIZE;
        float sum = 0.0f;
#pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2 *>(w_row + k));
            __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u2);
            sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k + 1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k + 2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k + 3];
        }
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0)
            logits[m] = sum;
    }
}

// =============================================================================
// LM Head -- two-phase argmax (for token output)
// =============================================================================

__global__ void ldg_lm_head_phase1(
    const float *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ weight,
    float *__restrict__ block_max_vals,
    int *__restrict__ block_max_idxs)
{
    __shared__ float s_hidden[HIDDEN_SIZE];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE)
        s_hidden[i] = hidden[i];
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int m = row_start + warp_id; m < row_end; m += LDG_LM_BLOCK_SIZE / WARP_SIZE) {
        const __nv_bfloat16 *w_row = weight + m * HIDDEN_SIZE;
        float sum = 0.0f;
#pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2 *>(w_row + k));
            __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u2);
            sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k + 1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k + 2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k + 3];
        }
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0 && sum > local_max) {
            local_max = sum;
            local_max_idx = m;
        }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_idx[LDG_LM_BLOCK_SIZE / WARP_SIZE];
    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int max_idx = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;
        for (int off = WARP_SIZE / 2; off > 0; off /= 2) {
            float ov = __shfl_down_sync(0xffffffff, max_val, off);
            int oi = __shfl_down_sync(0xffffffff, max_idx, off);
            if (ov > max_val) {
                max_val = ov;
                max_idx = oi;
            }
        }
        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = max_val;
            block_max_idxs[blockIdx.x] = max_idx;
        }
    }
}

__global__ void ldg_lm_head_phase2(
    const float *__restrict__ block_max_vals,
    const int *__restrict__ block_max_idxs,
    int *__restrict__ output_token,
    int num_blocks)
{
    __shared__ float s_max_vals[1024];
    __shared__ int s_max_idxs[1024];
    int tid = threadIdx.x;
    float local_max = -INFINITY;
    int local_idx = -1;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > local_max) {
            local_max = v;
            local_idx = block_max_idxs[i];
        }
    }
    s_max_vals[tid] = local_max;
    s_max_idxs[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_max_vals[tid + s] > s_max_vals[tid]) {
            s_max_vals[tid] = s_max_vals[tid + s];
            s_max_idxs[tid] = s_max_idxs[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        *output_token = s_max_idxs[0];
}

// =============================================================================
// Host launch functions
// =============================================================================

extern "C" void launch_qwen3_5_ldg_decode(
    int input_token_id,
    int *output_token_id,
    const void *embed_weight,
    const void *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight,
    const void *cos_table,
    const void *sin_table,
    void *k_cache,
    void *v_cache,
    void *conv_state,
    void *recurrent_state,
    void *hidden_buffer,
    void *g_activations,
    void *g_residual,
    void *g_q,
    void *g_k,
    void *g_v,
    void *g_z,
    void *g_b,
    void *g_a,
    void *g_attn_out,
    void *g_mlp_intermediate,
    void *g_normalized,
    const int *full_layer_idx,
    const int *linear_layer_idx,
    int num_blocks,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    uint64_t *profiler_buffer,
    cudaStream_t stream)
{
    static unsigned int *d_barrier_counter = nullptr;
    static unsigned int *d_barrier_sense = nullptr;
    static unsigned int *d_kv_flag = nullptr;
    static unsigned int *d_attn_flag = nullptr;
    static bool init = false;
    if (!init) {
        cudaMalloc(&d_barrier_counter, sizeof(unsigned int));
        cudaMalloc(&d_barrier_sense, sizeof(unsigned int));
        cudaMalloc(&d_kv_flag, sizeof(unsigned int));
        cudaMalloc(&d_attn_flag, sizeof(unsigned int));
        init = true;
    }
    cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_barrier_sense, 0, sizeof(unsigned int), stream);

    ldg_decode_kernel<<<dim3(num_blocks), dim3(LDG_BLOCK_SIZE), 0, stream>>>(
        input_token_id,
        (const __nv_bfloat16 *)embed_weight,
        (const LayerWeights *)layer_weights,
        (const __nv_bfloat16 *)final_norm_weight,
        (const __nv_bfloat16 *)cos_table,
        (const __nv_bfloat16 *)sin_table,
        (__nv_bfloat16 *)k_cache,
        (__nv_bfloat16 *)v_cache,
        (float *)conv_state,
        (float *)recurrent_state,
        (__nv_bfloat16 *)hidden_buffer,
        (float *)g_activations,
        (float *)g_residual,
        (float *)g_q,
        (float *)g_k,
        (float *)g_v,
        (float *)g_z,
        (float *)g_b,
        (float *)g_a,
        (float *)g_attn_out,
        (float *)g_mlp_intermediate,
        (float *)g_normalized,
        full_layer_idx,
        linear_layer_idx,
        num_layers,
        position,
        cache_len,
        max_seq_len,
        attn_scale,
        profiler_buffer,
        d_barrier_counter,
        d_barrier_sense,
        d_kv_flag,
        d_attn_flag);

    // LM head argmax
    static float *d_block_max_vals = nullptr;
    static int *d_block_max_idxs = nullptr;
    if (!d_block_max_vals) {
        cudaMalloc(&d_block_max_vals, LDG_LM_NUM_BLOCKS * sizeof(float));
        cudaMalloc(&d_block_max_idxs, LDG_LM_NUM_BLOCKS * sizeof(int));
    }

    // LM head argmax
    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        d_block_max_vals,
        d_block_max_idxs);

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        d_block_max_vals,
        d_block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS);
}

extern "C" void launch_qwen3_5_ldg_decode_with_logits(
    int input_token_id,
    int *output_token_id,
    float *logits_output,
    const void *embed_weight,
    const void *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight,
    const void *cos_table,
    const void *sin_table,
    void *k_cache,
    void *v_cache,
    void *conv_state,
    void *recurrent_state,
    void *hidden_buffer,
    void *g_activations,
    void *g_residual,
    void *g_q,
    void *g_k,
    void *g_v,
    void *g_z,
    void *g_b,
    void *g_a,
    void *g_attn_out,
    void *g_mlp_intermediate,
    void *g_normalized,
    const int *full_layer_idx,
    const int *linear_layer_idx,
    int num_blocks,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    uint64_t *profiler_buffer,
    cudaStream_t stream)
{
    static unsigned int *d_barrier_counter = nullptr;
    static unsigned int *d_barrier_sense = nullptr;
    static unsigned int *d_kv_flag = nullptr;
    static unsigned int *d_attn_flag = nullptr;
    static bool init = false;
    if (!init) {
        cudaMalloc(&d_barrier_counter, sizeof(unsigned int));
        cudaMalloc(&d_barrier_sense, sizeof(unsigned int));
        cudaMalloc(&d_kv_flag, sizeof(unsigned int));
        cudaMalloc(&d_attn_flag, sizeof(unsigned int));
        init = true;
    }
    cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_barrier_sense, 0, sizeof(unsigned int), stream);

    ldg_decode_kernel<<<dim3(num_blocks), dim3(LDG_BLOCK_SIZE), 0, stream>>>(
        input_token_id,
        (const __nv_bfloat16 *)embed_weight,
        (const LayerWeights *)layer_weights,
        (const __nv_bfloat16 *)final_norm_weight,
        (const __nv_bfloat16 *)cos_table,
        (const __nv_bfloat16 *)sin_table,
        (__nv_bfloat16 *)k_cache,
        (__nv_bfloat16 *)v_cache,
        (float *)conv_state,
        (float *)recurrent_state,
        (__nv_bfloat16 *)hidden_buffer,
        (float *)g_activations,
        (float *)g_residual,
        (float *)g_q,
        (float *)g_k,
        (float *)g_v,
        (float *)g_z,
        (float *)g_b,
        (float *)g_a,
        (float *)g_attn_out,
        (float *)g_mlp_intermediate,
        (float *)g_normalized,
        full_layer_idx,
        linear_layer_idx,
        num_layers,
        position,
        cache_len,
        max_seq_len,
        attn_scale,
        profiler_buffer,
        d_barrier_counter,
        d_barrier_sense,
        d_kv_flag,
        d_attn_flag);

    // Full logits
    ldg_lm_head_logits<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        logits_output);

    // Also compute argmax for greedy output
    static float *d_block_max_vals = nullptr;
    static int *d_block_max_idxs = nullptr;
    if (!d_block_max_vals) {
        cudaMalloc(&d_block_max_vals, LDG_LM_NUM_BLOCKS * sizeof(float));
        cudaMalloc(&d_block_max_idxs, LDG_LM_NUM_BLOCKS * sizeof(int));
    }

    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        d_block_max_vals,
        d_block_max_idxs);

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        d_block_max_vals,
        d_block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS);
}
