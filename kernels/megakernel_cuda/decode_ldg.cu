/**
 * Fused Decode with __ldg() cached reads  (ported from MegaQwen)
 *
 * Full transformer decode pipeline for Qwen3-0.6B in a single persistent
 * kernel with AtomicGridSync + flag-based partial barriers:
 * embed -> 28x(RMSNorm+QKV -> QKNorm+RoPE+Cache -> Attention ->
 * OProj+PostNorm+MLP) -> FinalNorm -> LM Head.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "sm_profiler.h"

// sm-profiler event IDs
#define SM_PROF_EMBEDDING        0
#define SM_PROF_QKV_PROJ         1
#define SM_PROF_QK_NORM_ROPE     2
#define SM_PROF_ATTN_COMPUTE     3
#define SM_PROF_ATTN_PREFETCH    4
#define SM_PROF_O_PROJ_MLP       5
#define SM_PROF_FINAL_NORM       6
#define SM_PROF_GRID_SYNC        7
#define SM_PROF_NUM_EVENTS       8

// =============================================================================
// Configuration & model constants
// =============================================================================

constexpr int WARP_SIZE = 32;

constexpr int HIDDEN_SIZE      = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS      = 16;
constexpr int NUM_KV_HEADS     = 8;
constexpr int HEAD_DIM         = 128;
constexpr int Q_SIZE  = NUM_Q_HEADS  * HEAD_DIM;   // 2048
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM;   // 1024

constexpr int LDG_BLOCK_SIZE = 256;
constexpr int LDG_NUM_WARPS  = LDG_BLOCK_SIZE / WARP_SIZE;  // 8
constexpr float LDG_RMS_EPS  = 1e-6f;

// Shared memory bank conflict padding: 1 pad per 8 elements (stride 9)
// Reduces 8-way bank conflicts to ≤2-way in float4 matvec reads.
// Safe for float4: each lane's 8 consecutive elements never cross a pad slot.
#define SMEM_PAD_IDX(i) ((i) + (i) / 8)
#define SMEM_PAD_SIZE(n) ((n) + (n) / 8)

// LM head
constexpr int LDG_LM_NUM_BLOCKS  = 1184;
constexpr int LDG_LM_BLOCK_SIZE  = 256;
constexpr int LDG_VOCAB_SIZE     = 151936;

struct LDGLayerWeights {
    const __nv_bfloat16* input_layernorm_weight;
    const __nv_bfloat16* q_proj_weight;
    const __nv_bfloat16* k_proj_weight;
    const __nv_bfloat16* v_proj_weight;
    const __nv_bfloat16* q_norm_weight;
    const __nv_bfloat16* k_norm_weight;
    const __nv_bfloat16* o_proj_weight;
    const __nv_bfloat16* post_attn_layernorm_weight;
    const __nv_bfloat16* gate_proj_weight;
    const __nv_bfloat16* up_proj_weight;
    const __nv_bfloat16* down_proj_weight;
};

// =============================================================================
// Atomic barrier for persistent kernel (replaces cooperative grid.sync())
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
        while (*vgen <= my_gen) {
        }
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

// Forward declaration for prefetch (defined in Phase 3)
__device__ void ldg_prefetch_weights_l2(
    const __nv_bfloat16* __restrict__ weights, int num_elements
);

// =============================================================================
// Phase 1: RMSNorm + QKV Projection
// =============================================================================

__device__ void ldg_matvec_qkv(
    AtomicGridSync& grid,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ v_weight,
    float* __restrict__ g_normalized,
    float* __restrict__ g_residual,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    uint64_t* __restrict__ profiler_buffer,
    bool prof_on
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Redundant RMSNorm: every block computes its own copy in shared memory
    // to avoid a grid.sync() between norm and QKV projection
    __shared__ float s_normalized[SMEM_PAD_SIZE(HIDDEN_SIZE)];
    {
        __shared__ float smem_reduce[LDG_NUM_WARPS];

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float v = __bfloat162float(__ldg(input + i));
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
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(norm_weight + i));
            s_normalized[SMEM_PAD_IDX(i)] = s_normalized[SMEM_PAD_IDX(i)] * rstd * w;
        }
        __syncthreads();
    }

    // Distributed residual write: each block writes its share
    {
        int res_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
        int res_start = block_id * res_per_block;
        int res_end = min(res_start + res_per_block, HIDDEN_SIZE);
        for (int i = res_start + threadIdx.x; i < res_end; i += LDG_BLOCK_SIZE)
            g_residual[i] = __bfloat162float(__ldg(input + i));
    }

    // QKV projection with vec4 __ldg (reads from shared memory, no grid.sync needed)
    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

    for (int m_base = row_start; m_base < row_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const __nv_bfloat16* weight_row;
            float* output_ptr;

            if (m < Q_SIZE) {
                weight_row = q_weight + m * HIDDEN_SIZE;
                output_ptr = q_out + m;
            } else if (m < Q_SIZE + KV_SIZE) {
                weight_row = k_weight + (m - Q_SIZE) * HIDDEN_SIZE;
                output_ptr = k_out + (m - Q_SIZE);
            } else {
                weight_row = v_weight + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
                output_ptr = v_out + (m - Q_SIZE - KV_SIZE);
            }

            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);
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
    // Prefetch gate+up weights during upcoming grid.sync wait
    {
        constexpr int gate_total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
        constexpr int up_total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
        constexpr int combined = gate_total + up_total;
        int elems_per_block = (combined + num_blocks - 1) / num_blocks;
        int my_start = block_id * elems_per_block;
        int my_end = min(my_start + elems_per_block, combined);
        for (int pos = my_start; pos < my_end; ) {
            if (pos < gate_total) {
                int chunk = min(my_end, gate_total) - pos;
                ldg_prefetch_weights_l2(gate_weight + pos, chunk);
                pos += chunk;
            } else {
                int off = pos - gate_total;
                int chunk = my_end - pos;
                ldg_prefetch_weights_l2(up_weight + off, chunk);
                pos += chunk;
            }
        }
    }
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

// =============================================================================
// Phase 2: QK Norm + RoPE + KV Cache Write
// =============================================================================

__device__ void ldg_qk_norm_rope_cache(
    AtomicGridSync& grid,
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int position,
    int max_seq_len,
    const __nv_bfloat16* __restrict__ o_weight,
    unsigned int* __restrict__ kv_flag,
    unsigned int* __restrict__ attn_flag,
    int layer_idx,
    uint64_t* __restrict__ profiler_buffer,
    bool prof_on
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16* cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_table + position * HEAD_DIM;

    // Q heads
    int q_heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int q_head_start = block_id * q_heads_per_block;
    int q_head_end = min(q_head_start + q_heads_per_block, NUM_Q_HEADS);

    for (int h = q_head_start + warp_id; h < q_head_end; h += LDG_NUM_WARPS) {
        float* q_head = q + h * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
            sum_sq += q_head[i] * q_head[i];
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float q_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
            q_local[j] = q_head[i] * scale * __bfloat162float(__ldg(q_norm_weight + i));

        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));
            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);
            if (i < HEAD_DIM/2)
                q_head[i] = q_local[j] * cos_v - pair_v * sin_v;
            else
                q_head[i] = pair_v * sin_v + q_local[j] * cos_v;
        }
    }

    // K heads + cache write
    int k_heads_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
    int k_head_start = block_id * k_heads_per_block;
    int k_head_end = min(k_head_start + k_heads_per_block, NUM_KV_HEADS);

    for (int h = k_head_start + warp_id; h < k_head_end; h += LDG_NUM_WARPS) {
        float* k_head = k + h * HEAD_DIM;
        const float* v_head = v + h * HEAD_DIM;
        __nv_bfloat16* k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __nv_bfloat16* v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
            sum_sq += k_head[i] * k_head[i];
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float k_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
            k_local[j] = k_head[i] * scale * __bfloat162float(__ldg(k_norm_weight + i));

        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));
            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);

            float k_final;
            if (i < HEAD_DIM/2)
                k_final = k_local[j] * cos_v - pair_v * sin_v;
            else
                k_final = pair_v * sin_v + k_local[j] * cos_v;
            k_head[i] = k_final;
            k_cache_head[i] = __float2bfloat16(k_final);
            v_cache_head[i] = __float2bfloat16(v_head[i]);
        }
    }
    // Prefetch O weight during idle DRAM window
    {
        constexpr int o_total = Q_SIZE * HIDDEN_SIZE;
        int elems_per_block = (o_total + num_blocks - 1) / num_blocks;
        int my_offset = block_id * elems_per_block;
        if (my_offset < o_total)
            ldg_prefetch_weights_l2(o_weight + my_offset, min(elems_per_block, o_total - my_offset));
    }
    // kv_flag partial barrier: attention blocks (0..ATTN_BLOCKS-1) synchronize
    // using a monotonic atomic counter so all Q/K/V writes are visible before attention
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    const int ATTN_BLOCKS = NUM_Q_HEADS; // 16
    if (block_id < ATTN_BLOCKS) {
        __syncthreads();  // finish QK norm + cache writes within block
        if (threadIdx.x == 0) {
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            atomicAdd(kv_flag, 1);
            // Wait for all attention blocks to have completed their writes
            unsigned int target = (unsigned int)(ATTN_BLOCKS * (layer_idx + 1));
            volatile unsigned int* vf = (volatile unsigned int*)kv_flag;
            while (*vf < target) {}
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
        }
        __syncthreads();
    }
    // Blocks >= ATTN_BLOCKS: skip entirely (they don't do attention)
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

// =============================================================================
// Phase 3: Flash-decoding Attention (+ L2 weight prefetch by idle blocks)
// =============================================================================

__device__ void ldg_prefetch_weights_l2(
    const __nv_bfloat16* __restrict__ weights, int num_elements
) {
    // Use PTX prefetch.global.L2::evict_last to hint L2 to keep data persistent.
    // Each prefetch touches one cache line (128 bytes = 64 bf16 elements).
    // 256 threads × 128 bytes = 32 KB per iteration = 256 cache lines.
    // Stride: LDG_BLOCK_SIZE cache lines = 256 * 128 bytes = 32768 bytes per step.
    const char* base = reinterpret_cast<const char*>(weights);
    int total_bytes = num_elements * 2;  // bf16 = 2 bytes
    for (int offset = threadIdx.x * 128; offset < total_bytes; offset += LDG_BLOCK_SIZE * 128) {
        asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(base + offset));
    }
}

__device__ void ldg_attention(
    AtomicGridSync& grid,
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ attn_out,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    const __nv_bfloat16* __restrict__ o_weight,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    const __nv_bfloat16* __restrict__ down_weight,
    unsigned int* __restrict__ attn_flag,
    int layer_idx,
    uint64_t* __restrict__ profiler_buffer,
    bool prof_on
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const int ATTN_BLOCKS = NUM_Q_HEADS;

    // Idle blocks prefetch current layer's o_proj, gate, up, down weights into L2
    if (block_id >= ATTN_BLOCKS) {
        sm_profiler_event_start(profiler_buffer, SM_PROF_ATTN_PREFETCH, prof_on);
        int prefetch_block_id = block_id - ATTN_BLOCKS;
        int num_prefetch_blocks = num_blocks - ATTN_BLOCKS;
        // Split idle blocks into 4 groups: O, gate, up, down
        constexpr int num_groups = 4;
        int group = prefetch_block_id * num_groups / num_prefetch_blocks;
        int grp_start = group * num_prefetch_blocks / num_groups;
        int adj = prefetch_block_id - grp_start;
        int grp_count = ((group + 1) * num_prefetch_blocks / num_groups) - grp_start;
        if (group == 0) {
            int total = Q_SIZE * HIDDEN_SIZE;
            int elems = total / grp_count;
            int offset = adj * elems;
            if (offset < total)
                ldg_prefetch_weights_l2(o_weight + offset, min(elems, total - offset));
        } else if (group == 1) {
            int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
            int elems = total / grp_count;
            int offset = adj * elems;
            if (offset < total)
                ldg_prefetch_weights_l2(gate_weight + offset, min(elems, total - offset));
        } else if (group == 2) {
            int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
            int elems = total / grp_count;
            int offset = adj * elems;
            if (offset < total)
                ldg_prefetch_weights_l2(up_weight + offset, min(elems, total - offset));
        } else {
            int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
            int elems = total / grp_count;
            int offset = adj * elems;
            if (offset < total)
                ldg_prefetch_weights_l2(down_weight + offset, min(elems, total - offset));
        }
        sm_profiler_event_end(profiler_buffer, SM_PROF_ATTN_PREFETCH, prof_on);
        // Non-attention blocks: wait for attention completion via attn_flag
        sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
        if (threadIdx.x == 0) {
            unsigned int target = (unsigned int)(ATTN_BLOCKS * (layer_idx + 1));
            volatile unsigned int* vf = (volatile unsigned int*)attn_flag;
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
        const float* q_head = q + qh * HEAD_DIM;
        float* out_head = attn_out + qh * HEAD_DIM;

        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Cache Q head in registers using contiguous vec4 pattern
        int q_idx = lane_id * 4;
        float q_local[4];
        q_local[0] = q_head[q_idx + 0];
        q_local[1] = q_head[q_idx + 1];
        q_local[2] = q_head[q_idx + 2];
        q_local[3] = q_head[q_idx + 3];

        for (int pos = warp_id; pos < cache_len; pos += LDG_NUM_WARPS) {
            const __nv_bfloat16* k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16* v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            // Vectorized uint2 load for K cache
            float score = 0.0f;
            uint2 k_u2 = __ldg(reinterpret_cast<const uint2*>(k_pos + q_idx));
            __nv_bfloat16* k_ptr = reinterpret_cast<__nv_bfloat16*>(&k_u2);
            score += q_local[0] * __bfloat162float(k_ptr[0]) +
                     q_local[1] * __bfloat162float(k_ptr[1]) +
                     q_local[2] * __bfloat162float(k_ptr[2]) +
                     q_local[3] * __bfloat162float(k_ptr[3]);
            score = ldg_warp_reduce_sum(score) * attn_scale;
            score = __shfl_sync(0xffffffff, score, 0);

            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            sum_exp = sum_exp * exp_diff + expf(score - max_score);
            float weight = expf(score - max_score);

            // Vectorized uint2 load for V cache
            uint2 v_u2 = __ldg(reinterpret_cast<const uint2*>(v_pos + q_idx));
            __nv_bfloat16* v_ptr = reinterpret_cast<__nv_bfloat16*>(&v_u2);
            out_acc[0] = out_acc[0] * exp_diff + weight * __bfloat162float(v_ptr[0]);
            out_acc[1] = out_acc[1] * exp_diff + weight * __bfloat162float(v_ptr[1]);
            out_acc[2] = out_acc[2] * exp_diff + weight * __bfloat162float(v_ptr[2]);
            out_acc[3] = out_acc[3] * exp_diff + weight * __bfloat162float(v_ptr[3]);
        }

        if (lane_id == 0) {
            s_max_score[warp_id] = max_score;
            s_sum_exp[warp_id] = sum_exp;
        }
        int out_base = lane_id * 4;
        s_out_acc[warp_id][out_base + 0] = out_acc[0];
        s_out_acc[warp_id][out_base + 1] = out_acc[1];
        s_out_acc[warp_id][out_base + 2] = out_acc[2];
        s_out_acc[warp_id][out_base + 3] = out_acc[3];
        __syncthreads();

        // Warp 0 combines partial results
        if (warp_id == 0) {
            float global_max = s_max_score[0];
            for (int w = 1; w < LDG_NUM_WARPS; w++)
                if (s_max_score[w] > -INFINITY)
                    global_max = fmaxf(global_max, s_max_score[w]);

            float total_sum_exp = 0.0f;
            float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int w = 0; w < LDG_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {
                    float sc = expf(s_max_score[w] - global_max);
                    total_sum_exp += s_sum_exp[w] * sc;
                    int base = lane_id * 4;
                    final_out[0] += s_out_acc[w][base + 0] * sc;
                    final_out[1] += s_out_acc[w][base + 1] * sc;
                    final_out[2] += s_out_acc[w][base + 2] * sc;
                    final_out[3] += s_out_acc[w][base + 3] * sc;
                }
            }
            int base = lane_id * 4;
            out_head[base + 0] = final_out[0] / total_sum_exp;
            out_head[base + 1] = final_out[1] / total_sum_exp;
            out_head[base + 2] = final_out[2] / total_sum_exp;
            out_head[base + 3] = final_out[3] / total_sum_exp;
        }
        __syncthreads();
    }
    sm_profiler_event_end(profiler_buffer, SM_PROF_ATTN_COMPUTE, prof_on);
    // Attention blocks signal completion via attn_flag
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    if (threadIdx.x == 0)
        atomicAdd(attn_flag, 1);
    // Attention blocks also wait for all attention to complete before proceeding
    if (threadIdx.x == 0) {
        unsigned int target = (unsigned int)(ATTN_BLOCKS * (layer_idx + 1));
        volatile unsigned int* vf = (volatile unsigned int*)attn_flag;
        while (*vf < target) {}
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    __syncthreads();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
}

// =============================================================================
// Phase 4: O Projection + Post-Norm + MLP
// =============================================================================

__device__ void ldg_o_proj_postnorm_mlp(
    AtomicGridSync& grid,
    const __nv_bfloat16* __restrict__ o_weight,
    const __nv_bfloat16* __restrict__ post_norm_weight,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    const __nv_bfloat16* __restrict__ down_weight,
    const float* __restrict__ attn_out,
    float* __restrict__ g_residual,
    float* __restrict__ g_activations,
    float* __restrict__ g_mlp_intermediate,
    __nv_bfloat16* __restrict__ hidden_out,
    const __nv_bfloat16* __restrict__ next_q_weight,
    const __nv_bfloat16* __restrict__ next_k_weight,
    const __nv_bfloat16* __restrict__ next_v_weight,
    bool is_last_layer,
    uint64_t* __restrict__ profiler_buffer,
    bool prof_on
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Cache attn_out in shared memory to avoid repeated global reads
    __shared__ float s_attn[SMEM_PAD_SIZE(Q_SIZE)];
    for (int i = threadIdx.x; i < Q_SIZE; i += LDG_BLOCK_SIZE)
        s_attn[SMEM_PAD_IDX(i)] = attn_out[i];
    __syncthreads();

    // O projection + residual
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < hid_end) {
            const __nv_bfloat16* o_row = o_weight + m * Q_SIZE;
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < Q_SIZE; k += WARP_SIZE * 8) {
                uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(o_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);
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
            if (lane_id == 0) g_activations[m] = sum + g_residual[m];
        }
    }
    // Prefetch down weight during upcoming O proj grid.sync wait
    {
        constexpr int down_total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
        int elems_per_block = (down_total + num_blocks - 1) / num_blocks;
        int my_offset = block_id * elems_per_block;
        if (my_offset < down_total)
            ldg_prefetch_weights_l2(down_weight + my_offset, min(elems_per_block, down_total - my_offset));
    }
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

    // Post-attention RMSNorm: redundant across all blocks into shared memory
    // Eliminates grid.sync between norm and gate/up projection
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
            if (lane_id == 0) smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float w = __bfloat162float(__ldg(post_norm_weight + i));
            s_post_normalized[SMEM_PAD_IDX(i)] = s_post_normalized[SMEM_PAD_IDX(i)] * rstd * w;
        }
        __syncthreads();
    }

    // Distributed residual update: each block writes its share of g_residual
    {
        int res_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
        int res_start = block_id * res_per_block;
        int res_end = min(res_start + res_per_block, HIDDEN_SIZE);
        for (int i = res_start + threadIdx.x; i < res_end; i += LDG_BLOCK_SIZE)
            g_residual[i] = g_activations[i];
    }

    // Gate + Up + SiLU: all blocks participate
    {
        int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
        int int_start = block_id * int_per_block;
        int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

        for (int m_base = int_start; m_base < int_end; m_base += LDG_NUM_WARPS) {
            int m = m_base + warp_id;
            if (m < int_end) {
                const __nv_bfloat16* gate_row = gate_weight + m * HIDDEN_SIZE;
                const __nv_bfloat16* up_row   = up_weight   + m * HIDDEN_SIZE;

                float gate_sum = 0.0f, up_sum = 0.0f;
                #pragma unroll 4
                for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                    uint4 g_u4 = __ldg(reinterpret_cast<const uint4*>(gate_row + k));
                    uint4 u_u4 = __ldg(reinterpret_cast<const uint4*>(up_row + k));
                    __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(&g_u4);
                    __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(&u_u4);
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
                up_sum   = ldg_warp_reduce_sum(up_sum);
                if (lane_id == 0)
                    g_mlp_intermediate[m] = ldg_silu(gate_sum) * up_sum;
            }
        }
    }
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

    // Cache mlp_intermediate in shared memory for down projection
    __shared__ float s_mlp[SMEM_PAD_SIZE(INTERMEDIATE_SIZE)];
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += LDG_BLOCK_SIZE)
        s_mlp[SMEM_PAD_IDX(i)] = g_mlp_intermediate[i];
    __syncthreads();

    // Down projection + residual
    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < hid_end) {
            const __nv_bfloat16* down_row = down_weight + m * INTERMEDIATE_SIZE;
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 8) {
                uint4 d_u4 = __ldg(reinterpret_cast<const uint4*>(down_row + k));
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u4);
                int pk = SMEM_PAD_IDX(k);
                sum += __bfloat162float(d_ptr[0]) * s_mlp[pk + 0] +
                       __bfloat162float(d_ptr[1]) * s_mlp[pk + 1] +
                       __bfloat162float(d_ptr[2]) * s_mlp[pk + 2] +
                       __bfloat162float(d_ptr[3]) * s_mlp[pk + 3] +
                       __bfloat162float(d_ptr[4]) * s_mlp[pk + 4] +
                       __bfloat162float(d_ptr[5]) * s_mlp[pk + 5] +
                       __bfloat162float(d_ptr[6]) * s_mlp[pk + 6] +
                       __bfloat162float(d_ptr[7]) * s_mlp[pk + 7];
            }
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = __float2bfloat16(sum + g_residual[m]);
        }
    }
    // Prefetch next layer QKV during upcoming down proj grid.sync wait
    if (!is_last_layer) {
        constexpr int q_total = Q_SIZE * HIDDEN_SIZE;
        constexpr int k_total = KV_SIZE * HIDDEN_SIZE;
        constexpr int v_total = KV_SIZE * HIDDEN_SIZE;
        constexpr int combined = q_total + k_total + v_total;
        int elems_per_block = (combined + num_blocks - 1) / num_blocks;
        int my_start = block_id * elems_per_block;
        int my_end = min(my_start + elems_per_block, combined);
        for (int pos = my_start; pos < my_end; ) {
            if (pos < q_total) {
                int chunk = min(my_end, q_total) - pos;
                ldg_prefetch_weights_l2(next_q_weight + pos, chunk);
                pos += chunk;
            } else if (pos < q_total + k_total) {
                int off = pos - q_total;
                int chunk = min(my_end, q_total + k_total) - pos;
                ldg_prefetch_weights_l2(next_k_weight + off, chunk);
                pos += chunk;
            } else {
                int off = pos - q_total - k_total;
                int chunk = my_end - pos;
                ldg_prefetch_weights_l2(next_v_weight + off, chunk);
                pos += chunk;
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
    const __nv_bfloat16* __restrict__ embed_weight,
    const LDGLayerWeights* __restrict__ layer_weights,
    const __nv_bfloat16* __restrict__ final_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ hidden_buffer,
    float* __restrict__ g_activations,
    float* __restrict__ g_residual,
    float* __restrict__ g_q,
    float* __restrict__ g_k,
    float* __restrict__ g_v,
    float* __restrict__ g_attn_out,
    float* __restrict__ g_mlp_intermediate,
    float* __restrict__ g_normalized,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    uint64_t* __restrict__ profiler_buffer,
    unsigned int* __restrict__ barrier_counter,
    unsigned int* __restrict__ barrier_sense,
    unsigned int* __restrict__ kv_flag,
    unsigned int* __restrict__ attn_flag
) {
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
            volatile unsigned int* vs = (volatile unsigned int*)barrier_sense;
            while (*vs == 0) {}
        }
    }
    __syncthreads();
    AtomicGridSync grid{barrier_counter, barrier_sense, (unsigned int)gridDim.x, 1};

    // Embedding lookup
    sm_profiler_event_start(profiler_buffer, SM_PROF_EMBEDDING, prof_on);
    const __nv_bfloat16* embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
    for (int i = block_id * LDG_BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += num_blocks * LDG_BLOCK_SIZE)
        hidden_buffer[i] = __ldg(embed_row + i);
    sm_profiler_event_end(profiler_buffer, SM_PROF_EMBEDDING, prof_on);
    sm_profiler_event_start(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);
    grid.sync();
    sm_profiler_event_end(profiler_buffer, SM_PROF_GRID_SYNC, prof_on);

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const LDGLayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k = k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v = v_cache + layer * kv_cache_layer_stride;

        sm_profiler_event_start(profiler_buffer, SM_PROF_QKV_PROJ, prof_on);
        ldg_matvec_qkv(grid, hidden_buffer, w.input_layernorm_weight,
                        w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
                        g_activations, g_residual, g_q, g_k, g_v,
                        w.gate_proj_weight, w.up_proj_weight,
                        profiler_buffer, prof_on);
        sm_profiler_event_end(profiler_buffer, SM_PROF_QKV_PROJ, prof_on);

        sm_profiler_event_start(profiler_buffer, SM_PROF_QK_NORM_ROPE, prof_on);
        ldg_qk_norm_rope_cache(grid, g_q, g_k, g_v,
                                w.q_norm_weight, w.k_norm_weight,
                                cos_table, sin_table,
                                layer_k, layer_v,
                                position, max_seq_len,
                                w.o_proj_weight,
                                kv_flag, attn_flag, layer,
                                profiler_buffer, prof_on);
        sm_profiler_event_end(profiler_buffer, SM_PROF_QK_NORM_ROPE, prof_on);

        ldg_attention(grid, g_q, layer_k, layer_v, g_attn_out,
                       cache_len, max_seq_len, attn_scale,
                       w.o_proj_weight, w.gate_proj_weight, w.up_proj_weight,
                       w.down_proj_weight,
                       attn_flag, layer,
                       profiler_buffer, prof_on);

        sm_profiler_event_start(profiler_buffer, SM_PROF_O_PROJ_MLP, prof_on);
        bool is_last_layer = (layer == num_layers - 1);
        ldg_o_proj_postnorm_mlp(grid,
                                 w.o_proj_weight, w.post_attn_layernorm_weight,
                                 w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
                                 g_attn_out, g_residual, g_activations, g_mlp_intermediate,
                                 hidden_buffer,
                                 is_last_layer ? nullptr : layer_weights[layer + 1].q_proj_weight,
                                 is_last_layer ? nullptr : layer_weights[layer + 1].k_proj_weight,
                                 is_last_layer ? nullptr : layer_weights[layer + 1].v_proj_weight,
                                 is_last_layer,
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
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * wt;
        }
    }
    sm_profiler_event_end(profiler_buffer, SM_PROF_FINAL_NORM, prof_on);
}

// =============================================================================
// LM Head – full logits (for correctness testing)
// =============================================================================

__global__ void ldg_lm_head_logits(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ logits
) {
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
        const __nv_bfloat16* w_row = weight + m * HIDDEN_SIZE;
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);
            sum += __bfloat162float(w_ptr[0]) * s_hidden[k]   +
                   __bfloat162float(w_ptr[1]) * s_hidden[k+1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k+2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k+3];
        }
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0) logits[m] = sum;
    }
}

// =============================================================================
// LM Head – two-phase argmax (for token output)
// =============================================================================

__global__ void ldg_lm_head_phase1(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
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
        const __nv_bfloat16* w_row = weight + m * HIDDEN_SIZE;
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
            __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);
            sum += __bfloat162float(w_ptr[0]) * s_hidden[k]   +
                   __bfloat162float(w_ptr[1]) * s_hidden[k+1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k+2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k+3];
        }
        sum = ldg_warp_reduce_sum(sum);
        if (lane_id == 0 && sum > local_max) { local_max = sum; local_max_idx = m; }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int   warp_idx[LDG_LM_BLOCK_SIZE / WARP_SIZE];
    if (lane_id == 0) { warp_max[warp_id] = local_max; warp_idx[warp_id] = local_max_idx; }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int   max_idx = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;
        for (int off = WARP_SIZE / 2; off > 0; off /= 2) {
            float ov = __shfl_down_sync(0xffffffff, max_val, off);
            int   oi = __shfl_down_sync(0xffffffff, max_idx, off);
            if (ov > max_val) { max_val = ov; max_idx = oi; }
        }
        if (lane_id == 0) {
            block_max_vals[blockIdx.x] = max_val;
            block_max_idxs[blockIdx.x] = max_idx;
        }
    }
}

__global__ void ldg_lm_head_phase2(
    const float* __restrict__ block_max_vals,
    const int* __restrict__ block_max_idxs,
    int* __restrict__ output_token,
    int num_blocks
) {
    __shared__ float s_max_vals[1024];
    __shared__ int   s_max_idxs[1024];
    int tid = threadIdx.x;
    float local_max = -INFINITY;
    int   local_idx = -1;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > local_max) { local_max = v; local_idx = block_max_idxs[i]; }
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
    if (tid == 0) *output_token = s_max_idxs[0];
}

// =============================================================================
// Launch function – decode + argmax only (no full logits)
// =============================================================================

extern "C" void launch_ldg_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_blocks,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    uint64_t* profiler_buffer,
    cudaStream_t stream
) {
    // Static device memory for atomic barriers
    static unsigned int* d_barrier_counter = nullptr;
    static unsigned int* d_barrier_sense = nullptr;
    static unsigned int* d_kv_flag = nullptr;
    static unsigned int* d_attn_flag = nullptr;
    static bool barrier_init = false;
    if (!barrier_init) {
        cudaMalloc(&d_barrier_counter, sizeof(unsigned int));
        cudaMalloc(&d_barrier_sense, sizeof(unsigned int));
        cudaMalloc(&d_kv_flag, sizeof(unsigned int));
        cudaMalloc(&d_attn_flag, sizeof(unsigned int));
        barrier_init = true;
    }
    // Reset barrier state before every kernel launch
    cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_barrier_sense, 0, sizeof(unsigned int), stream);

    ldg_decode_kernel<<<dim3(num_blocks), dim3(LDG_BLOCK_SIZE), 0, stream>>>(
        input_token_id,
        (const __nv_bfloat16*)embed_weight,
        (const LDGLayerWeights*)layer_weights,
        (const __nv_bfloat16*)final_norm_weight,
        (const __nv_bfloat16*)cos_table,
        (const __nv_bfloat16*)sin_table,
        (__nv_bfloat16*)k_cache,
        (__nv_bfloat16*)v_cache,
        (__nv_bfloat16*)hidden_buffer,
        (float*)g_activations,
        (float*)g_residual,
        (float*)g_q,
        (float*)g_k,
        (float*)g_v,
        (float*)g_attn_out,
        (float*)g_mlp_intermediate,
        (float*)g_normalized,
        num_layers,
        position,
        cache_len,
        max_seq_len,
        attn_scale,
        profiler_buffer,
        d_barrier_counter,
        d_barrier_sense,
        d_kv_flag,
        d_attn_flag
    );

    // Argmax phase 1 + 2 (no full logits)
    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS
    );
}

// =============================================================================
// Launch function – decode + full logits + argmax
// =============================================================================

extern "C" void launch_ldg_decode_with_logits(
    int input_token_id,
    int* output_token_id,
    float* logits_output,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_blocks,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    uint64_t* profiler_buffer,
    cudaStream_t stream
) {
    // Static device memory for atomic barriers
    static unsigned int* d_barrier_counter = nullptr;
    static unsigned int* d_barrier_sense = nullptr;
    static unsigned int* d_kv_flag = nullptr;
    static unsigned int* d_attn_flag = nullptr;
    static bool barrier_init = false;
    if (!barrier_init) {
        cudaMalloc(&d_barrier_counter, sizeof(unsigned int));
        cudaMalloc(&d_barrier_sense, sizeof(unsigned int));
        cudaMalloc(&d_kv_flag, sizeof(unsigned int));
        cudaMalloc(&d_attn_flag, sizeof(unsigned int));
        barrier_init = true;
    }
    // Reset barrier state before every kernel launch
    cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_barrier_sense, 0, sizeof(unsigned int), stream);

    ldg_decode_kernel<<<dim3(num_blocks), dim3(LDG_BLOCK_SIZE), 0, stream>>>(
        input_token_id,
        (const __nv_bfloat16*)embed_weight,
        (const LDGLayerWeights*)layer_weights,
        (const __nv_bfloat16*)final_norm_weight,
        (const __nv_bfloat16*)cos_table,
        (const __nv_bfloat16*)sin_table,
        (__nv_bfloat16*)k_cache,
        (__nv_bfloat16*)v_cache,
        (__nv_bfloat16*)hidden_buffer,
        (float*)g_activations,
        (float*)g_residual,
        (float*)g_q,
        (float*)g_k,
        (float*)g_v,
        (float*)g_attn_out,
        (float*)g_mlp_intermediate,
        (float*)g_normalized,
        num_layers,
        position,
        cache_len,
        max_seq_len,
        attn_scale,
        profiler_buffer,
        d_barrier_counter,
        d_barrier_sense,
        d_kv_flag,
        d_attn_flag
    );

    // Full logits
    ldg_lm_head_logits<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        logits_output
    );

    // Argmax phase 1 + 2
    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS
    );
}
