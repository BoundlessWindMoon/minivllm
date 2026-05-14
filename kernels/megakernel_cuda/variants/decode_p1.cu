/**
 * decode_p1.cu — Megakernel + P1: redundant RMSNorm in SMEM.
 *
 * Baseline: ``naive`` (cooperative grid sync, block-0-only RMSNorm broadcast
 * via global memory).
 *
 * P1 change: every block redundantly recomputes RMSNorm into its own SMEM
 * buffer (``s_normalized``).  That removes two cross-block broadcasts per
 * layer:
 *   - pre-attn rmsnorm  -> qkv matvec  (no more grid.sync needed in between)
 *   - post-attn rmsnorm -> gate/up matvec (no more grid.sync needed in between)
 *
 * Net effect: 2 grid.syncs / layer eliminated (8 -> 6).  HW-independent.
 *
 * Other naive choices are preserved:
 *   - cooperative_groups grid sync everywhere
 *   - scalar bf16 weight / KV cache loads
 *   - no SMEM padding, no L2 prefetch
 *
 * Launch ABI (extern "C" launch_ldg_decode[_with_logits]) matches
 * decode_wrapper.cpp byte-for-byte.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include "sm_profiler.h"

namespace cg = cooperative_groups;

// =============================================================================
// Configuration & model constants (must match decode_ldg.cu)
// =============================================================================

constexpr int WARP_SIZE = 32;

constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;   // 2048
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM; // 1024

constexpr int LDG_BLOCK_SIZE = 256;
constexpr int LDG_NUM_WARPS = LDG_BLOCK_SIZE / WARP_SIZE; // 8
constexpr float LDG_RMS_EPS = 1e-6f;

constexpr int LDG_LM_NUM_BLOCKS = 1184;
constexpr int LDG_LM_BLOCK_SIZE = 256;
constexpr int LDG_VOCAB_SIZE = 151936;

struct LDGLayerWeights
{
    const __nv_bfloat16 *input_layernorm_weight;
    const __nv_bfloat16 *q_proj_weight;
    const __nv_bfloat16 *k_proj_weight;
    const __nv_bfloat16 *v_proj_weight;
    const __nv_bfloat16 *q_norm_weight;
    const __nv_bfloat16 *k_norm_weight;
    const __nv_bfloat16 *o_proj_weight;
    const __nv_bfloat16 *post_attn_layernorm_weight;
    const __nv_bfloat16 *gate_proj_weight;
    const __nv_bfloat16 *up_proj_weight;
    const __nv_bfloat16 *down_proj_weight;
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float silu(float x)
{
    return x / (1.0f + expf(-x));
}

// =============================================================================
// Phase 1: RMSNorm (redundant per-block, into SMEM) + QKV projection
// =============================================================================
//
// P1: every block redundantly computes RMSNorm of ``input`` into its own
// ``s_normalized`` SMEM buffer.  No grid sync between rmsnorm and matvec —
// the QKV matvec inner loop reads ``s_normalized[k]`` (SMEM, fast) instead of
// ``g_normalized[k]`` (global, was block-0-broadcast).
//
__device__ void p1_rmsnorm_qkv(
    cg::grid_group &grid,
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ norm_weight,
    const __nv_bfloat16 *__restrict__ q_weight,
    const __nv_bfloat16 *__restrict__ k_weight,
    const __nv_bfloat16 *__restrict__ v_weight,
    float *__restrict__ g_residual,
    float *__restrict__ q_out,
    float *__restrict__ k_out,
    float *__restrict__ v_out)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Distributed residual write – every block writes a slice of input -> float
    {
        int per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
        int s = block_id * per_block;
        int e = min(s + per_block, HIDDEN_SIZE);
        for (int i = s + threadIdx.x; i < e; i += LDG_BLOCK_SIZE)
            g_residual[i] = __bfloat162float(__ldg(input + i));
    }

    // P1: every block redundantly computes RMSNorm into SMEM
    __shared__ float s_normalized[HIDDEN_SIZE];
    __shared__ float smem_reduce[LDG_NUM_WARPS + 1]; // +1 for rstd slot at end
    {
        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
        {
            float v = __bfloat162float(__ldg(input + i));
            local_sum_sq += v * v;
        }
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane_id == 0)
            smem_reduce[warp_id] = local_sum_sq;
        __syncthreads();
        if (warp_id == 0)
        {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                smem_reduce[LDG_NUM_WARPS] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[LDG_NUM_WARPS];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
        {
            float v = __bfloat162float(__ldg(input + i));
            float w = __bfloat162float(__ldg(norm_weight + i));
            s_normalized[i] = v * rstd * w;
        }
        __syncthreads();
    }
    // P1: NO grid.sync between rmsnorm and matvec — every block has its own
    // copy of s_normalized in SMEM.

    // QKV projection – every block does its slice of [Q_SIZE+2*KV_SIZE] rows.
    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

    for (int m_base = row_start; m_base < row_end; m_base += LDG_NUM_WARPS)
    {
        int m = m_base + warp_id;
        if (m >= row_end)
            continue;

        const __nv_bfloat16 *weight_row;
        float *output_ptr;
        if (m < Q_SIZE)
        {
            weight_row = q_weight + m * HIDDEN_SIZE;
            output_ptr = q_out + m;
        }
        else if (m < Q_SIZE + KV_SIZE)
        {
            weight_row = k_weight + (m - Q_SIZE) * HIDDEN_SIZE;
            output_ptr = k_out + (m - Q_SIZE);
        }
        else
        {
            weight_row = v_weight + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
            output_ptr = v_out + (m - Q_SIZE - KV_SIZE);
        }

        float sum = 0.0f;
        for (int k = lane_id; k < HIDDEN_SIZE; k += WARP_SIZE)
        {
            float w = __bfloat162float(__ldg(weight_row + k));
            sum += w * s_normalized[k];
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            *output_ptr = sum;
    }

    grid.sync();
}

// =============================================================================
// Phase 2: QK Norm + RoPE + KV Cache write (identical to naive)
// =============================================================================

__device__ void p1_qk_norm_rope_cache(
    cg::grid_group &grid,
    float *__restrict__ q,
    float *__restrict__ k,
    const float *__restrict__ v,
    const __nv_bfloat16 *__restrict__ q_norm_weight,
    const __nv_bfloat16 *__restrict__ k_norm_weight,
    const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table,
    __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ v_cache,
    int position,
    int max_seq_len)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16 *cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16 *sin_pos = sin_table + position * HEAD_DIM;

    int q_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int q_start = block_id * q_per_block;
    int q_end = min(q_start + q_per_block, NUM_Q_HEADS);

    for (int h = q_start + warp_id; h < q_end; h += LDG_NUM_WARPS)
    {
        float *q_head = q + h * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
            sum_sq += q_head[i] * q_head[i];
        sum_sq = warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float q_local[HEAD_DIM / WARP_SIZE];
#pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
            q_local[j] = q_head[i] * scale * __bfloat162float(__ldg(q_norm_weight + i));

#pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
        {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));
            int pair_offset = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);
            if (i < HEAD_DIM / 2)
                q_head[i] = q_local[j] * cos_v - pair_v * sin_v;
            else
                q_head[i] = pair_v * sin_v + q_local[j] * cos_v;
        }
    }

    int k_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
    int k_start = block_id * k_per_block;
    int k_end = min(k_start + k_per_block, NUM_KV_HEADS);

    for (int h = k_start + warp_id; h < k_end; h += LDG_NUM_WARPS)
    {
        float *k_head = k + h * HEAD_DIM;
        const float *v_head = v + h * HEAD_DIM;
        __nv_bfloat16 *k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __nv_bfloat16 *v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
            sum_sq += k_head[i] * k_head[i];
        sum_sq = warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float k_local[HEAD_DIM / WARP_SIZE];
#pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
            k_local[j] = k_head[i] * scale * __bfloat162float(__ldg(k_norm_weight + i));

#pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
        {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));
            int pair_offset = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);

            float k_final;
            if (i < HEAD_DIM / 2)
                k_final = k_local[j] * cos_v - pair_v * sin_v;
            else
                k_final = pair_v * sin_v + k_local[j] * cos_v;
            k_head[i] = k_final;
            k_cache_head[i] = __float2bfloat16(k_final);
            v_cache_head[i] = __float2bfloat16(v_head[i]);
        }
    }

    grid.sync();
}

// =============================================================================
// Phase 3: Flash-decoding attention (identical to naive)
// =============================================================================

__device__ void p1_attention(
    cg::grid_group &grid,
    const float *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_cache,
    const __nv_bfloat16 *__restrict__ v_cache,
    float *__restrict__ attn_out,
    int cache_len,
    int max_seq_len,
    float attn_scale)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const int ATTN_BLOCKS = NUM_Q_HEADS;

    if (block_id < ATTN_BLOCKS)
    {
        __shared__ float s_max_score[LDG_NUM_WARPS];
        __shared__ float s_sum_exp[LDG_NUM_WARPS];
        __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

        int qh = block_id;
        int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
        const float *q_head = q + qh * HEAD_DIM;
        float *out_head = attn_out + qh * HEAD_DIM;

        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        int q_idx = lane_id * 4;

        for (int pos = warp_id; pos < cache_len; pos += LDG_NUM_WARPS)
        {
            const __nv_bfloat16 *k_pos =
                k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16 *v_pos =
                v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            float score = 0.0f;
            score += q_head[q_idx + 0] * __bfloat162float(__ldg(k_pos + q_idx + 0));
            score += q_head[q_idx + 1] * __bfloat162float(__ldg(k_pos + q_idx + 1));
            score += q_head[q_idx + 2] * __bfloat162float(__ldg(k_pos + q_idx + 2));
            score += q_head[q_idx + 3] * __bfloat162float(__ldg(k_pos + q_idx + 3));
            score = warp_reduce_sum(score) * attn_scale;
            score = __shfl_sync(0xffffffff, score, 0);

            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            sum_exp = sum_exp * exp_diff + expf(score - max_score);
            float weight = expf(score - max_score);

            float v0 = __bfloat162float(__ldg(v_pos + q_idx + 0));
            float v1 = __bfloat162float(__ldg(v_pos + q_idx + 1));
            float v2 = __bfloat162float(__ldg(v_pos + q_idx + 2));
            float v3 = __bfloat162float(__ldg(v_pos + q_idx + 3));
            out_acc[0] = out_acc[0] * exp_diff + weight * v0;
            out_acc[1] = out_acc[1] * exp_diff + weight * v1;
            out_acc[2] = out_acc[2] * exp_diff + weight * v2;
            out_acc[3] = out_acc[3] * exp_diff + weight * v3;
        }

        if (lane_id == 0)
        {
            s_max_score[warp_id] = max_score;
            s_sum_exp[warp_id] = sum_exp;
        }
        int out_base = lane_id * 4;
        s_out_acc[warp_id][out_base + 0] = out_acc[0];
        s_out_acc[warp_id][out_base + 1] = out_acc[1];
        s_out_acc[warp_id][out_base + 2] = out_acc[2];
        s_out_acc[warp_id][out_base + 3] = out_acc[3];
        __syncthreads();

        if (warp_id == 0)
        {
            float global_max = s_max_score[0];
            for (int w = 1; w < LDG_NUM_WARPS; w++)
                if (s_max_score[w] > -INFINITY)
                    global_max = fmaxf(global_max, s_max_score[w]);

            float total_sum_exp = 0.0f;
            float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int w = 0; w < LDG_NUM_WARPS; w++)
            {
                if (s_max_score[w] > -INFINITY)
                {
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
    grid.sync();
}

// =============================================================================
// Phase 4: O proj + residual + post-norm (redundant SMEM) + gate/up + down
// =============================================================================
//
// P1: post-attn RMSNorm is computed redundantly by every block into SMEM, so
// the gate/up matvec can read ``s_post_normalized`` directly without waiting
// for a cross-block broadcast.  The grid.sync between post-norm and gate/up
// is dropped.
//
__device__ void p1_o_proj_postnorm_mlp(
    cg::grid_group &grid,
    const __nv_bfloat16 *__restrict__ o_weight,
    const __nv_bfloat16 *__restrict__ post_norm_weight,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const __nv_bfloat16 *__restrict__ up_weight,
    const __nv_bfloat16 *__restrict__ down_weight,
    const float *__restrict__ attn_out,
    float *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_mlp_intermediate,
    __nv_bfloat16 *__restrict__ hidden_out)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // O projection + residual add
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS)
    {
        int m = m_base + warp_id;
        if (m >= hid_end)
            continue;
        const __nv_bfloat16 *o_row = o_weight + m * Q_SIZE;
        float sum = 0.0f;
        for (int k = lane_id; k < Q_SIZE; k += WARP_SIZE)
        {
            float w = __bfloat162float(__ldg(o_row + k));
            sum += w * attn_out[k];
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            g_activations[m] = sum + g_residual[m];
    }

    grid.sync();

    // P1: redundant per-block post-attn RMSNorm into SMEM
    __shared__ float s_post_normalized[HIDDEN_SIZE];
    __shared__ float smem_reduce[LDG_NUM_WARPS + 1];
    {
        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
        {
            float v = g_activations[i];
            local_sum_sq += v * v;
        }
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane_id == 0)
            smem_reduce[warp_id] = local_sum_sq;
        __syncthreads();
        if (warp_id == 0)
        {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                smem_reduce[LDG_NUM_WARPS] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[LDG_NUM_WARPS];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
        {
            float w = __bfloat162float(__ldg(post_norm_weight + i));
            s_post_normalized[i] = g_activations[i] * rstd * w;
        }
        __syncthreads();
    }

    // Distributed residual update – this block's slice of g_activations becomes
    // the pre-MLP residual.  These writes only need to be visible by the time
    // the post-gate/up grid.sync hits.
    {
        int per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
        int s = block_id * per_block;
        int e = min(s + per_block, HIDDEN_SIZE);
        for (int i = s + threadIdx.x; i < e; i += LDG_BLOCK_SIZE)
            g_residual[i] = g_activations[i];
    }

    // P1: NO grid.sync between rmsnorm and gate/up — every block has its own
    // copy of s_post_normalized in SMEM.

    // Gate + Up + SiLU
    {
        int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
        int int_start = block_id * int_per_block;
        int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

        for (int m_base = int_start; m_base < int_end; m_base += LDG_NUM_WARPS)
        {
            int m = m_base + warp_id;
            if (m >= int_end)
                continue;
            const __nv_bfloat16 *gate_row = gate_weight + m * HIDDEN_SIZE;
            const __nv_bfloat16 *up_row = up_weight + m * HIDDEN_SIZE;
            float gate_sum = 0.0f, up_sum = 0.0f;
            for (int k = lane_id; k < HIDDEN_SIZE; k += WARP_SIZE)
            {
                float gw = __bfloat162float(__ldg(gate_row + k));
                float uw = __bfloat162float(__ldg(up_row + k));
                float x = s_post_normalized[k];
                gate_sum += gw * x;
                up_sum += uw * x;
            }
            gate_sum = warp_reduce_sum(gate_sum);
            up_sum = warp_reduce_sum(up_sum);
            if (lane_id == 0)
                g_mlp_intermediate[m] = silu(gate_sum) * up_sum;
        }
    }

    grid.sync();

    // Down projection + residual
    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS)
    {
        int m = m_base + warp_id;
        if (m >= hid_end)
            continue;
        const __nv_bfloat16 *down_row = down_weight + m * INTERMEDIATE_SIZE;
        float sum = 0.0f;
        for (int k = lane_id; k < INTERMEDIATE_SIZE; k += WARP_SIZE)
        {
            float w = __bfloat162float(__ldg(down_row + k));
            sum += w * g_mlp_intermediate[k];
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            hidden_out[m] = __float2bfloat16(sum + g_residual[m]);
    }

    grid.sync();
}

// =============================================================================
// Main persistent decode kernel
// =============================================================================
//
// P1 drops g_normalized from the layer loop (rmsnorm is now SMEM-only).
// g_normalized is still required for the final RMSNorm before LM head, so we
// reuse the parameter slot for that final write.
//
__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1)
    ldg_decode_kernel(
        int input_token_id,
        const __nv_bfloat16 *__restrict__ embed_weight,
        const LDGLayerWeights *__restrict__ layer_weights,
        const __nv_bfloat16 *__restrict__ final_norm_weight,
        const __nv_bfloat16 *__restrict__ cos_table,
        const __nv_bfloat16 *__restrict__ sin_table,
        __nv_bfloat16 *__restrict__ k_cache,
        __nv_bfloat16 *__restrict__ v_cache,
        __nv_bfloat16 *__restrict__ hidden_buffer,
        float *__restrict__ g_activations,
        float *__restrict__ g_residual,
        float *__restrict__ g_q,
        float *__restrict__ g_k,
        float *__restrict__ g_v,
        float *__restrict__ g_attn_out,
        float *__restrict__ g_mlp_intermediate,
        float *__restrict__ g_normalized,
        int num_layers,
        int position,
        int cache_len,
        int max_seq_len,
        float attn_scale)
{
    cg::grid_group grid = cg::this_grid();

    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    // Embedding lookup – distributed across blocks
    {
        const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
        for (int i = block_id * LDG_BLOCK_SIZE + threadIdx.x;
             i < HIDDEN_SIZE;
             i += num_blocks * LDG_BLOCK_SIZE)
        {
            hidden_buffer[i] = __ldg(embed_row + i);
        }
    }
    grid.sync();

    int kv_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++)
    {
        const LDGLayerWeights &w = layer_weights[layer];
        __nv_bfloat16 *layer_k = k_cache + layer * kv_layer_stride;
        __nv_bfloat16 *layer_v = v_cache + layer * kv_layer_stride;

        p1_rmsnorm_qkv(grid, hidden_buffer, w.input_layernorm_weight,
                       w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
                       g_residual, g_q, g_k, g_v);

        p1_qk_norm_rope_cache(grid, g_q, g_k, g_v,
                              w.q_norm_weight, w.k_norm_weight,
                              cos_table, sin_table,
                              layer_k, layer_v,
                              position, max_seq_len);

        p1_attention(grid, g_q, layer_k, layer_v, g_attn_out,
                     cache_len, max_seq_len, attn_scale);

        p1_o_proj_postnorm_mlp(grid,
                               w.o_proj_weight, w.post_attn_layernorm_weight,
                               w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
                               g_attn_out, g_residual, g_activations,
                               g_mlp_intermediate, hidden_buffer);
    }

    // Final RMSNorm – block 0 only, writes g_normalized for the LM head.
    if (block_id == 0)
    {
        __shared__ float smem_reduce[LDG_NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;

        float local_sum_sq = 0.0f;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
        {
            float v = __bfloat162float(hidden_buffer[i]);
            g_activations[i] = v;
            local_sum_sq += v * v;
        }
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane_id == 0)
            smem_reduce[warp_id] = local_sum_sq;
        __syncthreads();
        if (warp_id == 0)
        {
            float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
        {
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * wt;
        }
    }
}

// =============================================================================
// LM Head – kept identical to decode_ldg.cu / decode_naive.cu
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

    for (int m = row_start + warp_id; m < row_end; m += LDG_LM_BLOCK_SIZE / WARP_SIZE)
    {
        const __nv_bfloat16 *w_row = weight + m * HIDDEN_SIZE;
        float sum = 0.0f;
#pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4)
        {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2 *>(w_row + k));
            __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u2);
            sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k + 1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k + 2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k + 3];
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            logits[m] = sum;
    }
}

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

    for (int m = row_start + warp_id; m < row_end; m += LDG_LM_BLOCK_SIZE / WARP_SIZE)
    {
        const __nv_bfloat16 *w_row = weight + m * HIDDEN_SIZE;
        float sum = 0.0f;
#pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4)
        {
            uint2 w_u2 = __ldg(reinterpret_cast<const uint2 *>(w_row + k));
            __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u2);
            sum += __bfloat162float(w_ptr[0]) * s_hidden[k] +
                   __bfloat162float(w_ptr[1]) * s_hidden[k + 1] +
                   __bfloat162float(w_ptr[2]) * s_hidden[k + 2] +
                   __bfloat162float(w_ptr[3]) * s_hidden[k + 3];
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0 && sum > local_max)
        {
            local_max = sum;
            local_max_idx = m;
        }
    }

    local_max = __shfl_sync(0xffffffff, local_max, 0);
    local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

    __shared__ float warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_idx[LDG_LM_BLOCK_SIZE / WARP_SIZE];
    if (lane_id == 0)
    {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float mv = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int mi = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;
        for (int off = WARP_SIZE / 2; off > 0; off /= 2)
        {
            float ov = __shfl_down_sync(0xffffffff, mv, off);
            int oi = __shfl_down_sync(0xffffffff, mi, off);
            if (ov > mv)
            {
                mv = ov;
                mi = oi;
            }
        }
        if (lane_id == 0)
        {
            block_max_vals[blockIdx.x] = mv;
            block_max_idxs[blockIdx.x] = mi;
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
    for (int i = tid; i < num_blocks; i += blockDim.x)
    {
        float v = block_max_vals[i];
        if (v > local_max)
        {
            local_max = v;
            local_idx = block_max_idxs[i];
        }
    }
    s_max_vals[tid] = local_max;
    s_max_idxs[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && s_max_vals[tid + s] > s_max_vals[tid])
        {
            s_max_vals[tid] = s_max_vals[tid + s];
            s_max_idxs[tid] = s_max_idxs[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        *output_token = s_max_idxs[0];
}

// =============================================================================
// Cooperative kernel launch helper
// =============================================================================

static void launch_p1_decode(
    int input_token_id,
    const __nv_bfloat16 *embed_weight,
    const LDGLayerWeights *layer_weights,
    const __nv_bfloat16 *final_norm_weight,
    const __nv_bfloat16 *cos_table,
    const __nv_bfloat16 *sin_table,
    __nv_bfloat16 *k_cache,
    __nv_bfloat16 *v_cache,
    __nv_bfloat16 *hidden_buffer,
    float *g_activations,
    float *g_residual,
    float *g_q,
    float *g_k,
    float *g_v,
    float *g_attn_out,
    float *g_mlp_intermediate,
    float *g_normalized,
    int num_blocks,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream)
{
    void *args[] = {
        (void *)&input_token_id,
        (void *)&embed_weight,
        (void *)&layer_weights,
        (void *)&final_norm_weight,
        (void *)&cos_table,
        (void *)&sin_table,
        (void *)&k_cache,
        (void *)&v_cache,
        (void *)&hidden_buffer,
        (void *)&g_activations,
        (void *)&g_residual,
        (void *)&g_q,
        (void *)&g_k,
        (void *)&g_v,
        (void *)&g_attn_out,
        (void *)&g_mlp_intermediate,
        (void *)&g_normalized,
        (void *)&num_layers,
        (void *)&position,
        (void *)&cache_len,
        (void *)&max_seq_len,
        (void *)&attn_scale,
    };
    cudaLaunchCooperativeKernel(
        (const void *)ldg_decode_kernel,
        dim3(num_blocks),
        dim3(LDG_BLOCK_SIZE),
        args,
        /*sharedMem*/ 0,
        stream);
}

// =============================================================================
// External C ABI – matches decode_wrapper.cpp
// =============================================================================

extern "C" void launch_ldg_decode(
    int input_token_id,
    int *output_token_id,
    const void *embed_weight,
    const LDGLayerWeights *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight,
    const void *cos_table,
    const void *sin_table,
    void *k_cache,
    void *v_cache,
    void *hidden_buffer,
    void *g_activations,
    void *g_residual,
    void *g_q,
    void *g_k,
    void *g_v,
    void *g_attn_out,
    void *g_mlp_intermediate,
    void *g_normalized,
    void *block_max_vals,
    void *block_max_idxs,
    int num_blocks,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    uint64_t *profiler_buffer,
    cudaStream_t stream)
{
    (void)profiler_buffer;
    launch_p1_decode(
        input_token_id,
        (const __nv_bfloat16 *)embed_weight,
        layer_weights,
        (const __nv_bfloat16 *)final_norm_weight,
        (const __nv_bfloat16 *)cos_table,
        (const __nv_bfloat16 *)sin_table,
        (__nv_bfloat16 *)k_cache,
        (__nv_bfloat16 *)v_cache,
        (__nv_bfloat16 *)hidden_buffer,
        (float *)g_activations,
        (float *)g_residual,
        (float *)g_q,
        (float *)g_k,
        (float *)g_v,
        (float *)g_attn_out,
        (float *)g_mlp_intermediate,
        (float *)g_normalized,
        num_blocks,
        num_layers,
        position,
        cache_len,
        max_seq_len,
        attn_scale,
        stream);

    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        (float *)block_max_vals,
        (int *)block_max_idxs);

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float *)block_max_vals,
        (const int *)block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS);
}

extern "C" void launch_ldg_decode_with_logits(
    int input_token_id,
    int *output_token_id,
    float *logits_output,
    const void *embed_weight,
    const LDGLayerWeights *layer_weights,
    const void *final_norm_weight,
    const void *lm_head_weight,
    const void *cos_table,
    const void *sin_table,
    void *k_cache,
    void *v_cache,
    void *hidden_buffer,
    void *g_activations,
    void *g_residual,
    void *g_q,
    void *g_k,
    void *g_v,
    void *g_attn_out,
    void *g_mlp_intermediate,
    void *g_normalized,
    void *block_max_vals,
    void *block_max_idxs,
    int num_blocks,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    uint64_t *profiler_buffer,
    cudaStream_t stream)
{
    (void)profiler_buffer;
    launch_p1_decode(
        input_token_id,
        (const __nv_bfloat16 *)embed_weight,
        layer_weights,
        (const __nv_bfloat16 *)final_norm_weight,
        (const __nv_bfloat16 *)cos_table,
        (const __nv_bfloat16 *)sin_table,
        (__nv_bfloat16 *)k_cache,
        (__nv_bfloat16 *)v_cache,
        (__nv_bfloat16 *)hidden_buffer,
        (float *)g_activations,
        (float *)g_residual,
        (float *)g_q,
        (float *)g_k,
        (float *)g_v,
        (float *)g_attn_out,
        (float *)g_mlp_intermediate,
        (float *)g_normalized,
        num_blocks,
        num_layers,
        position,
        cache_len,
        max_seq_len,
        attn_scale,
        stream);

    ldg_lm_head_logits<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        logits_output);

    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized,
        (const __nv_bfloat16 *)lm_head_weight,
        (float *)block_max_vals,
        (int *)block_max_idxs);

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float *)block_max_vals,
        (const int *)block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS);
}
