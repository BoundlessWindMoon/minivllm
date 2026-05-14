/**
 * decode_p7.cu — naive + P3 (atomic sync) + P6 placement + P7 (PTX
 * ``prefetch.global.L2::evict_last``).
 *
 * Forked from ``decode_p6.cu``. The placement of the prefetch (which 4 idle
 * blocks do it, which layer's weights they fetch, full cache-line coverage)
 * is identical to P6 — only the body of ``ldg_prefetch_weights_l2`` changes.
 *
 * P7 changes vs P6:
 *   - The streaming load (``ld.global.nc.v4.b32`` with an asm sink) is
 *     replaced by the explicit PTX hint ``prefetch.global.L2::evict_last``.
 *   - This hint:
 *       * Is *not* a real load — it produces no register output and consumes
 *         no LDST instruction slot. The compiler does not need an asm sink to
 *         keep it alive.
 *       * Tells the hardware to bring the line into L2 (and *not* L1), with
 *         the ``evict_last`` policy so it sits at the back of the LRU stack
 *         and survives subsequent unrelated traffic until it is consumed.
 *       * Issues at 128-byte stride per thread, same as P6, so coverage is
 *         the same: one prefetch per cache line, no gaps, no duplicates.
 *   - Hypothesis being tested: if P6's loss on the 4050 came from L1
 *     pollution / LDST-pipe contention rather than from L2 capacity, P7
 *     should claw back some of the loss. If the loss is purely L2-capacity
 *     bound (22 MB prefetch ≥ 24 MB L2), P7 will look about the same as P6.
 *
 * What is NOT in P7 (these are later rounds):
 *   - Fire-and-forget DRAM prefetch at the 4 sync windows — that's P9.
 *   - Site-#4 next-layer-QKV prefetch relocation — that's P10.
 *
 * P7 keeps these P6 / P3 / naive behaviours (so the delta is purely the
 * prefetch-instruction swap):
 *   - AtomicGridSync + partial barriers from P3.
 *   - Idle-block branch with 4-way prefetch group split from P6.
 *   - Block-0-only RMSNorm with ``g_normalized`` broadcast (P1 reverted).
 *   - Scalar bf16 weight / KV-cache loads (P4 reverted).
 *   - No SMEM padding (P8 reverted).
 *
 * The launch surface (``extern "C" launch_ldg_decode[_with_logits]``) is
 * byte-identical with naive / decode_ldg.cu so ``decode_wrapper.cpp`` works
 * unchanged.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "sm_profiler.h"

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

// LM head – kept identical to decode_ldg.cu for fair end-to-end comparison.
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
// AtomicGridSync — replaces cooperative_groups::this_grid().sync()
// =============================================================================
//
// Two atomic counters in global memory: ``counter`` tracks arrivals,
// ``generation`` is bumped by the last arriver so spinning blocks can exit.
// Each block stores its expected generation in ``local_gen``. Requires
// ``__launch_bounds__(BLOCK_SIZE, 1)`` so only one block is co-resident per
// SM (otherwise the spin-wait would deadlock against an unscheduled block).

struct AtomicGridSync
{
    unsigned int *counter;
    unsigned int *generation;
    unsigned int nblocks;
    unsigned int local_gen;

    __device__ void sync()
    {
        __syncthreads();
        if (threadIdx.x == 0)
        {
            unsigned int my_gen = local_gen;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1)
            {
                *counter = 0;
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                atomicAdd(generation, 1);
            }
            else
            {
                volatile unsigned int *vgen = (volatile unsigned int *)generation;
                while (*vgen <= my_gen)
                {
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
// P7 idle-block L2 prefetch helper (PTX prefetch.global.L2::evict_last).
// =============================================================================
//
// Each thread issues one ``prefetch.global.L2::evict_last`` per outer
// iteration at 128-byte stride. Compared to P6's ``ld.global.nc.v4.b32`` body:
//   - No register output, no asm sink needed (the hint is not a load).
//   - Does not occupy an LDST instruction slot the way the streaming load
//     does, so it does not contend with the attention block's real loads.
//   - Does not pollute L1 — only L2 is touched, with ``evict_last`` policy so
//     the line sits at the back of the LRU stack until it is consumed.
// Stride / coverage / launch shape are identical to P6 (256 threads × 128B
// per iter = 32 KB / iter = 256 cache lines, one prefetch per line).

__device__ void ldg_prefetch_weights_l2(
    const __nv_bfloat16 *__restrict__ weights, int num_elements)
{
    const char *base = reinterpret_cast<const char *>(weights);
    int total_bytes = num_elements * 2; // bf16 = 2 bytes
    for (int offset = threadIdx.x * 128; offset < total_bytes;
         offset += LDG_BLOCK_SIZE * 128)
    {
        asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(base + offset));
    }
}

// =============================================================================
// Phase 1: RMSNorm (block 0) + QKV projection (all blocks)
// =============================================================================
//
// Block 0 computes the normalized vector and stores it in g_normalized (float,
// global memory). A grid sync makes that result visible to every block, then
// each block does its share of the [Q_SIZE+2*KV_SIZE] matvec.
//
__device__ void naive_rmsnorm_qkv(
    AtomicGridSync &grid,
    const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ norm_weight,
    const __nv_bfloat16 *__restrict__ q_weight,
    const __nv_bfloat16 *__restrict__ k_weight,
    const __nv_bfloat16 *__restrict__ v_weight,
    float *__restrict__ g_normalized,
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

    // Block 0: RMSNorm reduction + broadcast into g_normalized
    if (block_id == 0)
    {
        __shared__ float smem_reduce[LDG_NUM_WARPS];
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
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
        {
            float v = __bfloat162float(__ldg(input + i));
            float w = __bfloat162float(__ldg(norm_weight + i));
            g_normalized[i] = v * rstd * w;
        }
    }
    // NAIVE: full grid sync between norm and projection (P1 reverts this away)
    grid.sync();

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

        // NAIVE: scalar bf16 loads from weight, no uint4 (P4 reverts this)
        float sum = 0.0f;
        for (int k = lane_id; k < HIDDEN_SIZE; k += WARP_SIZE)
        {
            float w = __bfloat162float(__ldg(weight_row + k));
            sum += w * g_normalized[k];
        }
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            *output_ptr = sum;
    }

    grid.sync();
}

// =============================================================================
// Phase 2: QK Norm + RoPE + KV Cache write
// =============================================================================
//
// 16 Q heads (one per Q head) + 8 KV heads. We use the first 16 blocks for Q
// heads and let those same blocks pick up the 8 KV heads (block_id < 8 also
// handles a KV head).
//
__device__ void naive_qk_norm_rope_cache(
    AtomicGridSync &grid,
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
    int max_seq_len,
    unsigned int *__restrict__ kv_flag,
    int layer_idx)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16 *cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16 *sin_pos = sin_table + position * HEAD_DIM;

    // Q heads – distribute NUM_Q_HEADS across all blocks
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

    // K + V heads – distribute NUM_KV_HEADS across all blocks
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

    // P3 partial barrier: only the 16 attention blocks need to see all Q / K / V
    // writes. Blocks >= ATTN_BLOCKS (16..19) skip this barrier and wait on
    // attn_flag inside the attention phase instead. kv_flag is monotonic across
    // layers (target = ATTN_BLOCKS * (layer_idx + 1)).
    const int ATTN_BLOCKS = NUM_Q_HEADS; // 16
    if (block_id < ATTN_BLOCKS)
    {
        __syncthreads();
        if (threadIdx.x == 0)
        {
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            atomicAdd(kv_flag, 1);
            unsigned int target = (unsigned int)(ATTN_BLOCKS * (layer_idx + 1));
            volatile unsigned int *vf = (volatile unsigned int *)kv_flag;
            while (*vf < target)
            {
            }
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
        }
        __syncthreads();
    }
}

// =============================================================================
// Phase 3: Flash-decoding attention
// =============================================================================
//
// 16 attention blocks (one per Q head). Remaining blocks idle through this
// phase – naive does nothing useful with them (no prefetch).
//
__device__ void naive_attention(
    AtomicGridSync &grid,
    const float *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_cache,
    const __nv_bfloat16 *__restrict__ v_cache,
    float *__restrict__ attn_out,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    // P6: idle-block prefetch needs the post-attn weight pointers
    const __nv_bfloat16 *__restrict__ o_weight,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const __nv_bfloat16 *__restrict__ up_weight,
    const __nv_bfloat16 *__restrict__ down_weight,
    unsigned int *__restrict__ attn_flag,
    int layer_idx)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const int ATTN_BLOCKS = NUM_Q_HEADS; // 16

    if (block_id < ATTN_BLOCKS)
    {
        __shared__ float s_max_score[LDG_NUM_WARPS];
        __shared__ float s_sum_exp[LDG_NUM_WARPS];
        __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

        int qh = block_id; // one Q head per block
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

            // NAIVE: scalar bf16 loads from KV cache (P4 reverts this)
            // NAIVE: Q read from global per iter, no register cache (P4 reverts this)
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
    else
    {
        // P6 idle-block L2 prefetch: split the 4 non-attention blocks into 4
        // groups (one per post-attn weight matrix) and have each block stream-
        // load its slice into L2 via __ldg(uint4) at full cache-line stride.
        int prefetch_block_id = block_id - ATTN_BLOCKS;
        int num_prefetch_blocks = num_blocks - ATTN_BLOCKS;
        if (num_prefetch_blocks > 0)
        {
            constexpr int num_groups = 4;
            int group = prefetch_block_id * num_groups / num_prefetch_blocks;
            int grp_start = group * num_prefetch_blocks / num_groups;
            int adj = prefetch_block_id - grp_start;
            int grp_count = ((group + 1) * num_prefetch_blocks / num_groups) - grp_start;
            if (group == 0)
            {
                int total = Q_SIZE * HIDDEN_SIZE;
                int elems = total / grp_count;
                int offset = adj * elems;
                if (offset < total)
                    ldg_prefetch_weights_l2(o_weight + offset, min(elems, total - offset));
            }
            else if (group == 1)
            {
                int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
                int elems = total / grp_count;
                int offset = adj * elems;
                if (offset < total)
                    ldg_prefetch_weights_l2(gate_weight + offset, min(elems, total - offset));
            }
            else if (group == 2)
            {
                int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
                int elems = total / grp_count;
                int offset = adj * elems;
                if (offset < total)
                    ldg_prefetch_weights_l2(up_weight + offset, min(elems, total - offset));
            }
            else
            {
                int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
                int elems = total / grp_count;
                int offset = adj * elems;
                if (offset < total)
                    ldg_prefetch_weights_l2(down_weight + offset, min(elems, total - offset));
            }
        }
    }
    // P3 partial barrier on attn_flag. Attention blocks signal completion then
    // spin; non-attention blocks (16..19) skip the work but spin on the same
    // flag so they don't run ahead of o_proj inputs.
    if (block_id < ATTN_BLOCKS)
    {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        if (threadIdx.x == 0)
            atomicAdd(attn_flag, 1);
    }
    if (threadIdx.x == 0)
    {
        unsigned int target = (unsigned int)(ATTN_BLOCKS * (layer_idx + 1));
        volatile unsigned int *vf = (volatile unsigned int *)attn_flag;
        while (*vf < target)
        {
        }
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    __syncthreads();
}

// =============================================================================
// Phase 4: O proj + residual + post-norm + gate/up + down + residual
// =============================================================================

__device__ void naive_o_proj_postnorm_mlp(
    AtomicGridSync &grid,
    const __nv_bfloat16 *__restrict__ o_weight,
    const __nv_bfloat16 *__restrict__ post_norm_weight,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const __nv_bfloat16 *__restrict__ up_weight,
    const __nv_bfloat16 *__restrict__ down_weight,
    const float *__restrict__ attn_out,
    float *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_mlp_intermediate,
    float *__restrict__ g_normalized,
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
        // NAIVE: scalar bf16 weight load, attn_out read from global (no SMEM cache)
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

    // Post-attention RMSNorm – block 0 produces g_normalized in float (P1 reverted)
    if (block_id == 0)
    {
        __shared__ float smem_reduce[LDG_NUM_WARPS];
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
                smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
        }
        __syncthreads();
        float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
        {
            float w = __bfloat162float(__ldg(post_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * w;
        }
    }

    // Distributed residual update – the pre-MLP activations become the new residual
    {
        int per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
        int s = block_id * per_block;
        int e = min(s + per_block, HIDDEN_SIZE);
        for (int i = s + threadIdx.x; i < e; i += LDG_BLOCK_SIZE)
            g_residual[i] = g_activations[i];
    }

    grid.sync();

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
            // NAIVE: scalar bf16 weight loads, read g_normalized from global
            float gate_sum = 0.0f, up_sum = 0.0f;
            for (int k = lane_id; k < HIDDEN_SIZE; k += WARP_SIZE)
            {
                float gw = __bfloat162float(__ldg(gate_row + k));
                float uw = __bfloat162float(__ldg(up_row + k));
                float x = g_normalized[k];
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
        // NAIVE: scalar bf16 weight load, read mlp_intermediate from global
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
// Main persistent decode kernel (all layers fused, cooperative launch)
// =============================================================================

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
        float attn_scale,
        unsigned int *__restrict__ barrier_counter,
        unsigned int *__restrict__ barrier_sense,
        unsigned int *__restrict__ kv_flag,
        unsigned int *__restrict__ attn_flag)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    // Bootstrap: block 0 resets kv_flag and attn_flag, then every block joins
    // a single arrival barrier driven by barrier_counter / barrier_sense before
    // constructing the AtomicGridSync view.
    if (block_id == 0 && threadIdx.x == 0)
    {
        atomicExch(kv_flag, 0u);
        atomicExch(attn_flag, 0u);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        unsigned int arrived = atomicAdd(barrier_counter, 1);
        if (arrived == (unsigned int)gridDim.x - 1)
        {
            *barrier_counter = 0;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            atomicAdd(barrier_sense, 1);
        }
        else
        {
            volatile unsigned int *vs = (volatile unsigned int *)barrier_sense;
            while (*vs == 0)
            {
            }
        }
    }
    __syncthreads();
    AtomicGridSync grid{barrier_counter, barrier_sense, (unsigned int)gridDim.x, 1};

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

        naive_rmsnorm_qkv(grid, hidden_buffer, w.input_layernorm_weight,
                          w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
                          g_normalized, g_residual, g_q, g_k, g_v);

        naive_qk_norm_rope_cache(grid, g_q, g_k, g_v,
                                 w.q_norm_weight, w.k_norm_weight,
                                 cos_table, sin_table,
                                 layer_k, layer_v,
                                 position, max_seq_len,
                                 kv_flag, layer);

        naive_attention(grid, g_q, layer_k, layer_v, g_attn_out,
                        cache_len, max_seq_len, attn_scale,
                        w.o_proj_weight, w.gate_proj_weight,
                        w.up_proj_weight, w.down_proj_weight,
                        attn_flag, layer);

        naive_o_proj_postnorm_mlp(grid,
                                  w.o_proj_weight, w.post_attn_layernorm_weight,
                                  w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
                                  g_attn_out, g_residual, g_activations,
                                  g_mlp_intermediate, g_normalized,
                                  hidden_buffer);
    }

    // Final RMSNorm – block 0 only, writes g_normalized
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
// LM Head – kept identical to decode_ldg.cu (not part of the megakernel
// ablation – we only vary the persistent kernel itself).
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
// Kernel launch helper (P3: regular <<<>>> launch with atomic barrier state)
// =============================================================================
//
// AtomicGridSync replaces the cooperative grid sync, so a plain <<<>>> launch
// is sufficient. ``__launch_bounds__(LDG_BLOCK_SIZE, 1)`` still pins one block
// per SM (mandatory: co-resident blocks would deadlock the spin-wait). The
// Python caller passes ``num_blocks = props.multiProcessorCount`` (20 on the
// 4050 Laptop), which equals NUM_Q_HEADS + 4 idle blocks.
//
// Per-launch device state (allocated once, reset each call):
//   * barrier_counter / barrier_sense — AtomicGridSync ``counter`` / ``generation``.
//     Reset to zero before every kernel launch.
//   * kv_flag / attn_flag             — monotonic counters bumped by partial
//     barriers. NOT reset across launches; the kernel resets them at start
//     (see bootstrap section).

static void launch_p3_decode(
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
    static unsigned int *d_barrier_counter = nullptr;
    static unsigned int *d_barrier_sense = nullptr;
    static unsigned int *d_kv_flag = nullptr;
    static unsigned int *d_attn_flag = nullptr;
    static bool barrier_init = false;
    if (!barrier_init)
    {
        cudaMalloc(&d_barrier_counter, sizeof(unsigned int));
        cudaMalloc(&d_barrier_sense, sizeof(unsigned int));
        cudaMalloc(&d_kv_flag, sizeof(unsigned int));
        cudaMalloc(&d_attn_flag, sizeof(unsigned int));
        barrier_init = true;
    }
    cudaMemsetAsync(d_barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_barrier_sense, 0, sizeof(unsigned int), stream);

    ldg_decode_kernel<<<dim3(num_blocks), dim3(LDG_BLOCK_SIZE), 0, stream>>>(
        input_token_id,
        embed_weight,
        layer_weights,
        final_norm_weight,
        cos_table,
        sin_table,
        k_cache,
        v_cache,
        hidden_buffer,
        g_activations,
        g_residual,
        g_q,
        g_k,
        g_v,
        g_attn_out,
        g_mlp_intermediate,
        g_normalized,
        num_layers,
        position,
        cache_len,
        max_seq_len,
        attn_scale,
        d_barrier_counter,
        d_barrier_sense,
        d_kv_flag,
        d_attn_flag);
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
    (void)profiler_buffer; // naive variant does not emit sm-profiler events
    launch_p3_decode(
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
    launch_p3_decode(
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
