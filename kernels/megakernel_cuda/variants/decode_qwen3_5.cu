/**
 * Qwen3.5-0.8B fused decode — one launch per layer.
 *
 * Uses global-memory scratch buffers to avoid shared-memory limits.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// =============================================================================
// Model constants for Qwen3.5-0.8B
// =============================================================================

constexpr int WARP_SIZE = 32;

constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3584;

constexpr int NUM_Q_HEADS = 8;
constexpr int NUM_KV_HEADS = 2;
constexpr int HEAD_DIM = 256;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM;
constexpr int ROTARY_DIM = 64;

constexpr int NUM_K_HEADS = 16;
constexpr int NUM_V_HEADS = 16;
constexpr int HEAD_K_DIM = 128;
constexpr int HEAD_V_DIM = 128;
constexpr int KEY_DIM = NUM_K_HEADS * HEAD_K_DIM;
constexpr int VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM;
constexpr int CONV_DIM = KEY_DIM * 2 + VALUE_DIM;
constexpr int CONV_KERNEL_SIZE = 4;

constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr float RMS_EPS = 1e-6f;

// =============================================================================
// Layer weight structs
// =============================================================================

struct FullLayerWeights {
    const __nv_bfloat16 *input_norm;
    const __nv_bfloat16 *q_proj;
    const __nv_bfloat16 *k_proj;
    const __nv_bfloat16 *v_proj;
    const __nv_bfloat16 *q_norm;
    const __nv_bfloat16 *k_norm;
    const __nv_bfloat16 *o_proj;
    const __nv_bfloat16 *post_norm;
    const __nv_bfloat16 *gate_proj;
    const __nv_bfloat16 *up_proj;
    const __nv_bfloat16 *down_proj;
};

struct LinearLayerWeights {
    const __nv_bfloat16 *input_norm;
    const __nv_bfloat16 *in_proj_qkv;
    const __nv_bfloat16 *conv1d_weight;
    const __nv_bfloat16 *in_proj_z;
    const __nv_bfloat16 *in_proj_b;
    const __nv_bfloat16 *in_proj_a;
    const float *dt_bias;
    const float *a_log;
    const __nv_bfloat16 *norm_weight;
    const __nv_bfloat16 *out_proj;
    const __nv_bfloat16 *post_norm;
    const __nv_bfloat16 *gate_proj;
    const __nv_bfloat16 *up_proj;
    const __nv_bfloat16 *down_proj;
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
    val = warp_reduce_sum(val);
    __shared__ float smem[NUM_WARPS];
    int tid = threadIdx.x;
    if ((tid % WARP_SIZE) == 0) {
        smem[tid / WARP_SIZE] = val;
    }
    __syncthreads();
    if (tid < NUM_WARPS) {
        val = smem[tid];
    } else {
        val = 0.0f;
    }
    val = warp_reduce_sum(val);
    if (tid == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus(float x) {
    return logf(1.0f + expf(x));
}

// =============================================================================
// RMSNorm (centered) — bfloat16 input
// =============================================================================

__device__ void rms_norm_centered(
    const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight,
    float *output,
    int size
) {
    int tid = threadIdx.x;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float v = __bfloat162float(input[i]);
        output[i] = v;
        local_sum_sq += v * v;
    }
    float sum_sq = block_reduce_sum(local_sum_sq);
    float rstd = rsqrtf(sum_sq / float(size) + RMS_EPS);
    for (int i = tid; i < size; i += blockDim.x) {
        float w = __bfloat162float(weight[i]);
        output[i] = output[i] * rstd * (1.0f + w);
    }
    __syncthreads();
}

// =============================================================================
// RMSNorm (centered) — float input
// =============================================================================

__device__ void rms_norm_f(
    const float *input,
    const __nv_bfloat16 *weight,
    float *output,
    int size
) {
    int tid = threadIdx.x;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float v = input[i];
        output[i] = v;
        local_sum_sq += v * v;
    }
    float sum_sq = block_reduce_sum(local_sum_sq);
    float rstd = rsqrtf(sum_sq / float(size) + RMS_EPS);
    for (int i = tid; i < size; i += blockDim.x) {
        float w = __bfloat162float(weight[i]);
        output[i] = output[i] * rstd * (1.0f + w);
    }
    __syncthreads();
}

// =============================================================================
// Mat-vec: output = input @ W^T
// =============================================================================

__device__ void matvec(
    const float *input,
    const __nv_bfloat16 *weight,
    float *output,
    int out_features,
    int in_features
) {
    int tid = threadIdx.x;
    for (int row = tid; row < out_features; row += blockDim.x) {
        float sum = 0.0f;
        const __nv_bfloat16 *w_row = weight + row * in_features;
        for (int k = 0; k < in_features; k++) {
            sum += input[k] * __bfloat162float(w_row[k]);
        }
        output[row] = sum;
    }
    __syncthreads();
}

// =============================================================================
// Full Attention Layer
// =============================================================================

__device__ void full_attention_layer(
    const __nv_bfloat16 *hidden_in,
    __nv_bfloat16 *hidden_out,
    const FullLayerWeights &w,
    __nv_bfloat16 *k_cache,
    __nv_bfloat16 *v_cache,
    const __nv_bfloat16 *cos_table,
    const __nv_bfloat16 *sin_table,
    int position,
    int cache_len,
    int max_seq_len,
    float *scratch
) {
    int tid = threadIdx.x;

    // Scratch layout
    float *s_norm      = scratch;
    float *s_q         = scratch + HIDDEN_SIZE;
    float *s_k         = scratch + HIDDEN_SIZE + 4096;
    float *s_v         = scratch + HIDDEN_SIZE + 4096 + KV_SIZE;
    float *s_gate      = scratch + HIDDEN_SIZE + 4096 + KV_SIZE * 2;
    float *s_attn_out  = scratch + HIDDEN_SIZE + 4096 + KV_SIZE * 2 + Q_SIZE;
    float *s_residual  = scratch + HIDDEN_SIZE + 4096 + KV_SIZE * 2 + Q_SIZE * 2;
    float *s_o_out     = scratch + HIDDEN_SIZE + 4096 + KV_SIZE * 2 + Q_SIZE * 2 + HIDDEN_SIZE;
    float *s_gate_mlp  = scratch + HIDDEN_SIZE + 4096 + KV_SIZE * 2 + Q_SIZE * 2 + HIDDEN_SIZE * 2;
    float *s_up_mlp    = scratch + HIDDEN_SIZE + 4096 + KV_SIZE * 2 + Q_SIZE * 2 + HIDDEN_SIZE * 2 + INTERMEDIATE_SIZE;
    float *s_mlp_out   = scratch + HIDDEN_SIZE + 4096 + KV_SIZE * 2 + Q_SIZE * 2 + HIDDEN_SIZE * 2 + INTERMEDIATE_SIZE * 2;

    // Step 1: RMSNorm
    rms_norm_centered(hidden_in, w.input_norm, s_norm, HIDDEN_SIZE);
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) s_residual[i] = __bfloat162float(hidden_in[i]);
    __syncthreads();

    // Step 2: QKV projection
    matvec(s_norm, w.q_proj, s_q, 4096, HIDDEN_SIZE);
    matvec(s_norm, w.k_proj, s_k, KV_SIZE, HIDDEN_SIZE);
    matvec(s_norm, w.v_proj, s_v, KV_SIZE, HIDDEN_SIZE);
    __syncthreads();

    // q_proj output is [q0,g0,q1,g1,...] interleaved per head (matching PyTorch view+chunk)
    for (int h = 0; h < NUM_Q_HEADS; h++) {
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            s_gate[h * HEAD_DIM + i] = s_q[h * HEAD_DIM * 2 + HEAD_DIM + i];
        }
    }
    __syncthreads();

    // Step 3: Q/K Norm per head
    for (int h = 0; h < NUM_Q_HEADS; h++) {
        float *q_head = s_q + h * HEAD_DIM * 2;
        float sum_sq = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) sum_sq += q_head[i] * q_head[i];
        sum_sq = block_reduce_sum(sum_sq);
        float rstd = rsqrtf(sum_sq / float(HEAD_DIM) + RMS_EPS);
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            q_head[i] = q_head[i] * rstd * (1.0f + __bfloat162float(w.q_norm[i]));
        }
    }
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        float *k_head = s_k + h * HEAD_DIM;
        float sum_sq = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) sum_sq += k_head[i] * k_head[i];
        sum_sq = block_reduce_sum(sum_sq);
        float rstd = rsqrtf(sum_sq / float(HEAD_DIM) + RMS_EPS);
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            k_head[i] = k_head[i] * rstd * (1.0f + __bfloat162float(w.k_norm[i]));
        }
    }
    __syncthreads();

    // Step 4: RoPE (partial)
    const __nv_bfloat16 *cos_pos = cos_table + position * ROTARY_DIM;
    const __nv_bfloat16 *sin_pos = sin_table + position * ROTARY_DIM;
    for (int h = 0; h < NUM_Q_HEADS; h++) {
        float *q_head = s_q + h * HEAD_DIM * 2;
        for (int i = tid; i < ROTARY_DIM / 2; i += blockDim.x) {
            int i1 = i, i2 = i + ROTARY_DIM / 2;
            float q1 = q_head[i1], q2 = q_head[i2];
            float c = __bfloat162float(cos_pos[i1]), s_ = __bfloat162float(sin_pos[i1]);
            q_head[i1] = q1 * c - q2 * s_;
            q_head[i2] = q2 * c + q1 * s_;
        }
    }
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        float *k_head = s_k + h * HEAD_DIM;
        for (int i = tid; i < ROTARY_DIM / 2; i += blockDim.x) {
            int i1 = i, i2 = i + ROTARY_DIM / 2;
            float k1 = k_head[i1], k2 = k_head[i2];
            float c = __bfloat162float(cos_pos[i1]), s_ = __bfloat162float(sin_pos[i1]);
            k_head[i1] = k1 * c - k2 * s_;
            k_head[i2] = k2 * c + k1 * s_;
        }
    }
    __syncthreads();

    // Step 5: KV cache write
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        __nv_bfloat16 *kch = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __nv_bfloat16 *vch = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
            kch[i] = __float2bfloat16(s_k[h * HEAD_DIM + i]);
            vch[i] = __float2bfloat16(s_v[h * HEAD_DIM + i]);
        }
    }
    __syncthreads();

    // Step 6: Attention — one warp per query head
    float attn_scale = 1.0f / sqrtf(float(HEAD_DIM));
    int kv_groups = NUM_Q_HEADS / NUM_KV_HEADS;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int elems_per_thread = HEAD_DIM / WARP_SIZE;  // 8

    for (int qh = warp_id; qh < NUM_Q_HEADS; qh += NUM_WARPS) {
        int kv_h = qh / kv_groups;
        float *q_head = s_q + qh * HEAD_DIM * 2;
        float *out_head = s_attn_out + qh * HEAD_DIM;
        float max_score = -INFINITY, sum_exp = 0.0f;
        float out_acc[8];
        for (int i = 0; i < elems_per_thread; i++) out_acc[i] = 0.0f;

        for (int pos = 0; pos < cache_len; pos++) {
            const __nv_bfloat16 *k_pos = k_cache + kv_h * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16 *v_pos = v_cache + kv_h * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            float score = 0.0f;
            for (int i = 0; i < elems_per_thread; i++) {
                int idx = lane_id * elems_per_thread + i;
                score += q_head[idx] * __bfloat162float(k_pos[idx]);
            }
            score = warp_reduce_sum(score) * attn_scale;
            score = __shfl_sync(0xffffffff, score, 0);
            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = expf(old_max - max_score);
            sum_exp = sum_exp * exp_diff + expf(score - max_score);
            float weight = expf(score - max_score);
            for (int i = 0; i < elems_per_thread; i++) {
                int idx = lane_id * elems_per_thread + i;
                out_acc[i] = out_acc[i] * exp_diff + weight * __bfloat162float(v_pos[idx]);
            }
        }
        for (int i = 0; i < elems_per_thread; i++) {
            int idx = lane_id * elems_per_thread + i;
            out_head[idx] = out_acc[i] / sum_exp;
        }
    }
    __syncthreads();

    // Step 7: Gated O-proj
    for (int i = tid; i < Q_SIZE; i += blockDim.x) s_attn_out[i] = s_attn_out[i] * sigmoid(s_gate[i]);
    __syncthreads();
    matvec(s_attn_out, w.o_proj, s_o_out, HIDDEN_SIZE, Q_SIZE);
    __syncthreads();
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) s_norm[i] = s_o_out[i] + s_residual[i];
    __syncthreads();

    // Step 8: Post-norm + MLP
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) s_residual[i] = s_norm[i];
    __syncthreads();
    rms_norm_f(s_norm, w.post_norm, s_norm, HIDDEN_SIZE);
    matvec(s_norm, w.gate_proj, s_gate_mlp, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    matvec(s_norm, w.up_proj, s_up_mlp, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    __syncthreads();
    for (int i = tid; i < INTERMEDIATE_SIZE; i += blockDim.x) s_gate_mlp[i] = silu(s_gate_mlp[i]) * s_up_mlp[i];
    __syncthreads();
    matvec(s_gate_mlp, w.down_proj, s_mlp_out, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    __syncthreads();
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) {
        hidden_out[i] = __float2bfloat16(s_mlp_out[i] + s_residual[i]);
    }
    __syncthreads();
}

// =============================================================================
// Linear Attention Layer
// =============================================================================

__device__ void linear_attention_layer(
    const __nv_bfloat16 *hidden_in,
    __nv_bfloat16 *hidden_out,
    const LinearLayerWeights &w,
    float *conv_state,
    float *recurrent_state,
    int position,
    float *scratch
) {
    int tid = threadIdx.x;

    // Scratch layout
    float *s_norm      = scratch;
    float *s_conv_in   = scratch + HIDDEN_SIZE;
    float *s_conv_out  = scratch + HIDDEN_SIZE + CONV_DIM;
    float *s_q         = scratch + HIDDEN_SIZE + CONV_DIM * 2;
    float *s_k         = scratch + HIDDEN_SIZE + CONV_DIM * 2 + KEY_DIM;
    float *s_v         = scratch + HIDDEN_SIZE + CONV_DIM * 2 + KEY_DIM * 2;
    float *s_beta      = scratch + HIDDEN_SIZE + CONV_DIM * 2 + KEY_DIM * 2 + VALUE_DIM;
    float *s_a         = scratch + HIDDEN_SIZE + CONV_DIM * 2 + KEY_DIM * 2 + VALUE_DIM + NUM_V_HEADS;
    float *s_z         = scratch + HIDDEN_SIZE + CONV_DIM * 2 + KEY_DIM * 2 + VALUE_DIM + NUM_V_HEADS * 2;
    float *s_attn_out  = scratch + HIDDEN_SIZE + CONV_DIM * 2 + KEY_DIM * 2 + VALUE_DIM + NUM_V_HEADS * 2 + VALUE_DIM;
    float *s_out_proj  = scratch + HIDDEN_SIZE + CONV_DIM * 2 + KEY_DIM * 2 + VALUE_DIM + NUM_V_HEADS * 2 + VALUE_DIM * 2;
    float *s_residual  = scratch + HIDDEN_SIZE + CONV_DIM * 2 + KEY_DIM * 2 + VALUE_DIM + NUM_V_HEADS * 2 + VALUE_DIM * 2 + HIDDEN_SIZE;
    float *s_gate_mlp  = scratch + HIDDEN_SIZE + CONV_DIM * 2 + KEY_DIM * 2 + VALUE_DIM + NUM_V_HEADS * 2 + VALUE_DIM * 2 + HIDDEN_SIZE * 2;
    float *s_up_mlp    = scratch + HIDDEN_SIZE + CONV_DIM * 2 + KEY_DIM * 2 + VALUE_DIM + NUM_V_HEADS * 2 + VALUE_DIM * 2 + HIDDEN_SIZE * 2 + INTERMEDIATE_SIZE;
    float *s_mlp_out   = scratch + HIDDEN_SIZE + CONV_DIM * 2 + KEY_DIM * 2 + VALUE_DIM + NUM_V_HEADS * 2 + VALUE_DIM * 2 + HIDDEN_SIZE * 2 + INTERMEDIATE_SIZE * 2;

    // Step 1: RMSNorm
    rms_norm_centered(hidden_in, w.input_norm, s_norm, HIDDEN_SIZE);
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) s_residual[i] = __bfloat162float(hidden_in[i]);
    __syncthreads();

    // Step 2: in_proj_qkv + conv1d + silu
    matvec(s_norm, w.in_proj_qkv, s_conv_in, CONV_DIM, HIDDEN_SIZE);
    __syncthreads();

    for (int c = tid; c < CONV_DIM; c += blockDim.x) {
        float mixed[CONV_KERNEL_SIZE];
        for (int k = 0; k < CONV_KERNEL_SIZE - 1; k++) mixed[k] = conv_state[c * (CONV_KERNEL_SIZE - 1) + k];
        mixed[CONV_KERNEL_SIZE - 1] = s_conv_in[c];
        float out = 0.0f;
        for (int k = 0; k < CONV_KERNEL_SIZE; k++) out += mixed[k] * __bfloat162float(w.conv1d_weight[c * CONV_KERNEL_SIZE + k]);
        s_conv_out[c] = silu(out);
        for (int k = 0; k < CONV_KERNEL_SIZE - 1; k++) conv_state[c * (CONV_KERNEL_SIZE - 1) + k] = mixed[k + 1];
    }
    __syncthreads();

    for (int i = tid; i < KEY_DIM; i += blockDim.x) { s_q[i] = s_conv_out[i]; s_k[i] = s_conv_out[KEY_DIM + i]; }
    for (int i = tid; i < VALUE_DIM; i += blockDim.x) s_v[i] = s_conv_out[KEY_DIM * 2 + i];
    __syncthreads();

    // Step 3: beta, gate, z
    matvec(s_norm, w.in_proj_b, s_beta, NUM_V_HEADS, HIDDEN_SIZE);
    matvec(s_norm, w.in_proj_a, s_a, NUM_V_HEADS, HIDDEN_SIZE);
    matvec(s_norm, w.in_proj_z, s_z, VALUE_DIM, HIDDEN_SIZE);
    __syncthreads();

    for (int h = tid; h < NUM_V_HEADS; h += blockDim.x) {
        s_beta[h] = sigmoid(s_beta[h]);
        float g = -expf(w.a_log[h]) * softplus(s_a[h] + w.dt_bias[h]);
        s_a[h] = expf(g);
    }
    __syncthreads();

    // Step 4: Q/K L2 norm
    for (int h = 0; h < NUM_V_HEADS; h++) {
        float *q_head = s_q + h * HEAD_K_DIM;
        float sum_sq = 0.0f;
        for (int i = tid; i < HEAD_K_DIM; i += blockDim.x) sum_sq += q_head[i] * q_head[i];
        sum_sq = block_reduce_sum(sum_sq);
        float rnorm = rsqrtf(sum_sq + 1e-6f);
        for (int i = tid; i < HEAD_K_DIM; i += blockDim.x) q_head[i] *= rnorm;
    }
    for (int h = 0; h < NUM_V_HEADS; h++) {
        float *k_head = s_k + h * HEAD_K_DIM;
        float sum_sq = 0.0f;
        for (int i = tid; i < HEAD_K_DIM; i += blockDim.x) sum_sq += k_head[i] * k_head[i];
        sum_sq = block_reduce_sum(sum_sq);
        float rnorm = rsqrtf(sum_sq + 1e-6f);
        for (int i = tid; i < HEAD_K_DIM; i += blockDim.x) k_head[i] *= rnorm;
    }
    __syncthreads();

    // Scale q by 1/sqrt(head_k_dim) (matching PyTorch chunk/recurrent paths)
    float q_scale = 1.0f / sqrtf(float(HEAD_K_DIM));
    for (int h = 0; h < NUM_V_HEADS; h++) {
        float *q_head = s_q + h * HEAD_K_DIM;
        for (int i = tid; i < HEAD_K_DIM; i += blockDim.x) q_head[i] *= q_scale;
    }
    __syncthreads();

    // Step 5: Recurrent state update (all heads in parallel)
    for (int idx = tid; idx < NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM; idx += blockDim.x) {
        int h = idx / (HEAD_K_DIM * HEAD_V_DIM);
        int rem = idx % (HEAD_K_DIM * HEAD_V_DIM);
        float *S = recurrent_state + h * HEAD_K_DIM * HEAD_V_DIM;
        S[rem] *= s_a[h];
    }
    __syncthreads();

    // Compute delta[j] = (v[j] - kv_mem[j]) * beta for all heads
    for (int j = tid; j < NUM_V_HEADS * HEAD_V_DIM; j += blockDim.x) {
        int h = j / HEAD_V_DIM;
        int jj = j % HEAD_V_DIM;
        float *S = recurrent_state + h * HEAD_K_DIM * HEAD_V_DIM;
        float *k_head = s_k + h * HEAD_K_DIM;
        float *v_head = s_v + h * HEAD_V_DIM;
        float kv_mem = 0.0f;
        for (int i = 0; i < HEAD_K_DIM; i++) kv_mem += S[i * HEAD_V_DIM + jj] * k_head[i];
        v_head[jj] = (v_head[jj] - kv_mem) * s_beta[h];
    }
    __syncthreads();

    // Update S for all heads
    for (int idx = tid; idx < NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM; idx += blockDim.x) {
        int h = idx / (HEAD_K_DIM * HEAD_V_DIM);
        int rem = idx % (HEAD_K_DIM * HEAD_V_DIM);
        int i = rem / HEAD_V_DIM;
        int j = rem % HEAD_V_DIM;
        float *S = recurrent_state + h * HEAD_K_DIM * HEAD_V_DIM;
        float *k_head = s_k + h * HEAD_K_DIM;
        float *v_head = s_v + h * HEAD_V_DIM;
        S[rem] += k_head[i] * v_head[j];
    }
    __syncthreads();

    // Compute output for all heads
    for (int j = tid; j < NUM_V_HEADS * HEAD_V_DIM; j += blockDim.x) {
        int h = j / HEAD_V_DIM;
        int jj = j % HEAD_V_DIM;
        float *S = recurrent_state + h * HEAD_K_DIM * HEAD_V_DIM;
        float *q_head = s_q + h * HEAD_K_DIM;
        float *out_head = s_attn_out + h * HEAD_V_DIM;
        float sum = 0.0f;
        for (int i = 0; i < HEAD_K_DIM; i++) sum += S[i * HEAD_V_DIM + jj] * q_head[i];
        out_head[jj] = sum;
    }
    __syncthreads();

    // Step 6: RMSNormGated + out_proj
    for (int h = 0; h < NUM_V_HEADS; h++) {
        float *out_head = s_attn_out + h * HEAD_V_DIM;
        float sum_sq = 0.0f;
        for (int i = tid; i < HEAD_V_DIM; i += blockDim.x) sum_sq += out_head[i] * out_head[i];
        sum_sq = block_reduce_sum(sum_sq);
        float rstd = rsqrtf(sum_sq / float(HEAD_V_DIM) + RMS_EPS);
        for (int i = tid; i < HEAD_V_DIM; i += blockDim.x) {
            out_head[i] = out_head[i] * rstd * __bfloat162float(w.norm_weight[i]) * silu(s_z[h * HEAD_V_DIM + i]);
        }
    }
    __syncthreads();

    matvec(s_attn_out, w.out_proj, s_out_proj, HIDDEN_SIZE, VALUE_DIM);
    __syncthreads();
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) s_norm[i] = s_out_proj[i] + s_residual[i];
    __syncthreads();

    // Step 7: Post-norm + MLP
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) s_residual[i] = s_norm[i];
    __syncthreads();
    rms_norm_f(s_norm, w.post_norm, s_norm, HIDDEN_SIZE);
    matvec(s_norm, w.gate_proj, s_gate_mlp, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    matvec(s_norm, w.up_proj, s_up_mlp, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    __syncthreads();
    for (int i = tid; i < INTERMEDIATE_SIZE; i += blockDim.x) s_gate_mlp[i] = silu(s_gate_mlp[i]) * s_up_mlp[i];
    __syncthreads();
    matvec(s_gate_mlp, w.down_proj, s_mlp_out, HIDDEN_SIZE, INTERMEDIATE_SIZE);
    __syncthreads();
    for (int i = tid; i < HIDDEN_SIZE; i += blockDim.x) {
        hidden_out[i] = __float2bfloat16(s_mlp_out[i] + s_residual[i]);
    }
    __syncthreads();
}

// Forward declaration of the kernel (defined below)
__global__ void qwen3_5_layer_kernel(
    const __nv_bfloat16 *hidden_in,
    __nv_bfloat16 *hidden_out,
    int layer_type,
    const void *layer_weights,
    __nv_bfloat16 *k_cache,
    __nv_bfloat16 *v_cache,
    float *conv_state,
    float *recurrent_state,
    const __nv_bfloat16 *cos_table,
    const __nv_bfloat16 *sin_table,
    int position,
    int cache_len,
    int max_seq_len,
    int layer_idx,
    float *scratch
);

// =============================================================================
// C-compatible launcher (called from decode_wrapper_qwen3_5.cpp)
// =============================================================================

extern "C" void launch_qwen3_5_layer(
    const void *hidden_in,
    void *hidden_out,
    int layer_type,
    const void *layer_weights,
    void *k_cache,
    void *v_cache,
    void *conv_state,
    void *recurrent_state,
    const void *cos_table,
    const void *sin_table,
    int position,
    int cache_len,
    int max_seq_len,
    int layer_idx,
    void *scratch,
    cudaStream_t stream
) {
    qwen3_5_layer_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_in),
        reinterpret_cast<__nv_bfloat16*>(hidden_out),
        layer_type,
        layer_weights,
        reinterpret_cast<__nv_bfloat16*>(k_cache),
        reinterpret_cast<__nv_bfloat16*>(v_cache),
        reinterpret_cast<float*>(conv_state),
        reinterpret_cast<float*>(recurrent_state),
        reinterpret_cast<const __nv_bfloat16*>(cos_table),
        reinterpret_cast<const __nv_bfloat16*>(sin_table),
        position,
        cache_len,
        max_seq_len,
        layer_idx,
        reinterpret_cast<float*>(scratch)
    );
}

// =============================================================================
// Main kernel
// =============================================================================

__global__ void qwen3_5_layer_kernel(
    const __nv_bfloat16 *hidden_in,
    __nv_bfloat16 *hidden_out,
    int layer_type,
    const void *layer_weights,
    __nv_bfloat16 *k_cache,
    __nv_bfloat16 *v_cache,
    float *conv_state,
    float *recurrent_state,
    const __nv_bfloat16 *cos_table,
    const __nv_bfloat16 *sin_table,
    int position,
    int cache_len,
    int max_seq_len,
    int layer_idx,
    float *scratch
) {
    if (layer_type == 0) {
        const FullLayerWeights *w = reinterpret_cast<const FullLayerWeights*>(layer_weights);
        __nv_bfloat16 *kcl = k_cache + layer_idx * NUM_KV_HEADS * max_seq_len * HEAD_DIM;
        __nv_bfloat16 *vcl = v_cache + layer_idx * NUM_KV_HEADS * max_seq_len * HEAD_DIM;
        full_attention_layer(hidden_in, hidden_out, *w, kcl, vcl, cos_table, sin_table,
                             position, cache_len, max_seq_len, scratch);
    } else {
        const LinearLayerWeights *w = reinterpret_cast<const LinearLayerWeights*>(layer_weights);
        float *csl = conv_state + layer_idx * CONV_DIM * (CONV_KERNEL_SIZE - 1);
        float *rsl = recurrent_state + layer_idx * NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM;
        linear_attention_layer(hidden_in, hidden_out, *w, csl, rsl, position, scratch);
    }
}
