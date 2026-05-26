/**
 * C++ wrapper for Qwen3.5 persistent megakernel.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <tuple>
#include <cstdio>

// Forward declarations from .cu
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
    cudaStream_t stream);

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
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

int decode(
    int input_token_id,
    int position,
    int cache_len,
    at::Tensor layer_weights_bytes,
    at::Tensor embed_weight,
    at::Tensor final_norm_weight,
    at::Tensor lm_head_weight,
    at::Tensor cos_table,
    at::Tensor sin_table,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor conv_state,
    at::Tensor recurrent_state,
    at::Tensor hidden_buffer,
    at::Tensor g_activations,
    at::Tensor g_residual,
    at::Tensor g_q,
    at::Tensor g_k,
    at::Tensor g_v,
    at::Tensor g_z,
    at::Tensor g_b,
    at::Tensor g_a,
    at::Tensor g_attn_out,
    at::Tensor g_mlp_intermediate,
    at::Tensor g_normalized,
    at::Tensor full_layer_idx,
    at::Tensor linear_layer_idx,
    int64_t num_blocks,
    int64_t num_layers,
    int64_t max_seq_len,
    double attn_scale)
{
    auto output_token = at::empty({1},
        at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream();

    launch_qwen3_5_ldg_decode(
        input_token_id,
        output_token.data_ptr<int>(),
        embed_weight.data_ptr<at::BFloat16>(),
        reinterpret_cast<const void*>(layer_weights_bytes.data_ptr<uint8_t>()),
        final_norm_weight.data_ptr<at::BFloat16>(),
        lm_head_weight.data_ptr<at::BFloat16>(),
        cos_table.data_ptr<at::BFloat16>(),
        sin_table.data_ptr<at::BFloat16>(),
        k_cache.data_ptr<at::BFloat16>(),
        v_cache.data_ptr<at::BFloat16>(),
        conv_state.data_ptr<float>(),
        recurrent_state.data_ptr<float>(),
        hidden_buffer.data_ptr<at::BFloat16>(),
        g_activations.data_ptr<float>(),
        g_residual.data_ptr<float>(),
        g_q.data_ptr<float>(),
        g_k.data_ptr<float>(),
        g_v.data_ptr<float>(),
        g_z.data_ptr<float>(),
        g_b.data_ptr<float>(),
        g_a.data_ptr<float>(),
        g_attn_out.data_ptr<float>(),
        g_mlp_intermediate.data_ptr<float>(),
        g_normalized.data_ptr<float>(),
        full_layer_idx.data_ptr<int>(),
        linear_layer_idx.data_ptr<int>(),
        static_cast<int>(num_blocks),
        static_cast<int>(num_layers),
        position,
        cache_len,
        static_cast<int>(max_seq_len),
        static_cast<float>(attn_scale),
        nullptr,
        stream
    );

    cudaStreamSynchronize(stream);

    int token_id = output_token.cpu().item<int>();
    return token_id;
}

std::tuple<int, at::Tensor> decode_with_logits(
    int input_token_id,
    int position,
    int cache_len,
    at::Tensor layer_weights_bytes,
    at::Tensor embed_weight,
    at::Tensor final_norm_weight,
    at::Tensor lm_head_weight,
    at::Tensor cos_table,
    at::Tensor sin_table,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor conv_state,
    at::Tensor recurrent_state,
    at::Tensor hidden_buffer,
    at::Tensor g_activations,
    at::Tensor g_residual,
    at::Tensor g_q,
    at::Tensor g_k,
    at::Tensor g_v,
    at::Tensor g_z,
    at::Tensor g_b,
    at::Tensor g_a,
    at::Tensor g_attn_out,
    at::Tensor g_mlp_intermediate,
    at::Tensor g_normalized,
    at::Tensor full_layer_idx,
    at::Tensor linear_layer_idx,
    int64_t num_blocks,
    int64_t num_layers,
    int64_t max_seq_len,
    double attn_scale)
{
    auto logits = at::empty({248320},
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    auto output_token = at::empty({1},
        at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream();

    launch_qwen3_5_ldg_decode_with_logits(
        input_token_id,
        output_token.data_ptr<int>(),
        logits.data_ptr<float>(),
        embed_weight.data_ptr<at::BFloat16>(),
        reinterpret_cast<const void*>(layer_weights_bytes.data_ptr<uint8_t>()),
        final_norm_weight.data_ptr<at::BFloat16>(),
        lm_head_weight.data_ptr<at::BFloat16>(),
        cos_table.data_ptr<at::BFloat16>(),
        sin_table.data_ptr<at::BFloat16>(),
        k_cache.data_ptr<at::BFloat16>(),
        v_cache.data_ptr<at::BFloat16>(),
        conv_state.data_ptr<float>(),
        recurrent_state.data_ptr<float>(),
        hidden_buffer.data_ptr<at::BFloat16>(),
        g_activations.data_ptr<float>(),
        g_residual.data_ptr<float>(),
        g_q.data_ptr<float>(),
        g_k.data_ptr<float>(),
        g_v.data_ptr<float>(),
        g_z.data_ptr<float>(),
        g_b.data_ptr<float>(),
        g_a.data_ptr<float>(),
        g_attn_out.data_ptr<float>(),
        g_mlp_intermediate.data_ptr<float>(),
        g_normalized.data_ptr<float>(),
        full_layer_idx.data_ptr<int>(),
        linear_layer_idx.data_ptr<int>(),
        static_cast<int>(num_blocks),
        static_cast<int>(num_layers),
        position,
        cache_len,
        static_cast<int>(max_seq_len),
        static_cast<float>(attn_scale),
        nullptr,
        stream
    );

    cudaStreamSynchronize(stream);

    int token_id = output_token.cpu().item<int>();
    return std::make_tuple(token_id, logits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &decode, "Qwen3.5 persistent decode (argmax only)");
    m.def("decode_with_logits", &decode_with_logits, "Qwen3.5 persistent decode with full logits");
}
