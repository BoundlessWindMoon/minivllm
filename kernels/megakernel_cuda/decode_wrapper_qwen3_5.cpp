#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <tuple>

// Forward declaration of the C-compatible launcher (defined in decode_qwen3_5.cu)
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
);

// =============================================================================
// Python bindings
// =============================================================================

void run_layer(
    at::Tensor hidden_in,
    at::Tensor hidden_out,
    int64_t layer_type,
    at::Tensor layer_weights,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor conv_state,
    at::Tensor recurrent_state,
    at::Tensor cos_table,
    at::Tensor sin_table,
    int64_t position,
    int64_t cache_len,
    int64_t max_seq_len,
    int64_t layer_idx,
    at::Tensor scratch
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_qwen3_5_layer(
        hidden_in.data_ptr<at::BFloat16>(),
        hidden_out.data_ptr<at::BFloat16>(),
        static_cast<int>(layer_type),
        reinterpret_cast<const void*>(layer_weights.data_ptr<uint8_t>()),
        k_cache.data_ptr<at::BFloat16>(),
        v_cache.data_ptr<at::BFloat16>(),
        conv_state.data_ptr<float>(),
        recurrent_state.data_ptr<float>(),
        cos_table.data_ptr<at::BFloat16>(),
        sin_table.data_ptr<at::BFloat16>(),
        static_cast<int>(position),
        static_cast<int>(cache_len),
        static_cast<int>(max_seq_len),
        static_cast<int>(layer_idx),
        scratch.data_ptr<float>(),
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_layer", &run_layer,
          "Run one Qwen3.5 layer (full or linear attention)");
}
