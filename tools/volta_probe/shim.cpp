#include <torch/extension.h>
#include <optional>
#include <string>
#include <vector>

at::Tensor flash_attention_prefill_paged(
    const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
    std::optional<at::Tensor>& out_, const at::Tensor& block_table,
    const at::Tensor& seq_lens, const float softmax_scale,
    const std::string& kv_cache_dtype, const float k_scale, const float v_scale,
    const bool is_causal, const int window_size_left,
    const int window_size_right, const std::optional<at::Tensor>& anchor_lens,
    const int64_t anchored_window);

at::Tensor prefill(const at::Tensor& q, const at::Tensor& k_cache,
                   const at::Tensor& v_cache, const at::Tensor& block_table,
                   const at::Tensor& seq_lens, double softmax_scale) {
  std::optional<at::Tensor> out;
  std::optional<at::Tensor> anchor;
  return flash_attention_prefill_paged(
      q, k_cache, v_cache, out, block_table, seq_lens,
      static_cast<float>(softmax_scale), "auto", 1.0f, 1.0f,
      /*is_causal=*/true, -1, -1, anchor, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("prefill", &prefill); }
