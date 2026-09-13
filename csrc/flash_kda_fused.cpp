#include "flash_kda_fused.h"
#include "sm100/fused_params.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <limits>

void fwd_fused(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    torch::Tensor beta, float scale, torch::Tensor out,
    torch::Tensor A_log, torch::Tensor dt_bias, float lower_bound,
    std::optional<torch::Tensor> initial_state,
    std::optional<torch::Tensor> final_state,
    std::map<std::string, torch::Tensor> const& buffers,
    std::map<std::string, int64_t> const& config
) {
    TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.size(3) == 128,
                "q must be a CUDA tensor [B,T,H,128]");
    c10::cuda::CUDAGuard guard(q.device());
    int major, minor;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.get_device()));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, q.get_device()));
    TORCH_CHECK(major == 10 && (minor == 0 || minor == 3),
                "Fused prefill kernels require SM100/SM103");
    auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    auto check_tensor = [&](torch::Tensor const& t, char const* name, torch::ScalarType dtype) {
        TORCH_CHECK(t.device() == q.device() && t.is_contiguous() && t.scalar_type() == dtype,
                    name, " has an incompatible device, layout, or dtype");
    };
    for (auto const& tensor : {q, k, v, g, out}) {
        check_tensor(tensor, "q/k/v/g/out", torch::kBFloat16);
        TORCH_CHECK(tensor.sizes() == q.sizes(), "q/k/v/g/out must have identical shapes");
    }
    int64_t H = q.size(2), T = q.size(0) * q.size(1);
    TORCH_CHECK(H > 0 && T > 0 && T <= std::numeric_limits<int>::max(), "Invalid fused shape");
    check_tensor(beta, "beta", torch::kBFloat16);
    TORCH_CHECK(beta.dim() == 3 && beta.size(0) == q.size(0) && beta.size(1) == q.size(1)
                && beta.size(2) == H, "beta must be [B,T,H]");
    check_tensor(A_log, "A_log", torch::kFloat32);
    check_tensor(dt_bias, "dt_bias", torch::kFloat32);
    TORCH_CHECK(A_log.numel() == H && dt_bias.numel() == H * 128, "Invalid gate parameter shape");
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(lower_bound) && lower_bound <= 0,
                "scale must be finite and lower_bound must be finite and nonpositive");
    for (auto const& [name, tensor] : buffers) {
        TORCH_CHECK(tensor.device() == q.device() && tensor.is_contiguous(),
                    "Fused buffer ", name, " must be contiguous on the input device");
    }
    auto const& cu = buffers.at("cu_seqlens");
    check_tensor(cu, "cu_seqlens", torch::kInt64);
    TORCH_CHECK(cu.dim() == 1 && cu.numel() >= 2, "cu_seqlens must be [N+1]");
    int64_t N = cu.numel() - 1;
    auto const& order = buffers.at("seq_order");
    check_tensor(order, "seq_order", torch::kInt32);
    TORCH_CHECK(order.numel() == N, "seq_order must contain N entries");
    bool fp32 = initial_state ? initial_state->scalar_type() == torch::kFloat32
                             : final_state && final_state->scalar_type() == torch::kFloat32;
    for (auto const& state : {initial_state, final_state}) {
        if (!state) continue;
        check_tensor(*state, "state", fp32 ? torch::kFloat32 : torch::kBFloat16);
        TORCH_CHECK(state->dim() == 4 && state->size(0) == N && state->size(1) == H
                    && state->size(2) == 128 && state->size(3) == 128,
                    "state must be [N,H,128,128]");
    }
    auto const& carrier = buffers.at("beta_carrier");
    check_tensor(carrier, "beta_carrier", torch::kBFloat16);
    TORCH_CHECK(carrier.dim() == 2, "beta_carrier must be two-dimensional");

    flash_kda::fused::FusedParams p{};
    p.q = q.data_ptr(); p.k = k.data_ptr(); p.v = v.data_ptr(); p.g = g.data_ptr();
    p.beta = beta.data_ptr(); p.out = out.data_ptr();
    p.A_log = A_log.data_ptr(); p.dt_bias = dt_bias.data_ptr();
    p.cu_seqlens = cu.data_ptr(); p.seq_order = order.data_ptr();
    p.num_heads = H; p.num_sequences = N;
    p.use_initial_state = initial_state.has_value(); p.store_final_state = final_state.has_value();
    p.scale = scale; p.lower_bound = lower_bound;
    p.beta_token_stride = H; p.state_slot_stride = H * 128 * 128;
    p.initial_state = initial_state ? initial_state->data_ptr() : out.data_ptr();
    p.final_state = final_state ? final_state->data_ptr() : out.data_ptr();
    p.initial_state_f32 = p.initial_state; p.final_state_f32 = p.final_state;
    if (auto it = buffers.find("tile_schedule"); it != buffers.end()) {
        p.tile_schedule = it->second.data_ptr();
        p.tile_schedule_counts = buffers.at("tile_schedule_counts").data_ptr();
        p.schedule_stride = config.at("schedule_stride");
    }
    if (auto it = config.find("uniform_seq_len"); it != config.end()) {
        p.uniform_seq_len = it->second;
        p.persistent_tasks = config.at("persistent_tasks");
        p.persistent_stride = config.at("persistent_stride");
    }
    flash_kda::fused::FusedLaunchConfig launch_config{
        int(config.at("variant")), int(config.at("blocks")),
        int(T), int(carrier.size(1)), int(carrier.size(0)), carrier.data_ptr()
    };
    flash_kda::fused::launch_fwd_fused(p, launch_config, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
