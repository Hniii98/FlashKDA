#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

namespace flash_kda::fused {

// Operands and scheduling metadata shared by the native fused entry points.
// BF16/FP32 state aliases preserve the original upstream endpoint instructions.
struct FusedParams {
    // Token inputs, gate parameters, and output.
    void *q{};
    void *k{};
    void *v{};
    void *g{};
    void *beta{};
    void *out{};
    void *A_log{};
    void *dt_bias{};
    float scale{};
    float lower_bound{};

    // Recurrent state. Only the selected endpoint dtype is accessed.
    void *initial_state{};
    void *final_state{};
    void *initial_state_f32{};
    void *final_state_f32{};
    int use_initial_state{};
    int store_final_state{};
    long long state_slot_stride{};

    // Sequence boundaries and contiguous tensor strides.
    void *cu_seqlens{};
    void *seq_order{};
    int num_heads{};
    int num_sequences{};
    long long beta_token_stride{};

    // Scalar LPT or uniform V-tile persistent schedule.
    void *tile_schedule{};
    void *tile_schedule_counts{};
    int schedule_stride{};
    int uniform_seq_len{};
    int persistent_tasks{};
    int persistent_stride{};
};

struct FusedTensorMaps {
    CUtensorMap q, k, v, g, beta, out;
};

struct FusedKernelSpec {
    void const *kernel;
    int threads, smem_bytes, chunk, value_rows, value_tma_rank;
};
FusedKernelSpec get_fused_spec(int variant);

struct FusedLaunchConfig {
    int variant, blocks, total_tokens, beta_heads, beta_tokens;
    void *beta_carrier;
};
void launch_fwd_fused(FusedParams &params, FusedLaunchConfig const &config, cudaStream_t stream);

} // namespace flash_kda::fused
