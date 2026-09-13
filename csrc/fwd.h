#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>

template <int D, bool HasStateIn = true, bool HasStateOut = true, bool StateFP32 = false, bool IsVarlen = true>
void launch_fwd(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    float scale,
    void* final_state_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    cudaStream_t stream
);

namespace flash_kda::fused {

// All fused specializations share the same VTile Direct operands and schedule.
struct FusedParams {
    void *q, *k, *v, *g, *beta, *out;
    void *A_log, *dt_bias;
    void *initial_state, *final_state;
    void *initial_state_f32, *final_state_f32;
    int64_t const* cu_seqlens;
    int const* seq_order;
    int num_heads, num_sequences, seq_len;
    int use_initial_state, store_final_state;
    int64_t state_slot_stride;
    float scale, lower_bound;
};

struct FusedTensorMaps {
    CUtensorMap q, k, v, g, out;
};

void launch_fwd_fused(
    FusedParams const& params,
    int total_tokens,
    bool full_chunks,
    bool state_fp32,
    cudaStream_t stream
);

} // namespace flash_kda::fused
