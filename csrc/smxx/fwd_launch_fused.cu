#include "fwd_kernel_fused.cuh"

#include <stdexcept>
#include <unordered_set>

namespace flash_kda::fused {
namespace {

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

CUtensorMap make_tma(void* data, std::initializer_list<uint64_t> shape,
                     std::initializer_list<uint64_t> strides, std::initializer_list<uint32_t> box,
                     CUtensorMapSwizzle swizzle) {
    CUtensorMap map{};
    uint32_t element_strides[4] = {1, 1, 1, 1};
    auto status =
        cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, shape.size(), data, shape.begin(),
                               strides.begin(), box.begin(), element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        char const* message = nullptr;
        cuGetErrorString(status, &message);
        throw std::runtime_error(message ? message : "Invalid fused TMA descriptor");
    }
    return map;
}

template <int NumHeads, bool FullChunks> void const* kernel_for_state(bool state_fp32) {
    auto kernel = state_fp32 ? _flash_kda_fwd_fused_vtile_direct<NumHeads, FullChunks, true>
                             : _flash_kda_fwd_fused_vtile_direct<NumHeads, FullChunks, false>;
    return reinterpret_cast<void const*>(kernel);
}

template <int NumHeads> void const* kernel_for_chunks(bool full_chunks, bool state_fp32) {
    return full_chunks ? kernel_for_state<NumHeads, true>(state_fp32)
                       : kernel_for_state<NumHeads, false>(state_fp32);
}

} // namespace

// ==================== launch_fwd_fused ====================
void launch_fwd_fused(FusedParams const& params, int total_tokens, bool full_chunks, bool state_fp32,
                      cudaStream_t stream) {
    if (params.num_sequences == 0)
        return;

    // Compile-time scalar/stride specialization, not an algorithm selector.
    bool default_scalars = params.scale == 0.08838834764831845f && params.lower_bound == -5.0f;
    void const* kernel =
        default_scalars && params.num_heads == 64   ? kernel_for_chunks<64>(full_chunks, state_fp32)
        : default_scalars && params.num_heads == 96 ? kernel_for_chunks<96>(full_chunks, state_fp32)
                                                    : kernel_for_chunks<0>(full_chunks, state_fp32);

    constexpr int kThreads     = 1024;
    constexpr int kSharedBytes = FusedLayouts<0, false>::kSmemTotal;
    thread_local std::unordered_set<void const*> configured[32];
    int device;
    check_cuda(cudaGetDevice(&device));
    if (device >= 32 || !configured[device].count(kernel)) {
        check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
        if (device < 32)
            configured[device].insert(kernel);
    }

    FusedTensorMaps maps{};
    if (total_tokens > 0) {
        uint64_t H = params.num_heads, T = total_tokens;
        auto qkv_map = [&](void* data) {
            return make_tma(data, {64, T, H, 2}, {H * 256, 256, 128}, {64, 32, 1, 2},
                            CU_TENSOR_MAP_SWIZZLE_128B);
        };
        maps = {qkv_map(params.q), qkv_map(params.k), qkv_map(params.v),
                make_tma(params.g, {128, H, T}, {256, H * 256}, {128, 1, 32}, CU_TENSOR_MAP_SWIZZLE_NONE),
                qkv_map(params.out)};
    }
    // CUDA owns a copy of the grid-constant descriptors on launch and graph capture.
    void* args[] = {&maps, const_cast<FusedParams*>(&params)};
    check_cuda(cudaLaunchKernel(kernel, dim3(params.num_sequences * params.num_heads), dim3(kThreads), args,
                                kSharedBytes, stream));
}

} // namespace flash_kda::fused
