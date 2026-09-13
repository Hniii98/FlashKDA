#include "fused_params.h"
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace flash_kda::fused {

namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

CUtensorMap make_tma(void *data, std::initializer_list<uint64_t> shape,
                     std::initializer_list<uint64_t> strides, std::initializer_list<uint32_t> box,
                     CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    CUtensorMap map{};
    uint32_t element_strides[5] = {1, 1, 1, 1, 1};
    auto status =
        cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, shape.size(), data, shape.begin(),
                               strides.begin(), box.begin(), element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        char const *message = nullptr;
        cuGetErrorString(status, &message);
        throw std::runtime_error(std::string("Fused TMA descriptor: ") + message);
    }
    return map;
}

void configure(FusedKernelSpec const &spec) {
    thread_local std::unordered_set<void const *> configured[32];
    int device;
    check_cuda(cudaGetDevice(&device));
    if (device >= 32 || configured[device].count(spec.kernel) == 0) {
        check_cuda(
            cudaFuncSetAttribute(spec.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, spec.smem_bytes));
        if (device < 32)
            configured[device].insert(spec.kernel);
    }
}
} // namespace

void launch_fwd_fused(FusedParams &params, FusedLaunchConfig const &config, cudaStream_t stream) {
    auto spec = get_fused_spec(config.variant);
    configure(spec);
    uint64_t const H = params.num_heads, T = config.total_tokens;
    uint32_t const chunk = spec.chunk, value_rows = spec.value_rows;

    auto qk_map = [&](void *data, uint32_t splits = 2) {
        return make_tma(data, {64, T, H, 2}, {H * 256, 256, 128}, {64, chunk, 1, splits},
                        CU_TENSOR_MAP_SWIZZLE_128B);
    };
    auto value_map = [&](void *data, uint32_t rows) {
        return make_tma(data, {128, H, T}, {256, H * 256}, {rows, 1, chunk},
                        rows == 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE);
    };

    // N16 uses two value panels; V-tile uses the Q/K-style rank-four map.
    // M64 assigns disjoint halves of value/output to its two CTAs per head.
    FusedTensorMaps tensor_maps{qk_map(params.q),
                                qk_map(params.k),
                                chunk == 16 ? make_tma(params.v, {64, H, T, 2}, {256, H * 256, 128},
                                                       {64, 1, 16, 1}, CU_TENSOR_MAP_SWIZZLE_128B)
                                : spec.value_tma_rank == 4 ? qk_map(params.v)
                                                           : value_map(params.v, value_rows),
                                value_map(params.g, 128),
                                make_tma(config.beta_carrier,
                                         {uint64_t(config.beta_heads), uint64_t(config.beta_tokens)},
                                         {uint64_t(config.beta_heads) * 2}, {8, chunk}),
                                qk_map(params.out, value_rows / 64)};
    // CUDA copies these grid-constant parameters at launch/capture. No global
    // descriptor buffer, publication kernel, or graph-lifetime cache is needed.
    void *args[] = {&tensor_maps, &params};
    check_cuda(cudaLaunchKernel(spec.kernel, dim3(config.blocks), dim3(spec.threads), args, spec.smem_bytes,
                                stream));
}

} // namespace flash_kda::fused
