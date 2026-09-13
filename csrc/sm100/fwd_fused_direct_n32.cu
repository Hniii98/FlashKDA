#include "smxx/fwd_kernel_fused.cuh"

namespace flash_kda::fused {

FusedKernelSpec get_fused_direct_n32_spec(bool state_fp32) {
    auto kernel = state_fp32 ? _flash_kda_fwd_fused_direct_n32<true> : _flash_kda_fwd_fused_direct_n32<false>;
    return {reinterpret_cast<void const *>(kernel), 1024, 227968, 32, 128, 3};
}

} // namespace flash_kda::fused
