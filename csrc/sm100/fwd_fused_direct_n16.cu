#include "smxx/fwd_kernel_fused.cuh"

namespace flash_kda::fused {

FusedKernelSpec get_fused_direct_n16_spec(bool state_fp32) {
    auto kernel = state_fp32 ? _flash_kda_fwd_fused_direct_n16<true> : _flash_kda_fwd_fused_direct_n16<false>;
    return {reinterpret_cast<void const *>(kernel), 1024, 117376, 16, 128, 4};
}

} // namespace flash_kda::fused
