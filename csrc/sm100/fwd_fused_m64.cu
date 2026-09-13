#include "smxx/fwd_kernel_fused.cuh"

namespace flash_kda::fused {

FusedKernelSpec get_fused_m64_spec(bool state_fp32) {
    auto kernel = state_fp32 ? _flash_kda_fwd_fused_m64<true> : _flash_kda_fwd_fused_m64<false>;
    return {reinterpret_cast<void const *>(kernel), 1024, 227328, 32, 64, 3};
}

} // namespace flash_kda::fused
