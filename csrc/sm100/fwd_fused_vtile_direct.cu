#include "smxx/fwd_kernel_fused.cuh"

namespace flash_kda::fused {

FusedKernelSpec get_fused_vtile_direct_spec(bool state_fp32) {
    auto kernel =
        state_fp32 ? _flash_kda_fwd_fused_vtile_direct<true> : _flash_kda_fwd_fused_vtile_direct<false>;
    return {reinterpret_cast<void const *>(kernel), 1024, 226304, 32, 128, 4};
}

} // namespace flash_kda::fused
