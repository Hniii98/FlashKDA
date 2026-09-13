#include "smxx/fwd_kernel_fused.cuh"

namespace flash_kda::fused {

FusedKernelSpec get_fused_scalar_lpt_spec(bool state_fp32, int num_heads) {
    if (num_heads == 64) {
        auto kernel = state_fp32 ? _flash_kda_fwd_fused_scalar_lpt<true, 64>
                                 : _flash_kda_fwd_fused_scalar_lpt<false, 64>;
        return {reinterpret_cast<void const *>(kernel), 1024, 231424, 32, 128, 4};
    }
    auto kernel =
        state_fp32 ? _flash_kda_fwd_fused_scalar_lpt<true, 96> : _flash_kda_fwd_fused_scalar_lpt<false, 96>;
    return {reinterpret_cast<void const *>(kernel), 1024, 231424, 32, 128, 4};
}

} // namespace flash_kda::fused
