#include "smxx/fwd_kernel_fused.cuh"

namespace flash_kda::fused {

FusedKernelSpec get_fused_vtile_persistent_spec(bool state_fp32, int num_heads) {
    if (num_heads == 64) {
        auto kernel = state_fp32 ? _flash_kda_fwd_fused_vtile_persistent<true, 64>
                                 : _flash_kda_fwd_fused_vtile_persistent<false, 64>;
        return {reinterpret_cast<void const *>(kernel), 1024, 226304, 32, 128, 4};
    }
    auto kernel = state_fp32 ? _flash_kda_fwd_fused_vtile_persistent<true, 96>
                             : _flash_kda_fwd_fused_vtile_persistent<false, 96>;
    return {reinterpret_cast<void const *>(kernel), 1024, 226304, 32, 128, 4};
}

} // namespace flash_kda::fused
