#include "fused_params.h"
#include <stdexcept>

namespace flash_kda::fused {
FusedKernelSpec get_fused_direct_n32_spec(bool state_fp32);
FusedKernelSpec get_fused_direct_n16_spec(bool state_fp32);
FusedKernelSpec get_fused_m64_spec(bool state_fp32);
FusedKernelSpec get_fused_scalar_lpt_spec(bool state_fp32, int num_heads);
FusedKernelSpec get_fused_vtile_direct_spec(bool state_fp32);
FusedKernelSpec get_fused_vtile_persistent_spec(bool state_fp32, int num_heads);

FusedKernelSpec get_fused_spec(int variant) {
    switch (variant) {
    case 7:
        return get_fused_direct_n32_spec(false);
    case 6:
        return get_fused_direct_n32_spec(true);
    case 14:
        return get_fused_direct_n16_spec(false);
    case 18:
        return get_fused_direct_n16_spec(true);
    case 30:
        return get_fused_m64_spec(false);
    case 32:
        return get_fused_m64_spec(true);
    case 10:
        return get_fused_scalar_lpt_spec(false, 64);
    case 17:
        return get_fused_scalar_lpt_spec(true, 64);
    case 12:
        return get_fused_scalar_lpt_spec(false, 96);
    case 8:
        return get_fused_scalar_lpt_spec(true, 96);
    case 29:
        return get_fused_vtile_direct_spec(false);
    case 25:
        return get_fused_vtile_direct_spec(true);
    case 26:
        return get_fused_vtile_persistent_spec(false, 64);
    case 27:
        return get_fused_vtile_persistent_spec(true, 64);
    case 28:
        return get_fused_vtile_persistent_spec(false, 96);
    case 24:
        return get_fused_vtile_persistent_spec(true, 96);
    default:
        throw std::invalid_argument("Unsupported fused schedule");
    }
}

} // namespace flash_kda::fused
