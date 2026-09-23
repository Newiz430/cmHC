#ifndef MHC_PRE_CMHC_BACKWARD_KEY_ARCH22_H
#define MHC_PRE_CMHC_BACKWARD_KEY_ARCH22_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_cmhc_backward_data_arch22.h"

ASCENDC_TPL_ARGS_DECL(MhcPreCmhcBackward,
    ASCENDC_TPL_BOOL_DECL(TILINGKEY, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
        ASCENDC_TPL_BOOL_SEL(TILINGKEY, 0, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(MhcPreCmhcBackwardArch22TilingData)
    ),
)

#endif  // MHC_PRE_CMHC_BACKWARD_KEY_ARCH22_H
