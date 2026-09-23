#ifndef MHC_PRE_CMHC_BACKWARD_ARCH22_TILING_H_
#define MHC_PRE_CMHC_BACKWARD_ARCH22_TILING_H_

#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"

namespace optiling {
ge::graphStatus TilingMhcPreCmhcBackwardArch22(gert::TilingContext* context);
}

#endif // MHC_PRE_CMHC_BACKWARD_ARCH22_TILING_H_
