#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "arch22/mhc_pre_cmhc_backward_arch22_tiling.h"
#include "mhc_pre_cmhc_backward_tiling.h"
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {

static ge::graphStatus Tiling4MhcPreCmhcBackward(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "MhcPreCmhcBackward is running.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE("TilingForMhcPreCmhcBackward", "Tiling platformInfo is null"),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Using arch22 tiling for ASCEND910B");
    return TilingMhcPreCmhcBackwardArch22(context);
}

static ge::graphStatus TilingPrepare4MhcPreCmhcBackward(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("[TilingPrepare4mHC]", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MhcPreCmhcBackwardCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);

    OP_LOGI(context->GetNodeName(), "parse compile info success l1Size:%lu, l2Size:%lu, coreNum:%lu",
            compileInfoPtr->l1Size, compileInfoPtr->l2Size, compileInfoPtr->aicNum);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the MhcPreCmhcBackward op.
IMPL_OP_OPTILING(MhcPreCmhcBackward)
    .Tiling(Tiling4MhcPreCmhcBackward)
    .TilingParse<MhcPreCmhcBackwardCompileInfo>(TilingPrepare4MhcPreCmhcBackward);
} // namespace optiling
