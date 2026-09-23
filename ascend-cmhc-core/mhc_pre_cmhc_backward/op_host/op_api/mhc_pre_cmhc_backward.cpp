#include "mhc_pre_cmhc_backward.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MhcPreCmhcBackward);

const std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *>
MhcPreCmhcBackward(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *gradHIn,
                   const aclTensor *gradHPost, const aclTensor *gradHRes, const aclTensor *invRms,
                   const aclTensor *hMix, const aclTensor *hPre, const aclTensor *hPost, const aclTensor *gamma,
                   const aclTensor *perms, const aclTensor *coeff, float hcEps, aclOpExecutor *executor)
{
    L0_DFX(MhcPreCmhcBackward, x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, perms,
           coeff, hcEps);

    DataType outType = DataType::DT_FLOAT;
    Format format = Format::FORMAT_ND;
    auto outGradX = executor->AllocTensor(gradHIn->GetDataType(), format, format);
    auto outGradPhi = executor->AllocTensor(outType, format, format);
    auto outGradAlpha = executor->AllocTensor(outType, format, format);
    auto outGradBias = executor->AllocTensor(outType, format, format);
    auto outGradGamma = executor->AllocTensor(outType, format, format);

    auto ret = INFER_SHAPE(
        MhcPreCmhcBackward,
        OP_INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, perms, coeff),
        OP_OUTPUT(outGradX, outGradPhi, outGradAlpha, outGradBias, outGradGamma), OP_ATTR(hcEps));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr),
                        "MhcPreCmhcBackward InferShape failed.");
    auto ret1 = ADD_TO_LAUNCHER_LIST_AICORE(
        MhcPreCmhcBackward,
        OP_INPUT(x, phi, alpha, gradHIn, gradHPost, gradHRes, invRms, hMix, hPre, hPost, gamma, perms, coeff),
        OP_OUTPUT(outGradX, outGradPhi, outGradAlpha, outGradBias, outGradGamma), OP_ATTR(hcEps));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret1 != ACLNN_SUCCESS,
                                         return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr),
                                         "MhcPreCmhcBackward ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return std::tuple(outGradX, outGradPhi, outGradAlpha, outGradBias, outGradGamma);
}
} // namespace l0op
