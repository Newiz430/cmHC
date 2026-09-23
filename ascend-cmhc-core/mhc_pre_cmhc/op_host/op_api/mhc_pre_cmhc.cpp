#include "mhc_pre_cmhc.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MhcPreCmhc);

// For cmhc add: 签名加 permMats 输入 + 返回 tuple 加 coeff（7 个输出：hin/hPost/hRes/invRms/hMix/hPre/coeff）
const std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *>
MhcPreCmhc(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
       const aclTensor *gammaOptional, const aclTensor *permMats, int64_t outFlag, double normEps, double hcEps, aclOpExecutor *executor)
{
    L0_DFX(MhcPreCmhc, x, phi, alpha, bias, gammaOptional, permMats, outFlag, normEps, hcEps);

    DataType outType = DataType::DT_FLOAT;
    Format format = Format::FORMAT_ND;
    auto outHin = executor->AllocTensor(x->GetDataType(), format, format);
    auto outHpost = executor->AllocTensor(outType, format, format);
    auto outHres = executor->AllocTensor(outType, format, format);

    auto outInvRms = executor->AllocTensor(outType, format, format);
    auto outMmRes = executor->AllocTensor(outType, format, format);
    auto outHpre = executor->AllocTensor(outType, format, format);
    auto outCoeff = executor->AllocTensor(outType, format, format);  // For cmhc add: coeff 输出

    // For cmhc add: INFER_SHAPE 加 permMats 输入 + outCoeff 输出
    auto ret = INFER_SHAPE(MhcPreCmhc, OP_INPUT(x, phi, alpha, bias, gammaOptional, permMats),
                           OP_OUTPUT(outHin, outHpost, outHres, outInvRms, outMmRes, outHpre, outCoeff),
                           OP_ATTR(outFlag, static_cast<float>(normEps), static_cast<float>(hcEps)));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
                        "MhcPreCmhc InferShape failed.");
    // For cmhc add: ADD_TO_LAUNCHER 加 permMats 输入 + outCoeff 输出
    auto ret1 = ADD_TO_LAUNCHER_LIST_AICORE(MhcPreCmhc, OP_INPUT(x, phi, alpha, bias, gammaOptional, permMats),
                                            OP_OUTPUT(outHin, outHpost, outHres, outInvRms, outMmRes, outHpre, outCoeff),
                                            OP_ATTR(outFlag, static_cast<float>(normEps), static_cast<float>(hcEps)));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret1 != ACLNN_SUCCESS,
                                         return std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
                                         "MhcPreCmhc ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return std::tuple(outHin, outHpost, outHres, outInvRms, outMmRes, outHpre, outCoeff);
}
} // namespace l0op
