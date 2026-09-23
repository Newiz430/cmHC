#ifndef OP_API_MHC_PRE_CMHC_H_
#define OP_API_MHC_PRE_CMHC_H_

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
// For cmhc add: 签名加 permMats 输入 + 返回 tuple 加 coeff（7 个输出：hin/hPost/hRes/invRms/hMix/hPre/coeff）
const std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *>
MhcPreCmhc(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
       const aclTensor *gammaOptional, const aclTensor *permMats, int64_t outFlag, double normEps, double hcEps, aclOpExecutor *executor);
}

#endif // OP_API_MHC_PRE_CMHC_H_
