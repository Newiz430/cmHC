#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_MHC_PRE_CMHC_BACKWARD_H
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_MHC_PRE_CMHC_BACKWARD_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *>
MhcPreCmhcBackward(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *gradHIn,
               const aclTensor *gradHPost, const aclTensor *gradHRes, const aclTensor *invRms,
               const aclTensor *hMix, const aclTensor *hPre, const aclTensor *hPost, const aclTensor *gamma,
               const aclTensor *perms, const aclTensor *coeff, float hcEps, aclOpExecutor *executor);
}

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_MHC_PRE_CMHC_BACKWARD_H
