#ifndef OP_API_INC_MHC_PRE_CMHC_H
#define OP_API_INC_MHC_PRE_CMHC_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPreCmhcGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：MhcPreCmhc正向算子
 * @param [in] x: 输入tensor，shape为[B,S,N,D]或[T,N,D]，数据类型支持：BF16/FP16。
 * @param [in] phi: 输入tensor，shape为[n!+2n, nD]，数据类型支持：FP32。CMHC 只支持 alpha=[3]。
 * @param [in] alpha: 输入tensor，shape为[3]，数据类型支持：FP32。CMHC 必然有 hRes 输出。
 * @param [in] bias: 输入tensor，shape为[n!+2n]，数据类型支持：FP32。
 * @param [in] gammaOptional: 可选输入tensor，RMSNorm gamma，shape为[n, D]，数据类型支持：FP32（kernel 当前忽略，等价 gamma=1）。
 * @param [in] permMats: 输入tensor，已含 gamma-blend 的置换矩阵，shape为[n!,n,n]，数据类型支持：FP32。For cmhc add。
 * @param [in] normEps: 归一化epsilon参数。
 * @param [in] hcEps: h_pre sigmoid 后的加性 eps 偏置参数，建议值 1e-6。
 * @param [out] hIn: 输出tensor，shape为[B,S,D]或[T,D]，数据类型：BF16/FP16。
 * @param [out] hPost: 输出tensor，shape为[B,S,N]或[T,N]，数据类型：FP32。
 * @param [out] hRes: 输出tensor，双随机矩阵（已 softmax+置换矩阵 einsum），shape为[B,S,N*N]或[T,N*N]（CANN/ACLNN 边界，reshape 到 [B,S,N,N] 留给 MindSpore），数据类型：FP32。For cmhc add。
 * @param [out] invRmsOptional: 可选输出tensor，shape为[B,S]或[T]，数据类型：FP32。与 hMix/hPre/coeff 互存（全空或全非空）。
 * @param [out] hMixOptional: 可选输出tensor，shape为[B,S,N!+2N]或[T,N!+2N]，数据类型：FP32。互存。
 * @param [out] hPreOptional: 可选输出tensor，shape为[B,S,N]或[T,N]，数据类型：FP32。互存。
 * @param [out] coeffOptional: 可选输出tensor，softmax 输出，shape为[B,S,N!]或[T,N!]，数据类型：FP32。供反向，免重算 softmax。互存。For cmhc add。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
// For cmhc add: 签名加 permMats（gammaOptional 后）+ coeffOptional（hPreOptional 后）
ACLNN_API aclnnStatus aclnnMhcPreCmhcGetWorkspaceSize(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                                  const aclTensor *bias, const aclTensor *gammaOptional, const aclTensor *permMats,
                                                  double normEps, double hcEps, aclTensor *hIn, aclTensor *hPost, aclTensor *hRes,
                                                  aclTensor *invRmsOptional, aclTensor *hMixOptional,
                                                  aclTensor *hPreOptional, aclTensor *coeffOptional, uint64_t *workspaceSize,
                                                  aclOpExecutor **executor);

/**
 * @brief aclnnMhcPreCmhc 的第二段接口，执行算子计算。
 * @domain aclnn_ops_infer
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 aclnnMhcPreCmhcGetWorkspaceSize 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMhcPreCmhc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MHC_PRE_CMHC_H
