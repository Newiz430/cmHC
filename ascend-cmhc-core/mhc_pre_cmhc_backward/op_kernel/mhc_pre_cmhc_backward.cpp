#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "arch22/mhc_pre_cmhc_backward_kernel_arch22.h"
#include "arch22/mhc_pre_cmhc_backward_data_arch22.h"
#include "arch22/mhc_pre_cmhc_backward_key_arch22.h"

using namespace AscendC;

template <bool TILINGKEY>
__global__ __aicore__ void
mhc_pre_cmhc_backward(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR grad_h_in, GM_ADDR grad_h_post, GM_ADDR grad_h_res,
                      GM_ADDR inv_rms, GM_ADDR h_mix, GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR gamma, GM_ADDR perms,
                      GM_ADDR coeff, GM_ADDR grad_x, GM_ADDR grad_phi, GM_ADDR grad_alpha, GM_ADDR grad_bias,
                      GM_ADDR grad_gamma, GM_ADDR workspace, GM_ADDR tiling)
{
    // printf("mhc_pre_cmhc_backward start.\n");
    REGISTER_TILING_DEFAULT(MhcPreCmhcBackwardArch22TilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcPreCmhcBackwardArch22TilingData, tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    if constexpr (TILINGKEY) {
        MhcPreCmhcBackwardKernelArch22<DTYPE_X, DTYPE_PHI, true> op;
        op.mm1_.SetSubBlockIdx(0);
        op.mm1_.Init(&tilingData.mm1TilingData, &pipe);
        op.mm2_.SetSubBlockIdx(0);
        op.mm2_.Init(&tilingData.mm2TilingData, &pipe);

        op.Init(x, phi, h_pre, h_post, grad_h_in, grad_h_post, grad_h_res, alpha, h_mix, inv_rms, perms, coeff, grad_x,
                grad_phi, grad_alpha, grad_bias, usrWorkspace, &tilingData, &pipe);
        op.Process();
    } else {
        // TILINGKEY==0 即 context->GetDeterministic()!=1: 走非确定性路径。
        // 此处原为 <..., true>, 与 if 分支完全相同, 导致无论是否要求确定性都执行确定性
        // 分支 —— 多付一份 aicNum*fusionSize*(N*C) 的 workspace 归约 (ComputeDeterministic)
        // 与一次全局 SyncAll。
        MhcPreCmhcBackwardKernelArch22<DTYPE_X, DTYPE_PHI, false> op;
        op.mm1_.SetSubBlockIdx(0);
        op.mm1_.Init(&tilingData.mm1TilingData, &pipe);
        op.mm2_.SetSubBlockIdx(0);
        op.mm2_.Init(&tilingData.mm2TilingData, &pipe);

        op.Init(x, phi, h_pre, h_post, grad_h_in, grad_h_post, grad_h_res, alpha, h_mix, inv_rms, perms, coeff, grad_x,
                grad_phi, grad_alpha, grad_bias, usrWorkspace, &tilingData, &pipe);
        op.Process();
    }
}
