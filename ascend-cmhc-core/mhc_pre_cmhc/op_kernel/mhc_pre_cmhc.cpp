#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_cmhc_m_split_core.h"
#include "mhc_pre_cmhc_base.h"
using namespace MhcPreCmhc;

using namespace AscendC;

// For cmhc add: 入口签名加 permMats（gamma 之后）+ coeff（hPre 之后）
extern "C" __global__ __aicore__ void mhc_pre_cmhc(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma,
                                             GM_ADDR permMats, GM_ADDR hin, GM_ADDR hPost, GM_ADDR hRes,
                                             GM_ADDR invRms, GM_ADDR hMix, GM_ADDR hPre, GM_ADDR coeff,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    (void)gamma;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(MhcPreCmhcMembaseTilingData, tiling_data_in, tiling);
    const MhcPreCmhcMembaseTilingData *__restrict tilingData = &tiling_data_in;

    if (TILING_KEY_IS(0)) {
        // -----------split m--------------
        TPipe pipe;
        MhcPreCmhc::MhcPreCmhcStage1<DTYPE_X, true> op1;
        if (tilingData->d >= MhcPreCmhc::PRE_NZ_D_THRESHOLD &&
            tilingData->bs >= (tilingData->d >= 7168 ? MhcPreCmhc::PRE_NZ_BS_THRESHOLD_D7168
                                                      : MhcPreCmhc::PRE_NZ_BS_THRESHOLD_D3584)) {
            op1.cubeCompute_.mm1Nz_.Init(&tilingData->mm1TilingData, &pipe);
            op1.cubeCompute_.mm1Nz_.SetSubBlockIdx(0);
        } else {
            op1.cubeCompute_.mm1Nd_.Init(&tilingData->mm1TilingData, &pipe);
            op1.cubeCompute_.mm1Nd_.SetSubBlockIdx(0);
        }
        op1.Init(x, phi, invRms, hMix, userWs, tilingData, &pipe);
        op1.Process();
        pipe.Destroy();

        TPipe pipeStage2;
        MhcPreCmhc::MhcPreCmhcStage2<DTYPE_X, true> op2;
        op2.Init(x, alpha, bias, hin, hPost, hRes,
                    hPre, hMix, invRms, coeff,
                    userWs, tilingData, &pipeStage2);
        op2.Process(false);
        pipeStage2.Destroy();

        // For cmhc add: Stage3 独立 cube pass，大 GEMM coeff[bs,n!] @ perm_mats[n!,n²] → hRes[bs,n²]
        TPipe pipeStage3;
        MhcPreCmhc::MhcPreCmhcStage3<DTYPE_X> op3;
        op3.cubeCompute_.mmProj_.Init(&tilingData->mmProjTilingData, &pipeStage3);
        op3.cubeCompute_.mmProj_.SetSubBlockIdx(0);
        op3.Init(userWs, permMats, hRes, tilingData, &pipeStage3);
        op3.Process();
        pipeStage3.Destroy();
    }
    // #endif
}
#endif
