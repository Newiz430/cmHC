#ifndef MHC_PRE_CMHC_CUBE_COMPUTE_H
#define MHC_PRE_CMHC_CUBE_COMPUTE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "mhc_pre_cmhc_base.h"
#include "lib/matmul_intf.h"
#include <cstdint>

using AscendC::GlobalTensor;
using AscendC::TPosition;
using namespace AscendC;

namespace MhcPreCmhc {

constexpr MatmulConfig MHC_PRE_GRAD_MM1_CFG = GetMDLConfig(false, false, 0, true, false, false, true, true, true, true, true, true, true);

#define MHC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM template<typename T, bool isFac>
#define MHC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS HcCubeCompute<T, isFac>

// 不切K使用的矩阵乘实现
MHC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
class HcCubeCompute {
public:
    __aicore__ inline HcCubeCompute(){};
    __aicore__ inline void Init(const GlobalTensor<float> &xGm, const GlobalTensor<float> &phiGm,
                                const GlobalTensor<float> &workspaceGlobalAB, int64_t bs, int64_t n, int64_t c, int64_t vecCoreNum);
    __aicore__ inline void ProcessMatmulXPhi(const int32_t taskOffset, const int32_t mm1M);

public:
    GlobalTensor<float> xGm_;
    GlobalTensor<float> phiGm_;
    GlobalTensor<float> workspaceGlobalAB_;
    // Repo already co-instantiates mm1_(NZ) + mmProj_(ND) in HcResProjCubeCompute, so two MatmulImpl types are safe.
    // Runtime selects by usePreNz_ (D≥3584→NZ CopyNZ2NZ fast read; D<3584→ND baseline avoids AIV NZ-conversion overhead).
    // Only the SELECTED matmul is Init'd + used; the other stays dormant (no L1 alloc).
    using ATypeND = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;   // ND path (baseline, D<3584)
    using ATypeNz = matmul::MatmulType<TPosition::GM, CubeFormat::NZ, T>;   // NZ path (Pre-NZ, D≥3584)
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<ATypeNz, BType, CType, CType, MHC_PRE_GRAD_MM1_CFG> mm1Nz_;  // NZ path (D≥3584)
    matmul::MatmulImpl<ATypeND, BType, CType, CType, MHC_PRE_GRAD_MM1_CFG> mm1Nd_;  // ND path (D<3584)

    int64_t n_, c_, curBs_;
    int64_t mm1K_, mm1M_, mm1N_;
    int64_t nzKAligned_ = 0;  // For cmhc add: NZ K-dim aligned to C0=8 (FP32); used only by NZ path
    bool usePreNz_ = false;   // For cmhc add: runtime Pre-NZ selector (D≥PRE_NZ_D_THRESHOLD)
    int64_t ping4Cub = 1;
    uint64_t vecCoreNum_ = 0;
    uint64_t blockIdx_ = 0;
};

MHC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void MHC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::Init(const GlobalTensor<float> &xGm,
                                                                          const GlobalTensor<float> &phiGm,
                                                                          const GlobalTensor<float> &workspaceGlobalAB,
                                                                          int64_t bs, int64_t n, int64_t c, int64_t vecCoreNum)
{
    blockIdx_ = GetBlockIdx();
    xGm_ = xGm;
    phiGm_ = phiGm;
    workspaceGlobalAB_ = workspaceGlobalAB;
    vecCoreNum_ = vecCoreNum;
    n_ = n;
    c_ = c;
    curBs_ = bs;

    if constexpr (isFac) {
        mm1N_= Factorial(n_) + 2 * n_;
    } else {
        mm1N_= n_ * n_ + 2 * n_;
    }
    mm1K_ = n_ * c_;
    nzKAligned_ = (mm1K_ + 7) / 8 * 8;  // For cmhc add: NZ K-dim ceil to C0=8 (FP32)
    // (D=3584→bs≥4096, D≥7168→bs≥2048) since Init only receives d via 'c' param, not total bs.
    usePreNz_ = (c_ >= PRE_NZ_D_THRESHOLD);
}

MHC_PRE_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void MHC_PRE_CUBE_COMPUTE_TEMPLATE_CLASS::ProcessMatmulXPhi(const int32_t taskOffset, const int32_t mm1M)
{
    if (mm1M <= 0)
        return;
    // Only the host-selected matmul (Init'd in mhc_pre_cmhc.cpp) is used; the matching tiling data is set by host.
    if (usePreNz_) {
        // NZ path: SetTensorA offset uses nzKAligned_ (NZ K-padded to C0=8) instead of mm1K_ (raw ND K)
        mm1Nz_.SetTensorA(xGm_[blockIdx_ * curBs_ * 2 * nzKAligned_ + ping4Cub * vecCoreNum_ * curBs_ * nzKAligned_]);
        mm1Nz_.SetTensorB(phiGm_, true);
        // SetOrgShape M=curBs_*2 (full GM NZ M-dim, constant stride); SetSingleShape M=mm1M (actual task count).
        // NZ stride between K0-blocks = Ceil(orgHeight,16)*16*C0. With orgHeight=curBs_*2 (always mult of 16), stride is constant.
        // singleShape M=mm1M handles tail cases (mm1M < curBs_*2) — matmul reads with full stride but only processes mm1M rows.
        mm1Nz_.SetOrgShape(curBs_ * 2, mm1N_, mm1K_);
        mm1Nz_.SetSingleShape(mm1M, mm1N_, mm1K_);
        mm1Nz_.template IterateAll<false> (workspaceGlobalAB_[taskOffset * mm1N_]);
        mm1Nz_.End();
    } else {
        // ND path (baseline): SetTensorA offset uses mm1K_ (raw ND K), SetOrgShape/SetSingleShape both M=mm1M
        mm1Nd_.SetTensorA(xGm_[blockIdx_ * curBs_ * 2 * mm1K_ + ping4Cub * vecCoreNum_ * curBs_ * mm1K_]);
        mm1Nd_.SetTensorB(phiGm_, true);
        mm1Nd_.SetOrgShape(mm1M, mm1N_, mm1K_);
        mm1Nd_.SetSingleShape(mm1M, mm1N_, mm1K_);
        mm1Nd_.template IterateAll<false> (workspaceGlobalAB_[taskOffset * mm1N_]);
        mm1Nd_.End();
    }
    ping4Cub = 1 - ping4Cub;
}

// For cmhc add: HcResProjCubeCompute —— Stage3 大 GEMM coeff[bs,n!] @ perm_mats[n!,n²] → hRes[bs,n²]
// BType 第4模板参数 false（perm_mats 物理布局 [K,N]=[n!,n²]，不转置；对比 HcCubeCompute 的 phiGm_ 是 [N,K] 才用 true）
template<typename T>
class HcResProjCubeCompute {
public:
    __aicore__ inline HcResProjCubeCompute(){};
    __aicore__ inline void Init(const GlobalTensor<float> &coeffWsGm, const GlobalTensor<float> &permMatsGm,
                                const GlobalTensor<float> &hResGm, int64_t nPerm, int64_t nn) {
        coeffWsGm_ = coeffWsGm;
        permMatsGm_ = permMatsGm;
        hResGm_ = hResGm;
        nPerm_ = nPerm;
        nn_ = nn;
    }
    // For cmhc add: curM=min(mPerCore, bs-taskOffset)（v1.6 修正，避免 CeilDiv 尾核越界）
    __aicore__ inline void ProcessMatmulCoeffPerm(int32_t taskOffset, int32_t curM, int64_t nPerm, int64_t nn) {
        if (curM <= 0) return;
        mmProj_.SetTensorA(coeffWsGm_[taskOffset * nPerm]);           // A: coeff[本核 curM, n!]
        mmProj_.SetTensorB(permMatsGm_, false);                       // B: perm_mats[n!, n²]，isTrans=false
        mmProj_.SetOrgShape(curM, nn, nPerm);                         // M=curM, N=n², K=n!
        mmProj_.SetSingleShape(curM, nn, nPerm);                      // 每核单独 shape（IterateAll 内部分批）
        mmProj_.template IterateAll<false>(hResGm_[taskOffset * nn]);  // C: hRes[本核 curM, n²]
        mmProj_.End();
    }
public:
    GlobalTensor<float> coeffWsGm_;
    GlobalTensor<float> permMatsGm_;
    GlobalTensor<float> hResGm_;
    // For cmhc add: BType 第4参数 false（不转置）
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, false>;  // false!
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    matmul::MatmulImpl<AType, BType, CType, CType, MHC_PRE_GRAD_MM1_CFG> mmProj_;
    int64_t nPerm_ = 0;
    int64_t nn_ = 0;
};
}  // namespace MhcPreCmhc

#endif  // MHC_PRE_CMHC_CUBE_COMPUTE_H
