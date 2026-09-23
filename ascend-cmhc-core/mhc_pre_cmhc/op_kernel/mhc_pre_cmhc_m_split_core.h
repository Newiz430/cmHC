#ifndef MHC_PRE_CMHC_M_SPLIT_A3_CORE_H
#define MHC_PRE_CMHC_M_SPLIT_A3_CORE_H

#include "kernel_operator.h"
#include "mhc_pre_cmhc_base.h"
#include "mhc_pre_cmhc_cube_compute.h"

namespace MhcPreCmhc {
using namespace AscendC;
template <typename T, bool isFac>
class MhcPreCmhcStage1 {
public:
    __aicore__ inline MhcPreCmhcStage1() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR invRms, GM_ADDR hcBeforeNorm,
                                GM_ADDR workspace, const MhcPreCmhcMembaseTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        needGrad_ = tilingData->needGrad;

        // 获取当前核心 ID
        uint64_t blockIdx = GetBlockIdx();
        uint64_t cubeCoreId = blockIdx;
        uint64_t vecCoreId = blockIdx;
        
        // 计算 workspace 偏移
        int64_t ncSize = tilingData->hcMult * tilingData->d;

        // 计算 Cube Core ID（Vector Core ID / 2）
        if ASCEND_IS_AIV {
            cubeCoreId = blockIdx / CV_RATIO;
            int64_t maxBsPerLoop = tilingData->stage1BsFactor;  // For cmhc add: 从 tiling 读（原硬编码 32），UB 分配与 bsFactor 匹配
            int64_t xQueSize = maxBsPerLoop * tilingData->stage1NcFactor;
            pipe->InitBuffer(xQue, NUM_TWO, xQueSize * sizeof(T));
            pipe->InitBuffer(xCastQue, NUM_TWO, xQueSize * sizeof(float));
            pipe->InitBuffer(tmpQue, maxBsPerLoop * tilingData->stage1NcFactor * sizeof(float));
            pipe->InitBuffer(sumQue, maxBsPerLoop * sizeof(float));
            LocalTensor<float> sumLocal = sumQue.Get<float>();
            pipe->InitBuffer(invRmsQue, NUM_TWO, maxBsPerLoop * sizeof(float));
            // Conditional on Pre-NZ path (D≥3584 && bs≥threshold) — ND path doesn't convert, so skip alloc.
            if (tilingData->d >= PRE_NZ_D_THRESHOLD &&
                tilingData->bs >= (tilingData->d >= 7168 ? PRE_NZ_BS_THRESHOLD_D7168 : PRE_NZ_BS_THRESHOLD_D3584)) {
                pipe->InitBuffer(nzBuf, maxBsPerLoop * tilingData->stage1NcFactor * sizeof(float));
                // D=3584 xCastWsGm bypass tested in Run#58 → regressed +3-9%, partial L2 caching beneficial.
                if (tilingData->d >= 7168) {
                    xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                    xCastWsGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                }
            }
        }

        // 设置 GM 地址
        xGm.SetGlobalBuffer((__gm__ T *)x);
        phiGm.SetGlobalBuffer((__gm__ float *)phi);
        xCastWsGm.SetGlobalBuffer((__gm__ float *)workspace);

        // 设置额外的 workspace 地址（needGrad=false 时使用）
        if (needGrad_) {
            invRmsGm.SetGlobalBuffer((__gm__ float *)invRms);
            hcBeforeNormGm.SetGlobalBuffer((__gm__ float *)hcBeforeNorm);
        } else {
            int64_t xCastWsTotalSize = tilingData->stage1XCastWsSize / sizeof(float);
            invRmsWsGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize);
            int64_t invRmsWsTotalSize = tilingData->bs * sizeof(float) / sizeof(float);
            hcBeforeNormWsGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize + invRmsWsTotalSize);
        }

        // 初始化 Cube Compute
        if ASCEND_IS_AIC {
            // 根据 needGrad 决定输出地址
            if (needGrad_) {
                cubeCompute_.Init(xCastWsGm, phiGm, hcBeforeNormGm,
                                 tilingData->stage1BsFactor, tilingData->hcMult, tilingData->d, tilingData->stage1VecCoreNum);
            } else {
                cubeCompute_.Init(xCastWsGm, phiGm, hcBeforeNormWsGm,
                                 tilingData->stage1BsFactor, tilingData->hcMult, tilingData->d, tilingData->stage1VecCoreNum);
            }
            // receives d via 'c' param, not total bs; must match host tiling + mhc_pre_cmhc.cpp matmul Init).
            cubeCompute_.usePreNz_ = (tilingData->d >= PRE_NZ_D_THRESHOLD &&
                tilingData->bs >= (tilingData->d >= 7168 ? PRE_NZ_BS_THRESHOLD_D7168 : PRE_NZ_BS_THRESHOLD_D3584));
            // D=3584 bypass tested in Run#58 → regressed, partial L2 caching beneficial at that scale.
            if (cubeCompute_.usePreNz_ && tilingData->d >= 7168) {
                xCastWsGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
            return;
        }
    }

    __aicore__ inline void Process()
    {
        uint64_t blockIdx = GetBlockIdx();
        uint64_t vecCoreId = blockIdx;
        uint64_t cubeCoreId = blockIdx;

        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
        }

        // 循环处理 BS（每轮最多 32 个）
        // ASCEND_IS_AIC 直接根据总BS数，计算当前处理的BS数
        int32_t totalTasksAligned_ = AlignUp(tilingData->bs, tilingData->stage1VecCoreNum * tilingData->stage1BsFactor);

        // Cube Core 流程
        if ASCEND_IS_AIC {
            for (int32_t taskOffset = blockIdx * 2 * tilingData->stage1BsFactor; taskOffset < totalTasksAligned_;
                 taskOffset += tilingData->stage1CubeCoreNum * 2 * tilingData->stage1BsFactor) {
                int32_t tileTaskCount = min(static_cast<int32_t>(2 * tilingData->stage1BsFactor),
                                            static_cast<int32_t>(tilingData->bs - taskOffset));
                // 等待 Vector Core 完成 Cast
                CrossCoreWaitFlag(SYNC_AIV_TO_AIC_FLAG);
                // 执行矩阵乘
                if (tileTaskCount > 0) {
                    cubeCompute_.ProcessMatmulXPhi(taskOffset, tileTaskCount);
                }
                // 通知 Vector Core 完成
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
            }
        }

        if ASCEND_IS_AIV {
            int64_t ncSize = tilingData->hcMult * tilingData->d;
            for (int32_t taskOffset = blockIdx * tilingData->stage1BsFactor; taskOffset < totalTasksAligned_;
                 taskOffset += tilingData->stage1VecCoreNum * tilingData->stage1BsFactor) {
                CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
                int32_t curBs = min(static_cast<int32_t>(tilingData->stage1BsFactor),
                                            static_cast<int32_t>(tilingData->bs - taskOffset));
                if (curBs > 0) {
                    int64_t curBsOffset = taskOffset;
                    // 初始化 sum 为 0
                    LocalTensor<float> invRmsLocal = invRmsQue.AllocTensor<float>();
                    LocalTensor<float> squareLocal = tmpQue.Get<float>();
                    LocalTensor<float> sumLocal = sumQue.Get<float>();
                    Duplicate(sumLocal, 0.0f, curBs);

                    // 分段 ReduceSum
                    for (int64_t ncIdx = 0; ncIdx < tilingData->stage1NcLoop; ncIdx++) {
                        int64_t curNcSize = (ncIdx == tilingData->stage1NcLoop - 1)
                                            ? tilingData->stage1TailNcFactor
                                            : tilingData->stage1NcFactor;
                        int64_t ncOffset = ncIdx * tilingData->stage1NcFactor;
                        int64_t curNcSizeAlign = AlignUp(static_cast<uint64_t>(curNcSize), ELEMENTS_SIZE_PER_BLOCK);

                        // 1. CopyIn X (BF16)
                        LocalTensor<T> xLocal = xQue.AllocTensor<T>();
                        CopyIn(xGm[curBsOffset * ncSize + ncOffset], xLocal, curBs, curNcSize, ncSize - curNcSize);
                        xQue.EnQue(xLocal);

                        // 2. Cast X (BF16) → X_cast (FP32)
                        xLocal = xQue.DeQue<T>();
                        LocalTensor<float> xCastLocal = xCastQue.AllocTensor<float>();
                        Cast(xCastLocal, xLocal, RoundMode::CAST_NONE, curBs * curNcSizeAlign);
                        xQue.FreeTensor(xLocal);
                        xCastQue.EnQue(xCastLocal);

                        // 3. CopyOut X_cast to workspace — conditional Pre-NZ (D≥3584 && bs≥threshold → NZ, else ND)
                        // so AIC matmul reads via CopyNZ2NZ (simple contiguous copy, ~1018 GB/s) instead of
                        // CopyND2NZWithVecOp (slow vector ND2NZ conversion, ~386 GB/s).
                        // GM NZ layout: [ceil(ncSize/8), 2*bsFactor, 8] (K0=8 for FP32, M=2*bsFactor padded to 16)
                        xCastLocal = xCastQue.DeQue<float>();
                        if (tilingData->d >= PRE_NZ_D_THRESHOLD &&
                            tilingData->bs >= (tilingData->d >= 7168 ? PRE_NZ_BS_THRESHOLD_D7168 : PRE_NZ_BS_THRESHOLD_D3584)) {
                            // NZ path: ND→NZ conversion in UB + strided CopyOut to GM (NZ layout)
                            LocalTensor<float> nzLocal = nzBuf.Get<float>();
                            int64_t kBlocks = curNcSizeAlign / 8;  // curNcSizeAlign already 8-aligned (FP32)
                            // ND→NZ: per K0-block (8 K-elements), copy curBs rows from ND (strided) to NZ (contiguous)
                            DataCopyParams nzParams;
                            nzParams.blockCount = static_cast<uint16_t>(curBs);
                            nzParams.blockLen = 1;  // 1 block = 32B = 8 FP32 elements
                            nzParams.srcStride = static_cast<uint16_t>(curNcSizeAlign / 8 - 1);  // gap between ND rows
                            nzParams.dstStride = 0;  // contiguous in NZ buffer
                            for (int64_t kb = 0; kb < kBlocks; kb++) {
                                DataCopy(nzLocal[kb * curBs * 8], xCastLocal[kb * 8], nzParams);
                            }
                            PipeBarrier<PIPE_V>();
                            // V→MTE3 sync: nzLocal written by V pipe, CopyOut reads via MTE3
                            VToMTE3Sync();
                            // NZ CopyOut to GM: each K0-block = curBs*8 elements, GM stride = (2*bsFactor)*8
                            int64_t cubeCoreId = blockIdx / CV_RATIO;
                            int64_t rowOffset = (blockIdx % CV_RATIO) * tilingData->stage1BsFactor;
                            int64_t nzKAligned = (ncSize + 7) / 8 * 8;
                            int64_t nzGmBase = cubeCoreId * (2 * tilingData->stage1BsFactor) * nzKAligned
                                               + ping4vec * tilingData->stage1VecCoreNum * tilingData->stage1BsFactor * nzKAligned;
                            int64_t nzGmOffset = nzGmBase
                                               + (ncOffset / 8) * (2 * tilingData->stage1BsFactor) * 8
                                               + rowOffset * 8;
                            CopyOut(nzLocal, xCastWsGm[nzGmOffset],
                                    static_cast<uint16_t>(kBlocks), curBs * 8,
                                    (2 * tilingData->stage1BsFactor - curBs) * 8);
                            MTE3ToVSync();
                        } else {
                            // ND path (baseline): contiguous CopyOut, AIC reads via CopyND2NZWithVecOp
                            int64_t xCastoffset = blockIdx * tilingData->stage1BsFactor * ncSize
                                                + ping4vec * tilingData->stage1VecCoreNum * tilingData->stage1BsFactor * ncSize;
                            CopyOut(xCastLocal, xCastWsGm[xCastoffset + ncOffset], curBs, curNcSize, ncSize - curNcSize);
                        }
                        xCastQue.FreeTensor(xCastLocal);

                        // 4. 计算 inv_rms
                        // 4.1 Square
                        PipeBarrier<PIPE_V>();
                        Mul(squareLocal, xCastLocal, xCastLocal, curBs * curNcSizeAlign);
                        PipeBarrier<PIPE_V>();

                        // 对当前段进行 ReduceSum
                        uint32_t srcShape[2] = {static_cast<uint32_t>(curBs), static_cast<uint32_t>(curNcSize)};
                        AscendC::ReduceSum<float, Pattern::Reduce::AR, true>(squareLocal, squareLocal, srcShape, true);
                        PipeBarrier<PIPE_V>();
                        
                        // 累加到总和
                        Add(sumLocal, sumLocal, squareLocal, curBs);
                        PipeBarrier<PIPE_V>();
                    }
                    // 4.3 Compute inv_rms
                    // inv_rms = 1 / sqrt(sum / (n*c) + eps)
                    float invNc = 1.0f / static_cast<float>(ncSize);
                    Muls(invRmsLocal, sumLocal, invNc, curBs);
                    PipeBarrier<PIPE_V>();
                    Adds(invRmsLocal, invRmsLocal, tilingData->normEps, curBs);
                    PipeBarrier<PIPE_V>();
                    Sqrt(invRmsLocal, invRmsLocal, curBs);
                    PipeBarrier<PIPE_V>();
                    Duplicate(sumLocal, 1.0f, curBs);
                    PipeBarrier<PIPE_V>();
                    Div(invRmsLocal, sumLocal, invRmsLocal, curBs);
                    invRmsQue.EnQue(invRmsLocal);

                    // 5. CopyOut inv_rms to output
                    invRmsLocal = invRmsQue.DeQue<float>();
                    if (needGrad_) {
                        CopyOut(invRmsLocal, invRmsGm[curBsOffset], 1, curBs);
                    } else {
                        CopyOut(invRmsLocal, invRmsWsGm[curBsOffset], 1, curBs);
                    }
                    invRmsQue.FreeTensor(invRmsLocal);
                    ping4vec = 1 - ping4vec;
                }
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_TO_AIC_FLAG);
            }
            CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
            CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
        }
    }

public:
    TPipe *pipe;
    const MhcPreCmhcMembaseTilingData *tilingData;

    // GM 地址
    GlobalTensor<T> xGm;                    // 输入 X (BF16)
    GlobalTensor<float> phiGm;              // phi 矩阵 (FP32)
    GlobalTensor<float> invRmsGm;           // inv_rms 输出 (FP32)
    GlobalTensor<float> hcBeforeNormGm;     // hcBeforeNorm 输出 (FP32)
    GlobalTensor<float> xCastWsGm;          // X_cast workspace
    GlobalTensor<float> invRmsWsGm;         // invRms workspace (needGrad=false)
    GlobalTensor<float> hcBeforeNormWsGm;   // hcBeforeNorm workspace (needGrad=false)

    // Queue
    TQue<QuePosition::VECIN, NUM_TWO> xQue;         // X 输入队列
    TQue<QuePosition::VECOUT, NUM_TWO> xCastQue;    // X_cast 输出队列
    TBuf<QuePosition::VECCALC> tmpQue;
    TBuf<QuePosition::VECCALC> nzBuf;        // For cmhc add: ND→NZ conversion scratch buffer (Pre-NZ optimization)
    TBuf<QuePosition::VECCALC> sumAddQue;      // sum 队列
    TBuf<QuePosition::VECCALC> sumQue;         // sum 队列
    TQue<QuePosition::VECOUT, NUM_TWO> invRmsQue;   // inv_rms 队列

    // Cube Compute
    HcCubeCompute<float, isFac> cubeCompute_;
    
    // 标志
    bool needGrad_ = true;
    uint64_t ping4vec = 1;

    // 同步标志
    static constexpr uint64_t SYNC_AIV_TO_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_TO_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_MODE2 = NUM_TWO;
    static constexpr uint64_t CV_RATIO = 2;
};

template <typename T, bool isFac>
class MhcPreCmhcStage2 {
public:
    __aicore__ inline MhcPreCmhcStage2()
    {
    }

    // For cmhc add: Init 签名加 coeff（前向输出）
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR y,
                                GM_ADDR post, GM_ADDR hRes,
                                GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms, GM_ADDR coeff,
                                GM_ADDR workspace,
                                const MhcPreCmhcMembaseTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        needGrad_ = tilingData->needGrad;

        xGm.SetGlobalBuffer((__gm__ T *)x);
        hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        postGm.SetGlobalBuffer((__gm__ float *)post);
        hResGm.SetGlobalBuffer((__gm__ float *)hRes);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace);
        // For cmhc add: coeffGm 前向输出（供反向）；coeffWsGm workspace（Stage3 大 GEMM 读，偏移 tiling->coeffWsOffset）
        coeffGm.SetGlobalBuffer((__gm__ float *)coeff);
        coeffWsGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->coeffWsOffset);
        // xGm/yGm for D≥3584 bs≥8192: Stage2 reads xGm (940MB for D=3584 bs=8192) once, writes yGm once — no L2 reuse.
        //   Bypass keeps L2 free for hcBeforeNormGm (2MB, fits L2, Stage2 reuses) + phi (Stage1, 3.5MB) + perm_mats.
        //   bs≥8192 threshold: bs=4096 (case 22-25) REGRESSED +4-10% with bypass (partial L2 caching beneficial at small bs);
        //   bs≥8192 (case 26-31) IMPROVED -4-7% (data too large for L2, bypass wins). Break-even ~bs=8192.
        // hcBeforeNormGm NOT bypassed: 2MB for D=7168 bs=16384, fits L2 (24MB), Stage2 read benefits from L2 caching.
        if (tilingData->d >= 7168 &&
            tilingData->bs >= PRE_NZ_BS_THRESHOLD_D7168) {
            xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            yGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            coeffWsGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            coeffGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if (tilingData->d >= 3584 && tilingData->bs >= 8192) {
            xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            yGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        //   without losing xGm's partial L2 caching benefit that small-bs cases rely on)
        if (tilingData->d >= 3584 && tilingData->bs >= 4096 && tilingData->bs < 8192) {
            yGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }

        if (needGrad_) {
            hPreGm.SetGlobalBuffer((__gm__ float *)hPre);
            hcBeforeNormGm.SetGlobalBuffer((__gm__ float *)hcBeforeNorm);
            invRmsGm.SetGlobalBuffer((__gm__ float *)invRms);
        } else {
            int64_t xCastWsTotalSize = tilingData->stage1XCastWsSize / sizeof(float);
            invRmsGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize);
            int64_t invRmsWsTotalSize = tilingData->bs * sizeof(float) / sizeof(float);
            hcBeforeNormGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize + invRmsWsTotalSize);
        }
        if constexpr (isFac) {
            resSize = Factorial(tilingData->hcMult);
        } else {
            resSize = tilingData->hcMult * tilingData->hcMultAlign;
        }
        // InQue
        int64_t mixesQue01Size =  tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float);
        pipe->InitBuffer(mixesQue01, NUM_TWO, mixesQue01Size);
        pipe->InitBuffer(mixesQue2, NUM_TWO, tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(squareSumQue, NUM_TWO, tilingData->stage2RowFactor * SQUARE_SUM_SIZE * sizeof(float));
        int64_t xQueNum2 = tilingData->stage2RowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor);
        pipe->InitBuffer(xQue, NUM_TWO, xQueNum2 * sizeof(T));
        // OutQue
        pipe->InitBuffer(yQue, NUM_TWO, tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, NUM_TWO, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(combFragQue, NUM_TWO,
                         tilingData->stage2RowFactor * resSize * sizeof(float));
        pipe->InitBuffer(reduceQue, NUM_TWO, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        // TBuf
        pipe->InitBuffer(combFragBuf,
                         tilingData->stage2RowFactor * resSize * NUM_TWO * sizeof(float));
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2, resSize * sizeof(float));
        pipe->InitBuffer(rowBrcbBuf0, RoundUp<float>(tilingData->stage2RowFactor) * BLOCK_SIZE);
        pipe->InitBuffer(hcBrcbBuf1, RoundUp<float>(tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO) * BLOCK_SIZE);
        pipe->InitBuffer(reduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(mxies01ReduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(mxies02ReduceBuf, tilingData->stage2RowFactor * resSize * sizeof(float));
        pipe->InitBuffer(xCastBuf, xQueNum2 * sizeof(float));
        pipe->InitBuffer(yCastBuf, tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(float));

        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        rowBrcbLocal0 = rowBrcbBuf0.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        reduceLocal = reduceBuf.Get<float>();
        mxies01ReduceLocal = mxies01ReduceBuf.Get<float>();
        mxies02ReduceLocal = mxies02ReduceBuf.Get<float>();
        xCastLocal = xCastBuf.Get<float>();
        yCastLocal = yCastBuf.Get<float>();
        combFragBufLocal = combFragBuf.Get<float>();
    }

    __aicore__ inline void Process(bool isTailBsLoop = false)
    {
        SyncAll();
        isTailBsLoop_ = isTailBsLoop;
        
        int64_t curRowOfFormerBlock = isTailBsLoop ? tilingData->tailBsRowOfFormerBlock : tilingData->rowOfFormerBlock;
        int64_t curRowLoopOfFormerBlock = isTailBsLoop ? tilingData->tailBsRowLoopOfFormerBlock : tilingData->rowLoopOfFormerBlock;
        int64_t curRowLoopOfTailBlock = isTailBsLoop ? tilingData->tailBsRowLoopOfTailBlock : tilingData->rowLoopOfTailBlock;
        int64_t curSecondUsedCoreNum = isTailBsLoop ? tilingData->tailBsUsedCoreNum : tilingData->secondUsedCoreNum;
        int64_t curStage2RowFactor = isTailBsLoop ? tilingData->tailBsRowFactor : tilingData->stage2RowFactor;
        int64_t curTailRowFactorOfFormerBlock = isTailBsLoop ? tilingData->tailBsTailRowFactorOfFormerBlock : tilingData->tailRowFactorOfFormerBlock;
        int64_t curTailRowFactorOfTailBlock = isTailBsLoop ? tilingData->tailBsTailRowFactorOfTailBlock : tilingData->tailRowFactorOfTailBlock;
        int64_t curML1Size = isTailBsLoop ? tilingData->tailBsML1Size : tilingData->mL1Size;
        
        if ASCEND_IS_AIV {
            int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
            int64_t stage2BlockIdx = GetBlockIdx();
            int64_t stage2UsedCoreNum = curSecondUsedCoreNum;
            if (stage2BlockIdx >= stage2UsedCoreNum) {
                return;
            }
            int64_t mmLastAxisSize = CeilAlign(tilingData->hcMix, MM_CACHE_LINE_BYTES / sizeof(float));
            int64_t xCastFp32BufSize = curML1Size * CeilAlign(tilingData->cvLoopKSize, MM_CACHE_LINE_BYTES / sizeof(float)) * sizeof(float);
            int64_t workspaceSize1 = (tilingData->cubeCoreNum * DOUBLE_BUFFER * xCastFp32BufSize) / sizeof(float);
            int64_t workspaceSize2 = CeilAlign(stage1UsedCoreNum * tilingData->bs * mmLastAxisSize * sizeof(float), WORKSPACE_ALIGN_SIZE) / sizeof(float);
            CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
            if constexpr (isFac) {
                CopyIn(hcBaseGm[tilingData->hcMult * NUM_TWO], hcBase2Local, 1, resSize);
            } else {
                CopyIn(hcBaseGm[tilingData->hcMult * NUM_TWO], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
            }
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            int64_t rowOuterLoop =
                (stage2BlockIdx == stage2UsedCoreNum - 1) ? curRowLoopOfTailBlock : curRowLoopOfFormerBlock;
            int64_t tailRowFactor = (stage2BlockIdx == stage2UsedCoreNum - 1) ? curTailRowFactorOfTailBlock :
                                                                        curTailRowFactorOfFormerBlock;
            int64_t xGmBlockBaseOffsetPart2 = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult * tilingData->d;
            uint64_t mixBaseOffset = 0;
            for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
                int64_t xGmBsBaseOffsetPart2 = rowOuterIdx * curStage2RowFactor * tilingData->hcMult * tilingData->d;
                int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : curStage2RowFactor;
                                
                squareSumOutLocal = squareSumQue.AllocTensor<float>();
                Duplicate(rowBrcbLocal0, static_cast<float>(1.0f), curRowFactor);

                CopyIn(invRmsGm[stage2BlockIdx * curRowOfFormerBlock + rowOuterIdx * curStage2RowFactor],
                        squareSumOutLocal, 1, curRowFactor);
                
                squareSumQue.EnQue(squareSumOutLocal);
                squareSumOutLocal = squareSumQue.DeQue<float>();
                // 搬运矩阵乘的前两段结果--> 内存格式的变更： 连续地址改为 bs *n ; bs *n ,n为4 非对齐，则需要pad出对齐的8
                mxies01Local = mixesQue01.AllocTensor<float>();

                mixBaseOffset = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMix +
                                            rowOuterIdx * curStage2RowFactor * tilingData->hcMix;
                CopyInWithOuterFor(hcBeforeNormGm[mixBaseOffset], mxies01Local, 1, curRowFactor, tilingData->hcMult,
                                tilingData->bs, tilingData->hcMix);
                CopyInWithOuterFor(hcBeforeNormGm[mixBaseOffset + tilingData->hcMult],
                                mxies01Local[curRowFactor * tilingData->hcMultAlign],
                                1, curRowFactor, tilingData->hcMult, tilingData->bs,
                                tilingData->hcMix);
                // // wk:[2, kcorenum, bs, n^2 +2n]
                // // mx0[2,kcorenum, curRowfator, n]
                mixesQue01.EnQue(mxies01Local);
                mxies01Local = mixesQue01.DeQue<float>();

                ProcessPre(mxies01ReduceLocal, mxies01Local, hcBase0Local, squareSumOutLocal, rowBrcbLocal0, hcBrcbLocal1,
                            hcScaleGm.GetValue(0), tilingData->hcEps, curRowFactor, tilingData->hcMult);

                if (needGrad_) {
                    int64_t hPreBaseOffset = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult +
                                             rowOuterIdx * curStage2RowFactor * tilingData->hcMult;
                    VToMTE3Sync();
                    CopyOut(mxies01ReduceLocal, hPreGm[hPreBaseOffset], curRowFactor, tilingData->hcMult);
                    MTE3ToVSync();
                }
                // --- pre --
                for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                    int64_t curDFactor =
                        (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                    xLocal = xQue.template AllocTensor<T>();
                    CopyIn(xGm[xGmBlockBaseOffsetPart2 + xGmBsBaseOffsetPart2 +
                                dLoopIdx * tilingData->dFactor],
                            xLocal, curRowFactor * tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
                    xQue.template EnQue(xLocal);
                    xLocal = xQue.template DeQue<T>();
                    yLocal = yQue.template AllocTensor<T>();

                    ProcessY(yLocal, xLocal, mxies01ReduceLocal, hcBrcbLocal1, xCastLocal, yCastLocal, curRowFactor,
                                tilingData->hcMult, curDFactor);
                    xQue.template FreeTensor(xLocal);
                    yQue.template EnQue(yLocal);
                    yLocal = yQue.template DeQue<T>();
                    
                    CopyOut(yLocal,
                            yGm[stage2BlockIdx * curRowOfFormerBlock * tilingData->d +
                                rowOuterIdx * curStage2RowFactor * tilingData->d + dLoopIdx * tilingData->dFactor],
                            curRowFactor, curDFactor, tilingData->d - curDFactor);
                    yQue.template FreeTensor(yLocal);
                }
                // post
                postLocal = postQue.AllocTensor<float>();
                ProcessPost(postLocal, mxies01Local[curRowFactor * tilingData->hcMultAlign], hcBase1Local,
                            squareSumOutLocal, rowBrcbLocal0, hcBrcbLocal1, hcScaleGm.GetValue(1), curRowFactor,
                            tilingData->hcMult);
                mixesQue01.FreeTensor(mxies01Local); // 这里对应的申请在上面
                postQue.EnQue(postLocal);
                postLocal = postQue.DeQue<float>();

                CopyOut(postLocal,
                        postGm[stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult +
                                rowOuterIdx * curStage2RowFactor * tilingData->hcMult],
                        curRowFactor, tilingData->hcMult);
                postQue.FreeTensor(postLocal);

                // hRes
                mixes2Local = mixesQue2.AllocTensor<float>();
                mixBaseOffset = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMix +
                                rowOuterIdx * curStage2RowFactor * tilingData->hcMix;
                for (int64_t j = 0; j < curRowFactor; ++j) {
                    if constexpr (isFac) {
                        CopyIn(hcBeforeNormGm[mixBaseOffset + j * tilingData->hcMix + tilingData->hcMult * NUM_TWO],
                           mixes2Local[j * resSize], 1,
                           resSize);
                    } else {
                        CopyIn(hcBeforeNormGm[mixBaseOffset + j * tilingData->hcMix + tilingData->hcMult * NUM_TWO],
                           mixes2Local[j * tilingData->hcMult * tilingData->hcMultAlign], tilingData->hcMult,
                           tilingData->hcMult);
                    }
                }
                mixesQue2.EnQue(mixes2Local);
                mixes2Local = mixesQue2.DeQue<float>();

                // For Cmhc Add: 修正 mixes2 *= invRms 的 per-row 广播（原实现精度 bug 修复）
                // 原实现 MulABLastDimBrcInline<false>+未初始化 rsqrtLocal 在 resSize=24(numRemainPerLine>block) 路径
                //   src1RepStride=0 致所有行复用同一 rsqrt block（行间错位），softmax 放大后 coeff/hRes 精度超阈。
                // 修复：Brcb invRms[curRow] → rowBrcbLocal0[curRow,8]（每行8个相同 invRms[i]，与 ProcessPre 同模式），
                //   再按行 Mul：每行 resSize=24=3*8，3 个 8-block 各乘 rowBrcbLocal0[rowIdx*8]（src1RepStride=0 行内广播）。
                //   curRowFactor 通常 ≤64，coeff 计算量远小于 Stage1/3 matmul，性能可忽略。
                Brcb(rowBrcbLocal0, squareSumOutLocal, CeilDiv(curRowFactor, ONE_REPEAT_BLOCK_NUMS),
                      {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
                PipeBarrier<PIPE_V>();
                for (int32_t rowIdx = 0; rowIdx < curRowFactor; rowIdx++) {
                    Mul(mxies02ReduceLocal[rowIdx * resSize], mixes2Local[rowIdx * resSize],
                        rowBrcbLocal0[rowIdx * ONE_REPEAT_BLOCK_NUMS], ONE_REPEAT_BLOCK_NUMS,
                        static_cast<int32_t>(resSize / ONE_REPEAT_BLOCK_NUMS), {1, 1, 0, 1, 1, 0});
                }
                PipeBarrier<PIPE_V>();
                mixesQue2.FreeTensor(mixes2Local);
                Muls(mxies02ReduceLocal, mxies02ReduceLocal, hcScaleGm.GetValue(NUM_TWO),
                        curRowFactor * resSize);
                PipeBarrier<PIPE_V>();
                AddBAFirstDimBrcInline<float>(mxies02ReduceLocal, mxies02ReduceLocal, hcBase2Local, curRowFactor,
                                                resSize);
                // For cmhc add: softmax（eps=0，对齐 mint.softmax）→ coeff
                SoftmaxFP32Perf(mxies02ReduceLocal, mxies02ReduceLocal, reduceLocal, hcBrcbLocal1,
                                curRowFactor, resSize, 0.0f);
                PipeBarrier<PIPE_V>();
                squareSumQue.template FreeTensor(squareSumOutLocal);

                // For cmhc add: CopyOut coeff → coeffWsGm（workspace，Stage3 必读，永远写）+ coeffGm（前向输出，门控 needGrad_ 防 null 写）
                VToMTE3Sync();
                int64_t curRowCoeffOffset = stage2BlockIdx * curRowOfFormerBlock * resSize +
                                            rowOuterIdx * curStage2RowFactor * resSize;
                CopyOut(mxies02ReduceLocal, coeffWsGm[curRowCoeffOffset], curRowFactor, resSize);
                if (needGrad_) {
                    CopyOut(mxies02ReduceLocal, coeffGm[curRowCoeffOffset], curRowFactor, resSize);
                }
                MTE3ToVSync();
            }
        }
    }

private:
    TPipe *pipe;
    const MhcPreCmhcMembaseTilingData *tilingData;
    GlobalTensor<float> mixesGm;
    GlobalTensor<float> rsqrtGm;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> hResGm;
    // For cmhc add: coeff workspace（Stage3 大 GEMM 读）+ coeff 前向输出（供反向）
    GlobalTensor<float> coeffWsGm;
    GlobalTensor<float> coeffGm;

    GlobalTensor<float> hPreGm;
    GlobalTensor<float> hcBeforeNormGm;
    GlobalTensor<float> invRmsGm;

    bool needGrad_ = false;
    bool isTailBsLoop_ = false;

    TQue<QuePosition::VECIN, 1> mixesQue01;
    TQue<QuePosition::VECIN, 1> mixesQue2;
    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;

    TQue<QuePosition::VECOUT, 1> reduceQue;
    TQue<QuePosition::VECIN, 1> squareSumQue;

    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    TBuf<QuePosition::VECCALC> rowBrcbBuf0;
    TBuf<QuePosition::VECCALC> hcBrcbBuf1;
    TBuf<QuePosition::VECCALC> reduceBuf;

    TBuf<QuePosition::VECCALC> rsqrtBuf;
    TBuf<QuePosition::VECCALC> squareReduceBuf;
    TBuf<QuePosition::VECCALC> mxies01ReduceBuf;
    TBuf<QuePosition::VECCALC> mxies02ReduceBuf;

    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;
    TBuf<QuePosition::VECCALC> maskPatternBuf;
    TBuf<QuePosition::VECCALC> combFragBuf;

    LocalTensor<float> mxies01Local;
    LocalTensor<float> mixes2Local;
    LocalTensor<float> rsqrtLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
    LocalTensor<float> rowBrcbLocal0;
    LocalTensor<float> hcBrcbLocal1;
    LocalTensor<float> reduceLocal;
    LocalTensor<float> squareReduceLocal;
    LocalTensor<float> mxies01ReduceLocal;
    LocalTensor<float> mxies02ReduceLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> yCastLocal;
    LocalTensor<float> squareSumOutLocal;
    LocalTensor<uint32_t> maskPatternLocal;

    LocalTensor<float> reduceLocal2;
    LocalTensor<float> combFragBufLocal;

    int64_t resSize = 0;
};

// For cmhc add: MhcPreCmhcStage3 —— 独立 cube pass，大 GEMM coeff[bs,n!] @ perm_mats[n!,n²] → hRes[bs,n²]
// 仿 MhcPreCmhcStage1 结构；AIC 跑大 GEMM，AIV 不做事（但必须参与 SyncAll，见方案 §2.7 v1.4 全核参与）
template <typename T>
class MhcPreCmhcStage3 {
public:
    __aicore__ inline MhcPreCmhcStage3() {}

    // For cmhc add: Init 设置 GM 地址；AIC 分支初始化 cubeCompute_，AIV 不做事
    __aicore__ inline void Init(GM_ADDR workspace, GM_ADDR permMats, GM_ADDR hRes,
                                const MhcPreCmhcMembaseTilingData *tilingDataPtr, TPipe *pipePtr) {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        // coeffWsGm 从 userWs 偏移（Stage2 已 CopyOut，偏移 tiling->coeffWsOffset）
        coeffWsGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->coeffWsOffset);
        permMatsGm.SetGlobalBuffer((__gm__ float *)permMats);
        hResGm.SetGlobalBuffer((__gm__ float *)hRes);
        if (tilingData->d >= 7168 && tilingData->bs >= PRE_NZ_BS_THRESHOLD_D7168) {
            coeffWsGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            hResGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if ASCEND_IS_AIC {
            cubeCompute_.Init(coeffWsGm, permMatsGm, hResGm, tilingData->nPerm, tilingData->nn);
        }
    }

    // For cmhc add: Process —— AIC 跑大 GEMM，AIV return。
    // 关键：必须用 SyncAll<false>()（isAIVOnly=false）做真正的 AIC<->AIV 全核屏障。
    // 默认 SyncAll()（isAIVOnly=true）在 910b/910_93(arch2201) 上只同步 AIV 核，AIC 不与 AIV 建立 happens-before：
    // Stage2 的 AIV 用 MTE3 写 coeff 到 coeffWsGm，AIC 若不等 AIV 的 MTE3 完成就读 coeffWsGm 会读到垃圾值
    //（coeffGm 输出正确，因 host 直读 HBM）。SyncAll<false>() 的 AIV 侧用 PIPE_MTE3 通知、AIC 侧 wait，建立 MTE3->cube 可见性。
    // 与 Stage1->Stage2 不同：那条链路靠 Stage1 的 CrossCoreSetFlag/CrossCoreWaitFlag 保证，故其默认 SyncAll() 无害。
    __aicore__ inline void Process() {
        SyncAll<false>();  // 关键：AIC<->AIV 全核屏障（含 Stage2 早退出的 AIV 核），否则 AIC race AIV 的 coeff 写
        if ASCEND_IS_AIC {
            // 大 GEMM: coeff[bs, n!] @ perm_mats[n!, n²] → hRes[bs, n²]
            int64_t mPerCore = (tilingData->bs + tilingData->stage3CubeCoreNum - 1) /
                               tilingData->stage3CubeCoreNum;  // CeilDiv
            int32_t taskOffset = static_cast<int32_t>(GetBlockIdx() * mPerCore);
            if (taskOffset >= tilingData->bs) {
                return;  // 边界1: bs < cubeCoreNum 时，空闲核直接返回
            }
            // v1.6 修正: min(mPerCore, bs-taskOffset)，避免 CeilDiv 尾核越界（旧三元公式在高位核空闲时崩）
            int32_t remain = static_cast<int32_t>(tilingData->bs - taskOffset);
            int32_t curM = (remain < mPerCore) ? remain : static_cast<int32_t>(mPerCore);
            cubeCompute_.ProcessMatmulCoeffPerm(taskOffset, curM, tilingData->nPerm, tilingData->nn);
        }
        if ASCEND_IS_AIV {
            return;  // AIV 在 Stage3 不做事
        }
    }

public:
    TPipe *pipe;
    const MhcPreCmhcMembaseTilingData *tilingData;
    GlobalTensor<float> coeffWsGm;
    GlobalTensor<float> permMatsGm;
    GlobalTensor<float> hResGm;
    HcResProjCubeCompute<float> cubeCompute_;  // For cmhc add: Stage3 cube mm 封装
};

} // namespace MhcPreCmhc

#endif
