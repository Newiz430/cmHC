#ifndef MHC_PRE_CMHC_BACKWARD_DATA_ARCH22_H
#define MHC_PRE_CMHC_BACKWARD_DATA_ARCH22_H

#include <cstdint>

struct MhcPreCmhcBackwardArch22TilingData {
    int64_t batchSize = 0;
    int64_t seqLength = 0;
    int64_t c = 0;
    int64_t n = 0;
    int64_t c0 = 0;
    int64_t c1 = 0;
    int64_t cTail = 0;
    int64_t aivNum = 0;
    int64_t tileGradY = 0;
    int64_t tileHHat2 = 0;
    int64_t tileSize = 0;
    int64_t ubSize = 0;
    float eps = 0;
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
};

#endif // MHC_PRE_CMHC_BACKWARD_DATA_ARCH22_H
