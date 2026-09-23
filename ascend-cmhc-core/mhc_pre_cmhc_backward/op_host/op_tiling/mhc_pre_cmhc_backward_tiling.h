#ifndef MHC_PRE_CMHC_BACKWARD_TILING_H_
#define MHC_PRE_CMHC_BACKWARD_TILING_H_

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {
struct MhcPreCmhcBackwardCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0CSize{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
};
} // namespace optiling

#endif // MHC_PRE_CMHC_BACKWARD_TILING_H_
