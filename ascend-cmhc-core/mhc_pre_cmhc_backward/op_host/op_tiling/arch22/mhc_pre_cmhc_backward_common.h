#ifndef MHC_PRE_CMHC_BACKWARD_OP_HOST_OP_TILING_ARCH22_COMMON_H
#define MHC_PRE_CMHC_BACKWARD_OP_HOST_OP_TILING_ARCH22_COMMON_H

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

inline std::map<ge::DataType, uint64_t> kDataSizeMap = {
    {ge::DT_FLOAT, sizeof(float)},
    {ge::DT_INT32, sizeof(int32_t)},
    {ge::DT_INT64, sizeof(int64_t)}
};

/**
 * if b is 0, return a
 */
template<typename T>
inline T DivCeil(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

/**
 * if b is 0, return 0
 */
template<typename T>
inline T CeilAlign(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return DivCeil(a, b) * b;
}

/**
 * if b is 0, return a
 */
template<typename T>
inline T DivFloor(T a, T b)
{
    return b == 0 ? a : a / b;
}

/**
 * if b is 0, return 0
 */
template<typename T>
inline T FloorAlign(T a, T b)
{
    return b == 0 ? 0 : a / b * b;
}

/**
 * \brief Compute factorial of n
 * \param n input value
 * \return n! (factorial of n)
 */
inline int64_t Factorial(int64_t n)
{
    if (n <= 1) {
        return 1;
    }
    int64_t result = 1;
    for (int64_t i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

#endif // MHC_PRE_CMHC_BACKWARD_OP_HOST_OP_TILING_ARCH22_COMMON_H
