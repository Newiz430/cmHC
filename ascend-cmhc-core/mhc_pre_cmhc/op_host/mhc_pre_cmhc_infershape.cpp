#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace gert;
using namespace ge;

namespace ops {

const constexpr int64_t X_INDEX = 0;
const constexpr int64_t PHI_INDEX = 1;
const constexpr int64_t ALPHA_INDEX = 2;

const constexpr int64_t OUT_H_IN_INDEX = 0;
const constexpr int64_t OUT_H_POST_INDEX = 1;
const constexpr int64_t OUT_H_RES_INDEX = 2;
const constexpr int64_t OUT_INV_RMS_INDEX = 3;
const constexpr int64_t OUT_MM_RES_INDEX = 4;
const constexpr int64_t OUT_H_PRE_INDEX = 5;
// For cmhc add: coeff 输出 index（softmax 结果 [B,S,n!]，供反向）
const constexpr int64_t OUT_COEFF_INDEX = 6;

const constexpr int64_t BSND_DIM_NUM = 4;
const constexpr int64_t TND_DIM_NUM = 3;
const constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;

static void SetShape3D(gert::Shape *shape, uint64_t d0, uint64_t d1, uint64_t d2)
{
    shape->SetDimNum(3);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
}

static void SetShape2D(gert::Shape *shape, uint64_t d0, uint64_t d1)
{
    shape->SetDimNum(2);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
}

static void SetShape4D(gert::Shape *shape, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3)
{
    shape->SetDimNum(4);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
    shape->SetDim(3, d3);
}

static void SetShape1D(gert::Shape *shape, uint64_t d0)
{
    shape->SetDimNum(1);
    shape->SetDim(0, d0);
}

static void SetShapeFromX(gert::Shape *dst, const gert::Shape *src)
{
    dst->SetDimNum(src->GetDimNum());
    for (int64_t i = 0; i < src->GetDimNum(); ++i) {
        dst->SetDim(i, src->GetDim(i));
    }
}

static uint64_t Factorial(uint64_t n)
{
    uint64_t r = 1;
    for (uint64_t i = 2; i <= n; ++i) {
        r *= i;
    }
    return r;
}

static ge::graphStatus InferShape4MhcPreCmhc(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape MhcPreCmhc");
    const gert::Shape *xShape = context->GetDynamicInputShape(X_INDEX, 0);
    const gert::Shape *phiShape = context->GetDynamicInputShape(PHI_INDEX, 0);
    const gert::Shape *alphaShape = context->GetDynamicInputShape(ALPHA_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, phiShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, alphaShape);

    int64_t phiDim = phiShape->GetDimNum();
    int64_t xDim = xShape->GetDimNum();
    OP_CHECK_IF(phiDim != 2, OP_LOGE(context->GetNodeName(), "phiShapeDim should be 2, but got %ld", phiDim),
                return GRAPH_FAILED);
    OP_CHECK_IF(
        xDim != BSND_DIM_NUM && xDim != TND_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "xShapeDim should be %ld or %ld, but got %ld", BSND_DIM_NUM, TND_DIM_NUM, xDim),
        return GRAPH_FAILED);

    int64_t alphaDimSize = alphaShape->GetDim(0);
    // For cmhc add: alpha 只允许 3（CMHC 必然有 h_res，移除 size==2 路径）
    OP_CHECK_IF(alphaDimSize != 3,
                OP_LOGE(context->GetNodeName(), "Alpha dim[0] must be 3 for CMHC, got %ld", alphaDimSize),
                return GRAPH_FAILED);

    // For cmhc add: outShapes 加 coeff（index 7），数组扩到 7
    gert::Shape *outShapes[7] = {context->GetOutputShape(OUT_H_IN_INDEX),   context->GetOutputShape(OUT_H_POST_INDEX),
                                 context->GetOutputShape(OUT_H_RES_INDEX),  context->GetOutputShape(OUT_INV_RMS_INDEX),
                                 context->GetOutputShape(OUT_MM_RES_INDEX), context->GetOutputShape(OUT_H_PRE_INDEX),
                                 context->GetOutputShape(OUT_COEFF_INDEX)};
    if (xDim == BSND_DIM_NUM) {
        // b,s,n,d
        uint64_t b = xShape->GetDim(0), s = xShape->GetDim(1), n = xShape->GetDim(2), d = xShape->GetDim(3);
        // For cmhc add: matN 恒 n!+2n（CMHC 必然有 res 段）
        uint64_t matN = Factorial(n) + 2 * n;
        uint64_t nPerm = Factorial(n);  // For cmhc add: n!，softmax 维度
        uint64_t nn = n * n;            // For cmhc add: n²，h_res 输出维度
        SetShape3D(outShapes[OUT_H_IN_INDEX], b, s, d);
        SetShape3D(outShapes[OUT_H_POST_INDEX], b, s, n);
        // For cmhc add: h_res 统一 [B,S,n*n]（CANN/ACLNN 边界，非 [B,S,n,n]）
        SetShape3D(outShapes[OUT_H_RES_INDEX], b, s, nn);
        SetShape2D(outShapes[OUT_INV_RMS_INDEX], b, s);
        SetShape3D(outShapes[OUT_MM_RES_INDEX], b, s, matN);
        SetShape3D(outShapes[OUT_H_PRE_INDEX], b, s, n);
        // For cmhc add: coeff = [B,S,n!]（softmax 输出，供反向）
        SetShape3D(outShapes[OUT_COEFF_INDEX], b, s, nPerm);
    } else {
        // t,n,d
        uint64_t t = xShape->GetDim(0), n = xShape->GetDim(1), d = xShape->GetDim(2);
        uint64_t matN = Factorial(n) + 2 * n;
        uint64_t nPerm = Factorial(n);
        uint64_t nn = n * n;
        SetShape2D(outShapes[OUT_H_IN_INDEX], t, d);
        SetShape2D(outShapes[OUT_H_POST_INDEX], t, n);
        SetShape2D(outShapes[OUT_H_RES_INDEX], t, nn);  // For cmhc add: [T,n*n]
        SetShape1D(outShapes[OUT_INV_RMS_INDEX], t);
        SetShape2D(outShapes[OUT_MM_RES_INDEX], t, matN);
        SetShape2D(outShapes[OUT_H_PRE_INDEX], t, n);
        SetShape2D(outShapes[OUT_COEFF_INDEX], t, nPerm);  // For cmhc add: [T,n!]
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape MhcPreCmhc");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4MhcPreCmhc(gert::InferDataTypeContext *context)
{
    const auto xDtype = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(OUT_H_IN_INDEX, xDtype);
    context->SetOutputDataType(OUT_H_POST_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_H_RES_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_INV_RMS_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_MM_RES_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_H_PRE_INDEX, DataType::DT_FLOAT);
    // For cmhc add: coeff = FP32（softmax 输出）
    context->SetOutputDataType(OUT_COEFF_INDEX, DataType::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPreCmhc).InferShape(InferShape4MhcPreCmhc).InferDataType(InferDataType4MhcPreCmhc);
} // namespace ops
