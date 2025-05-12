#ifndef DHINFERENCE_ACTIVATION_H
#define DHINFERENCE_ACTIVATION_H

#include <vector>
#include "../utils/tensor.h"

namespace dhinference {

// 激活函数类型枚举
enum class ActivationType {
    RELU,
    GELU,
    SWISH,
    SILU
};

// 激活函数命名空间
namespace activation {

// 应用激活函数到输入张量
Tensor activate(const Tensor& input, ActivationType type);

// 各种激活函数的实现
Tensor relu(const Tensor& input);
Tensor gelu(const Tensor& input);
Tensor swish(const Tensor& input);
Tensor silu(const Tensor& input);

} // namespace activation

} // namespace dhinference

#endif // DHINFERENCE_ACTIVATION_H 