#include "../../include/ops/activation.h"
#include <cmath>
#include <algorithm>

namespace dhinference {
namespace activation {

Tensor activate(const Tensor& input, ActivationType type) {
    switch (type) {
        case ActivationType::RELU:
            return relu(input);
        case ActivationType::GELU:
            return gelu(input);
        case ActivationType::SWISH:
            return swish(input);
        case ActivationType::SILU:
            return silu(input);
        default:
            // 默认返回原始输入
            return input;
    }
}

Tensor relu(const Tensor& input) {
    Tensor result(input.shape());
    const float* in_data = input.data();
    float* out_data = result.data();
    
    for (size_t i = 0; i < input.size(); ++i) {
        out_data[i] = std::max(0.0f, in_data[i]);
    }
    
    return result;
}

Tensor gelu(const Tensor& input) {
    Tensor result(input.shape());
    const float* in_data = input.data();
    float* out_data = result.data();
    
    // GELU近似实现: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/π)
    
    for (size_t i = 0; i < input.size(); ++i) {
        const float x = in_data[i];
        const float x3 = x * x * x;
        out_data[i] = x * 0.5f * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x3)));
    }
    
    return result;
}

Tensor swish(const Tensor& input) {
    Tensor result(input.shape());
    const float* in_data = input.data();
    float* out_data = result.data();
    
    // Swish: x * sigmoid(x)
    for (size_t i = 0; i < input.size(); ++i) {
        const float x = in_data[i];
        const float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
        out_data[i] = x * sigmoid_x;
    }
    
    return result;
}

Tensor silu(const Tensor& input) {
    // SiLU与Swish相同: x * sigmoid(x)
    return swish(input);
}

} // namespace activation
} // namespace dhinference 