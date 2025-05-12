#include "../../include/ops/layernorm.h"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>

namespace dhinference {

LayerNorm::LayerNorm(int hidden_dim) : hidden_dim_(hidden_dim) {
    // 初始化参数
    gamma_.resize(hidden_dim, 1.0f); // 缩放参数初始化为1
    beta_.resize(hidden_dim, 0.0f);  // 偏移参数初始化为0
}

void LayerNorm::setParams(const std::vector<float>& gamma, const std::vector<float>& beta) {
    if (gamma.size() != static_cast<size_t>(hidden_dim_) || beta.size() != static_cast<size_t>(hidden_dim_)) {
        throw std::invalid_argument("LayerNorm参数大小不匹配");
    }
    
    gamma_ = gamma;
    beta_ = beta;
}

Tensor LayerNorm::forward(const Tensor& input) {
    const auto& shape = input.shape();
    
    // 确保输入是2D张量，且最后一维是hidden_dim
    if (shape.size() != 2 || shape[1] != static_cast<size_t>(hidden_dim_)) {
        throw std::invalid_argument("LayerNorm输入形状不匹配");
    }
    
    Tensor output(shape);
    const float* in_data = input.data();
    float* out_data = output.data();
    
    const size_t batch_size = shape[0];
    const size_t feat_size = shape[1];
    
    // 对每个样本进行归一化
    for (size_t i = 0; i < batch_size; ++i) {
        const float* row_data = in_data + i * feat_size;
        float* out_row = out_data + i * feat_size;
        
        // 计算均值
        float mean = 0.0f;
        for (size_t j = 0; j < feat_size; ++j) {
            mean += row_data[j];
        }
        mean /= feat_size;
        
        // 计算方差
        float var = 0.0f;
        for (size_t j = 0; j < feat_size; ++j) {
            float diff = row_data[j] - mean;
            var += diff * diff;
        }
        var /= feat_size;
        
        // 避免除以零，添加小的epsilon值
        constexpr float epsilon = 1e-5f;
        float inv_std = 1.0f / std::sqrt(var + epsilon);
        
        // 归一化并应用gamma和beta
        for (size_t j = 0; j < feat_size; ++j) {
            out_row[j] = gamma_[j] * (row_data[j] - mean) * inv_std + beta_[j];
        }
    }
    
    return output;
}

} // namespace dhinference
