#include "../../include/ops/matmul.h"
#include <stdexcept>
#include <algorithm>

namespace dhinference {

// MatMul类实现
MatMul::MatMul() : transpose_a_(false), transpose_b_(false), add_bias_(false) {}

void MatMul::setTransposeOptions(bool transposeA, bool transposeB) {
    transpose_a_ = transposeA;
    transpose_b_ = transposeB;
}

void MatMul::setAddBias(bool addBias, const std::vector<float>& bias) {
    add_bias_ = addBias;
    bias_ = bias;
}

Tensor MatMul::forward(const Tensor& A, const Tensor& B) {
    // 检查输入张量是否为2D
    if (A.shape().size() != 2 || B.shape().size() != 2) {
        throw std::invalid_argument("MatMul要求2D张量输入");
    }
    
    // 获取维度
    size_t m = A.shape()[0];
    size_t k = A.shape()[1];
    size_t n = B.shape()[1];
    
    // 如果需要转置，调整维度
    if (transpose_a_) {
        std::swap(m, k);
    }
    
    size_t k_b = B.shape()[0];
    if (transpose_b_) {
        std::swap(k_b, n);
    }
    
    // 检查内部维度是否匹配
    if (k != k_b) {
        throw std::invalid_argument("矩阵乘法维度不匹配: " + 
                                   std::to_string(k) + " vs " + std::to_string(k_b));
    }
    
    // 创建输出张量
    Tensor C({m, n}, 0.0f);
    
    // 获取数据指针
    const float* a_data = A.data();
    const float* b_data = B.data();
    float* c_data = C.data();
    
    // 矩阵乘法实现
    if (!transpose_a_ && !transpose_b_) {
        // C = A * B, 标准矩阵乘法
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                c_data[i * n + j] = sum;
            }
        }
    } else if (transpose_a_ && !transpose_b_) {
        // C = A^T * B
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += a_data[p * m + i] * b_data[p * n + j];
                }
                c_data[i * n + j] = sum;
            }
        }
    } else if (!transpose_a_ && transpose_b_) {
        // C = A * B^T
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += a_data[i * k + p] * b_data[j * k + p];
                }
                c_data[i * n + j] = sum;
            }
        }
    } else { // transpose_a_ && transpose_b_
        // C = A^T * B^T
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += a_data[p * m + i] * b_data[j * k + p];
                }
                c_data[i * n + j] = sum;
            }
        }
    }
    
    // 如果需要，添加偏置
    if (add_bias_) {
        if (bias_.size() != n) {
            throw std::invalid_argument("偏置大小不匹配");
        }
        
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                c_data[i * n + j] += bias_[j];
            }
        }
    }
    
    return C;
}

// Linear类实现
Linear::Linear(int in_features, int out_features, bool use_bias)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias) {
    
    // 初始化权重
    weights_ = Tensor({static_cast<size_t>(in_features), static_cast<size_t>(out_features)}, 0.0f);
    
    // 初始化偏置（如果需要）
    if (use_bias) {
        bias_.resize(out_features, 0.0f);
    }
    
    // 设置矩阵乘法选项
    matmul_.setTransposeOptions(false, true); // 权重矩阵需要转置
    matmul_.setAddBias(use_bias, bias_);
}

void Linear::setParams(const std::vector<float>& weights, const std::vector<float>& bias) {
    // 检查权重大小
    if (weights.size() != static_cast<size_t>(in_features_ * out_features_)) {
        throw std::invalid_argument("权重大小不匹配");
    }
    
    // 复制权重数据
    std::copy(weights.begin(), weights.end(), weights_.data());
    
    // 检查和设置偏置（如果使用）
    if (use_bias_) {
        if (bias.empty()) {
            // 如果没有提供偏置，使用零初始化
            std::fill(bias_.begin(), bias_.end(), 0.0f);
        } else if (bias.size() != static_cast<size_t>(out_features_)) {
            throw std::invalid_argument("偏置大小不匹配");
        } else {
            bias_ = bias;
        }
        
        // 更新MatMul的偏置设置
        matmul_.setAddBias(true, bias_);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // 检查输入维度
    const auto& shape = input.shape();
    if (shape.size() < 1 || shape.back() != static_cast<size_t>(in_features_)) {
        throw std::invalid_argument("线性层输入维度不匹配");
    }
    
    // 对于2D输入，直接使用矩阵乘法
    if (shape.size() == 2) {
        return matmul_.forward(input, weights_);
    }
    
    // 对于更高维度的输入，先展平后两个维度，然后再重塑
    // 暂未实现，这部分代码仅处理2D情况
    throw std::runtime_error("线性层暂不支持>2D输入");
}

int Linear::getInFeatures() const {
    return in_features_;
}

int Linear::getOutFeatures() const {
    return out_features_;
}

const Tensor& Linear::getWeights() const {
    return weights_;
}

const std::vector<float>& Linear::getBias() const {
    return bias_;
}

bool Linear::useBias() const {
    return use_bias_;
}

} // namespace dhinference
