#include "../../include/ops/ffn.h"
#include "../../include/ops/activation.h"
#include <vector>

namespace dhinference {

FeedForward::FeedForward(int hidden_dim, int expansion_factor, ActivationType activation)
    : hidden_dim_(hidden_dim),
      expanded_dim_(hidden_dim * expansion_factor),
      activation_(activation) {
    // 初始化权重矩阵为零
    w1_ = Tensor({static_cast<size_t>(hidden_dim), static_cast<size_t>(expanded_dim_)}, 0.0f);
    w2_ = Tensor({static_cast<size_t>(expanded_dim_), static_cast<size_t>(hidden_dim)}, 0.0f);
}

void FeedForward::setWeights(const std::vector<float>& w1, const std::vector<float>& w2) {
    // 检查大小是否匹配
    if (w1.size() != hidden_dim_ * expanded_dim_) {
        throw std::invalid_argument("W1权重大小不匹配");
    }
    if (w2.size() != expanded_dim_ * hidden_dim_) {
        throw std::invalid_argument("W2权重大小不匹配");
    }
    
    // 复制权重数据
    std::copy(w1.begin(), w1.end(), w1_.data());
    std::copy(w2.begin(), w2.end(), w2_.data());
}

Tensor FeedForward::forward(const Tensor& input) {
    // 实现前馈网络前向传播: W2 * act(W1 * input)
    
    // 确保输入尺寸正确
    if (input.shape().size() != 2 || input.shape()[1] != static_cast<size_t>(hidden_dim_)) {
        throw std::invalid_argument("输入张量维度不匹配");
    }
    
    // W1 * input
    Tensor hidden = input.matmul(w1_);
    
    // 应用激活函数
    Tensor activated = activation::activate(hidden, activation_);
    
    // W2 * activated
    Tensor output = activated.matmul(w2_);
    
    return output;
}

Tensor FeedForward::activate(const Tensor& input) {
    // 使用activation命名空间中的函数
    return activation::activate(input, activation_);
}

} // namespace dhinference
