#include "../../include/ops/attention.h"
#include <stdexcept>
#include <cmath>

namespace dhinference {

MultiHeadAttention::MultiHeadAttention(int input_dim, int hidden_dim, int num_heads)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), num_heads_(num_heads) {
    
    // 确保hidden_dim能被num_heads整除
    if (hidden_dim % num_heads != 0) {
        throw std::invalid_argument("hidden_dim必须能被num_heads整除");
    }
    
    // 计算每个头的维度
    head_dim_ = hidden_dim / num_heads;
    
    // 初始化权重矩阵
    wq_ = Tensor({static_cast<size_t>(input_dim), static_cast<size_t>(hidden_dim)}, 0.0f);
    wk_ = Tensor({static_cast<size_t>(input_dim), static_cast<size_t>(hidden_dim)}, 0.0f);
    wv_ = Tensor({static_cast<size_t>(input_dim), static_cast<size_t>(hidden_dim)}, 0.0f);
    wo_ = Tensor({static_cast<size_t>(hidden_dim), static_cast<size_t>(hidden_dim)}, 0.0f);
}

void MultiHeadAttention::setWeights(
    const std::vector<float>& wq,
    const std::vector<float>& wk,
    const std::vector<float>& wv,
    const std::vector<float>& wo) {
    
    // 检查权重大小
    size_t expected_qkv_size = input_dim_ * hidden_dim_;
    size_t expected_o_size = hidden_dim_ * hidden_dim_;
    
    if (wq.size() != expected_qkv_size || wk.size() != expected_qkv_size || 
        wv.size() != expected_qkv_size || wo.size() != expected_o_size) {
        throw std::invalid_argument("注意力权重大小不匹配");
    }
    
    // 复制权重
    std::copy(wq.begin(), wq.end(), wq_.data());
    std::copy(wk.begin(), wk.end(), wk_.data());
    std::copy(wv.begin(), wv.end(), wv_.data());
    std::copy(wo.begin(), wo.end(), wo_.data());
}

Tensor MultiHeadAttention::forward(const Tensor& input) {
    // 检查输入维度
    const auto& shape = input.shape();
    if (shape.size() != 2 || shape[1] != static_cast<size_t>(input_dim_)) {
        throw std::invalid_argument("注意力层输入维度不匹配");
    }
    
    const size_t batch_size = shape[0];
    
    // 计算Q, K, V投影
    Tensor q = input.matmul(wq_);  // [batch_size, hidden_dim]
    Tensor k = input.matmul(wk_);  // [batch_size, hidden_dim]
    Tensor v = input.matmul(wv_);  // [batch_size, hidden_dim]
    
    // 计算注意力分数
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    
    // 计算QK^T
    Tensor k_t = k.transpose();  // [hidden_dim, batch_size]
    Tensor attention_scores = q.matmul(k_t) * scale;  // [batch_size, batch_size]
    
    // 应用softmax
    // 对每一行进行softmax
    for (size_t i = 0; i < batch_size; ++i) {
        float max_val = attention_scores.at({i, 0});
        for (size_t j = 1; j < batch_size; ++j) {
            max_val = std::max(max_val, attention_scores.at({i, j}));
        }
        
        float sum = 0.0f;
        for (size_t j = 0; j < batch_size; ++j) {
            float val = std::exp(attention_scores.at({i, j}) - max_val);
            attention_scores.at({i, j}) = val;
            sum += val;
        }
        
        for (size_t j = 0; j < batch_size; ++j) {
            attention_scores.at({i, j}) /= sum;
        }
    }
    
    // 计算注意力输出
    Tensor output = attention_scores.matmul(v);  // [batch_size, hidden_dim]
    
    // 通过输出投影
    output = output.matmul(wo_);  // [batch_size, hidden_dim]
    
    return output;
}

} // namespace dhinference
