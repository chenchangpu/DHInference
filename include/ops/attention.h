#ifndef DHINFERENCE_ATTENTION_H
#define DHINFERENCE_ATTENTION_H

#include <vector>
#include "../utils/tensor.h"

namespace dhinference {

// 多头注意力类
class MultiHeadAttention {
public:
    // 构造函数
    MultiHeadAttention(int input_dim, int hidden_dim, int num_heads);
    
    // 设置权重
    void setWeights(
        const std::vector<float>& wq,
        const std::vector<float>& wk,
        const std::vector<float>& wv,
        const std::vector<float>& wo
    );
    
    // 前向计算
    Tensor forward(const Tensor& input);

private:
    int input_dim_;
    int hidden_dim_;
    int num_heads_;
    int head_dim_;
    
    // 权重矩阵
    Tensor wq_;  // 查询权重
    Tensor wk_;  // 键权重
    Tensor wv_;  // 值权重
    Tensor wo_;  // 输出权重
};

} // namespace dhinference

#endif // DHINFERENCE_ATTENTION_H 