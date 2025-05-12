#ifndef DHINFERENCE_FFN_H
#define DHINFERENCE_FFN_H

#include <vector>
#include "../utils/tensor.h"
#include "../ops/activation.h" // 引入激活函数

namespace dhinference {

// 前馈神经网络类
class FeedForward {
public:
    // 构造函数
    FeedForward(int hidden_dim, int expansion_factor, ActivationType activation);
    
    // 设置权重
    void setWeights(
        const std::vector<float>& w1,
        const std::vector<float>& w2
    );
    
    // 前向计算
    Tensor forward(const Tensor& input);

private:
    int hidden_dim_;
    int expanded_dim_;
    ActivationType activation_;
    
    // 权重矩阵
    Tensor w1_;  // 扩展权重
    Tensor w2_;  // 投影权重
    
    // 激活函数计算
    Tensor activate(const Tensor& input);
};

} // namespace dhinference

#endif // DHINFERENCE_FFN_H 