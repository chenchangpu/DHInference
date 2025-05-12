#ifndef DHINFERENCE_LAYERNORM_H
#define DHINFERENCE_LAYERNORM_H

#include <vector>
#include "../utils/tensor.h"

namespace dhinference {

// 层归一化类
class LayerNorm {
public:
    // 构造函数
    LayerNorm(int hidden_dim);
    
    // 设置参数
    void setParams(const std::vector<float>& gamma, const std::vector<float>& beta);
    
    // 前向计算
    Tensor forward(const Tensor& input);

private:
    int hidden_dim_;
    
    // 参数
    std::vector<float> gamma_;
    std::vector<float> beta_;
};

} // namespace dhinference

#endif // DHINFERENCE_LAYERNORM_H 