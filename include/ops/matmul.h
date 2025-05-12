#ifndef DHINFERENCE_MATMUL_H
#define DHINFERENCE_MATMUL_H

#include <vector>
#include "../utils/tensor.h"

namespace dhinference {

// 矩阵乘法运算类
class MatMul {
public:
    // 构造函数
    MatMul();
    
    // 前向计算
    Tensor forward(const Tensor& A, const Tensor& B);
    
    // 设置转置选项
    void setTransposeOptions(bool transposeA, bool transposeB);
    
    // 设置是否需要添加偏置
    void setAddBias(bool addBias, const std::vector<float>& bias = {});

private:
    bool transpose_a_;
    bool transpose_b_;
    bool add_bias_;
    std::vector<float> bias_;
};

// 线性层(全连接层)
class Linear {
public:
    // 构造函数
    Linear(int in_features, int out_features, bool use_bias = true);
    
    // 设置权重和偏置
    void setParams(const std::vector<float>& weights, const std::vector<float>& bias = {});
    
    // 前向计算
    Tensor forward(const Tensor& input);
    
    // 获取输入输出特征维度
    int getInFeatures() const;
    int getOutFeatures() const;
    
    // 获取权重张量
    const Tensor& getWeights() const;
    
    // 获取偏置向量
    const std::vector<float>& getBias() const;
    
    // 是否使用偏置
    bool useBias() const;

private:
    int in_features_;
    int out_features_;
    bool use_bias_;
    Tensor weights_;           // 权重矩阵
    std::vector<float> bias_;  // 偏置向量
    MatMul matmul_;            // 矩阵乘法运算
};

} // namespace dhinference

#endif // DHINFERENCE_MATMUL_H 