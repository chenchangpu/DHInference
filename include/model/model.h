#ifndef DHINFERENCE_MODEL_H
#define DHINFERENCE_MODEL_H

#include <string>
#include <vector>
#include <memory>
#include "../utils/tensor.h"
#include "../ops/activation.h"
#include "../ops/attention.h"
#include "../ops/layernorm.h"
#include "../ops/ffn.h"

namespace dhinference {

// 模型配置结构
struct ModelConfig {
    int n_layers;           // 层数
    int n_heads;            // 注意力头数
    int input_dim;          // 输入维度
    int layer_hidden_dim;   // 隐藏层维度
    int ffn_expansion;      // FFN扩展系数
    ActivationType act_func; // 激活函数类型
};

// Transformer层类
class Layer {
public:
    // 构造函数
    Layer(int input_dim, int hidden_dim, int num_heads, 
          int ffn_expansion, ActivationType activation);
    
    // 设置注意力权重
    void setAttentionWeights(
        const std::vector<float>& wq,
        const std::vector<float>& wk,
        const std::vector<float>& wv,
        const std::vector<float>& wo
    );
    
    // 设置注意力层归一化参数
    void setAttentionNormParams(
        const std::vector<float>& gamma,
        const std::vector<float>& beta
    );
    
    // 设置FFN权重
    void setFFNWeights(
        const std::vector<float>& w1,
        const std::vector<float>& w2
    );
    
    // 设置FFN层归一化参数
    void setFFNNormParams(
        const std::vector<float>& gamma,
        const std::vector<float>& beta
    );
    
    // 前向计算
    Tensor forward(const Tensor& input);

private:
    int input_dim_;
    int hidden_dim_;
    int num_heads_;
    int ffn_expansion_;
    ActivationType activation_;
    
    // 组件
    std::unique_ptr<LayerNorm> attention_norm_;
    std::unique_ptr<MultiHeadAttention> attention_;
    std::unique_ptr<LayerNorm> ffn_norm_;
    std::unique_ptr<FeedForward> ffn_;
};

// 模型类
class Model {
public:
    // 构造函数
    Model(const ModelConfig& config);
    
    // 从文件加载模型
    static std::unique_ptr<Model> loadFromFile(const std::string& filepath);
    
    // 前向推理 - 输入和输出都是张量
    Tensor forward(const Tensor& input);
    
    // 重载前向推理 - 输入和输出都是float向量
    std::vector<float> forward(const std::vector<float>& input);
    
    // 获取模型配置
    const ModelConfig& getConfig() const;
    
    // 获取层
    Layer* getLayer(int index);

private:
    ModelConfig config_;
    std::vector<std::unique_ptr<Layer>> layers_;
};

} // namespace dhinference

#endif // DHINFERENCE_MODEL_H 