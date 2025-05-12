#include "../../include/model/model.h"
#include "../../include/model/model_loader.h"
#include "../../include/utils/logger.h"
#include <stdexcept>

namespace dhinference {

// Layer类实现
Layer::Layer(int input_dim, int hidden_dim, int num_heads, 
            int ffn_expansion, ActivationType activation)
    : input_dim_(input_dim), 
      hidden_dim_(hidden_dim), 
      num_heads_(num_heads),
      ffn_expansion_(ffn_expansion),
      activation_(activation) {
    
    // 创建组件
    attention_norm_ = std::make_unique<LayerNorm>(input_dim);
    attention_ = std::make_unique<MultiHeadAttention>(input_dim, hidden_dim, num_heads);
    ffn_norm_ = std::make_unique<LayerNorm>(hidden_dim);
    ffn_ = std::make_unique<FeedForward>(hidden_dim, ffn_expansion, activation);
}

void Layer::setAttentionWeights(
    const std::vector<float>& wq,
    const std::vector<float>& wk,
    const std::vector<float>& wv,
    const std::vector<float>& wo) {
    
    attention_->setWeights(wq, wk, wv, wo);
}

void Layer::setAttentionNormParams(
    const std::vector<float>& gamma,
    const std::vector<float>& beta) {
    
    attention_norm_->setParams(gamma, beta);
}

void Layer::setFFNWeights(
    const std::vector<float>& w1,
    const std::vector<float>& w2) {
    
    ffn_->setWeights(w1, w2);
}

void Layer::setFFNNormParams(
    const std::vector<float>& gamma,
    const std::vector<float>& beta) {
    
    ffn_norm_->setParams(gamma, beta);
}

Tensor Layer::forward(const Tensor& input) {
    // 检查输入形状
    const auto& shape = input.shape();
    if (shape.size() != 2) {
        throw std::invalid_argument("Layer输入必须是2维张量");
    }
    
    // 第一个子层: Attention
    // 1. 对输入进行LayerNorm
    Tensor norm1 = attention_norm_->forward(input);  // [seq_len, hidden_dim]
    
    // 2. 计算注意力
    Tensor attn_out = attention_->forward(norm1);  // [seq_len, hidden_dim]
    
    // 3. 残差连接
    Tensor residual1 = input + attn_out;  // [seq_len, hidden_dim]
    
    // 第二个子层: FFN
    // 4. 对残差输出进行LayerNorm
    Tensor norm2 = ffn_norm_->forward(residual1);  // [seq_len, hidden_dim]
    
    // 5. 计算FFN
    Tensor ffn_out = ffn_->forward(norm2);  // [seq_len, hidden_dim]
    
    // 6. 残差连接
    Tensor residual2 = residual1 + ffn_out;  // [seq_len, hidden_dim]
    
    return residual2;
}

// Model类实现
Model::Model(const ModelConfig& config) : config_(config) {
    LOG_INFO("创建模型: 层数=" + std::to_string(config.n_layers));
    
    // 创建每一层
    for (int i = 0; i < config.n_layers; ++i) {
        // 第一层的输入维度是config.input_dim，其他层的输入维度是hidden_dim
        int input_dim = (i == 0) ? config.input_dim : config.layer_hidden_dim;
        
        layers_.push_back(std::make_unique<Layer>(
            input_dim,
            config.layer_hidden_dim,
            config.n_heads,
            config.ffn_expansion,
            config.act_func
        ));
        
        LOG_INFO("创建第 " + std::to_string(i+1) + " 层");
    }
}

std::unique_ptr<Model> Model::loadFromFile(const std::string& filepath) {
    ModelLoader loader;
    return loader.loadModel(filepath);
}

Tensor Model::forward(const Tensor& input) {
    Tensor output = input;
    
    // 依次通过每一层
    for (auto& layer : layers_) {
        output = layer->forward(output);
    }
    
    return output;
}

std::vector<float> Model::forward(const std::vector<float>& input) {
    // 计算序列长度

    size_t seq_len = input.size() / config_.input_dim;
    
    // 创建2维输入张量
    Tensor input_tensor(input.data(), {seq_len, static_cast<size_t>(config_.input_dim)});
    
    // 执行前向传播
    Tensor output_tensor = forward(input_tensor);
    
    // 获取输出数据
    return std::vector<float>(output_tensor.data(), output_tensor.data() + output_tensor.size());
}

const ModelConfig& Model::getConfig() const {
    return config_;
}

Layer* Model::getLayer(int index) {
    if (index < 0 || index >= static_cast<int>(layers_.size())) {
        throw std::out_of_range("层索引超出范围");
    }
    return layers_[index].get();
}

} // namespace dhinference 