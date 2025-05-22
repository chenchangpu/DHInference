#include "../../include/model/model_loader.h"
#include "../../include/utils/logger.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace dhinference {

ModelLoader::ModelLoader() {
    // 初始化日志记录器
    Logger::getInstance().setLogLevel(LogLevel::INFO);
}

std::unique_ptr<Model> ModelLoader::loadModel(const std::string& filepath) {
    LOG_INFO("开始加载模型: " + filepath);
    
    // 打开模型文件
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开模型文件: " + filepath);
    }
    
    // 读取模型配置
    ModelConfig config = parseModelConfig(file);
    
    // 创建模型实例
    auto model = std::make_unique<Model>(config);
    
    // 读取每一层的参数
    int input_dim = config.input_dim;
    for (int i = 0; i < config.n_layers; ++i) {
        LOG_INFO("加载第 " + std::to_string(i+1) + " 层参数");
        
        // 加载第i层参数
        loadLayerParameters(file, *model, i, input_dim, config.layer_hidden_dim, config.ffn_expansion);
        
        // 下一层的输入维度是上一层的hidden_dim（除了第一层）
        // 在本例中，简化为所有层使用相同的hidden_dim
        input_dim = config.layer_hidden_dim;
    }
    
    LOG_INFO("模型加载完成");
    return model;
}

ModelConfig ModelLoader::parseModelConfig(std::ifstream& file) {
    LOG_INFO("解析模型配置");
    
    ModelConfig config;
    
    // 读取基本配置
    config.n_layers = readFromStream<int>(file);
    config.n_heads = readFromStream<int>(file);
    
    // 读取激活函数类型
    int act_func_int = readFromStream<int>(file);
    config.act_func = static_cast<ActivationType>(act_func_int);
    
    // 读取隐藏层维度
    config.layer_hidden_dim = readFromStream<int>(file);
    
    // 读取FFN扩展系数
    config.ffn_expansion = readFromStream<int>(file);
    
    // 设置输入维度（=hidden_dim）
    config.input_dim = config.layer_hidden_dim;
    
    LOG_INFO("模型配置: 层数=" + std::to_string(config.n_layers) +
             ", 头数=" + std::to_string(config.n_heads) +
             ", 激活函数=" + std::to_string(static_cast<int>(config.act_func)) +
             ", 隐藏维度=" + std::to_string(config.layer_hidden_dim) +
             ", FFN扩展=" + std::to_string(config.ffn_expansion));
    
    return config;
}

void ModelLoader::loadLayerParameters(std::ifstream& file, Model& model, int layer_idx, 
                                       int input_dim, int hidden_dim, int ffn_expansion) {
    Layer* layer = model.getLayer(layer_idx);
    
    // 加载注意力层的LayerNorm参数
    // 第一层的attention_norm使用input_dim，其他层使用hidden_dim
    int attention_norm_dim = (layer_idx == 0) ? input_dim : hidden_dim;
    loadLayerNormParams(file, layer, attention_norm_dim, false);

    // 加载注意力层参数
    loadAttentionParams(file, layer, input_dim, hidden_dim);
    
    // 加载FFN层的LayerNorm参数
    loadLayerNormParams(file, layer, hidden_dim, true);

    // 加载FFN参数
    int expanded_dim = hidden_dim * ffn_expansion;
    loadFFNParams(file, layer, hidden_dim, expanded_dim);

}

void ModelLoader::loadAttentionParams(std::ifstream& file, Layer* layer, 
                                       int input_dim, int hidden_dim) {
    LOG_INFO("加载注意力层参数");
    
    // 读取Wq矩阵 (input_dim x hidden_dim)
    std::vector<float> wq = readFloatMatrix(file, input_dim, hidden_dim);
    
    // 读取Wk矩阵 (input_dim x hidden_dim)
    std::vector<float> wk = readFloatMatrix(file, input_dim, hidden_dim);
    
    // 读取Wv矩阵 (input_dim x hidden_dim)
    std::vector<float> wv = readFloatMatrix(file, input_dim, hidden_dim);
    
    // 读取Wo矩阵 (hidden_dim x hidden_dim)
    std::vector<float> wo = readFloatMatrix(file, hidden_dim, hidden_dim);
    
    // 设置注意力层权重
    layer->setAttentionWeights(wq, wk, wv, wo);
}

void ModelLoader::loadLayerNormParams(std::ifstream& file, Layer* layer, 
                                       int hidden_dim, bool is_ffn) {
    LOG_INFO("加载LayerNorm参数");
    
    // 读取gamma向量 (hidden_dim)
    std::vector<float> gamma = readFloatArray(file, hidden_dim);
    
    // 读取beta向量 (hidden_dim)
    std::vector<float> beta = readFloatArray(file, hidden_dim);
    
    // 设置LayerNorm参数
    if (is_ffn) {
        layer->setFFNNormParams(gamma, beta);
    } else {
        layer->setAttentionNormParams(gamma, beta);
    }
}

void ModelLoader::loadFFNParams(std::ifstream& file, Layer* layer, 
                                int hidden_dim, int expanded_dim) {
    LOG_INFO("加载FFN参数");
    
    // 读取W1矩阵 (hidden_dim x expanded_dim)
    std::vector<float> w1 = readFloatMatrix(file, hidden_dim, expanded_dim);
    
    // 读取W2矩阵 (expanded_dim x hidden_dim)
    std::vector<float> w2 = readFloatMatrix(file, expanded_dim, hidden_dim);
    
    // 设置FFN权重
    layer->setFFNWeights(w1, w2);
}

// // load 1-D array
// std::vector<float> ModelLoader::readFloatArray(std::ifstream& file, size_t size) {
//     std::vector<float> array(size);
//     for (size_t i = 0; i < size; ++i) {
//         array[i] = readFromStream<float>(file);
//     }
//     return array;
// }

// // load 2-D matrix
// std::vector<float> ModelLoader::readFloatMatrix(std::ifstream& file, size_t rows, size_t cols) {
//     std::vector<float> matrix(rows * cols);
//     for (size_t i = 0; i < rows; ++i) {
//         for (size_t j = 0; j < cols; ++j) {
//             matrix[i * cols + j] = readFromStream<float>(file);
//         }
//     }
//     return matrix;
// }

std::vector<float> ModelLoader::readFloatArray(std::ifstream& file, size_t size) {
    std::vector<float> array(size);
    // 一次性读取整个数组，减少IO操作
    file.read(reinterpret_cast<char*>(array.data()), size * sizeof(float));
    return array;
}

std::vector<float> ModelLoader::readFloatMatrix(std::ifstream& file, size_t rows, size_t cols) {
    std::vector<float> matrix(rows * cols);
    // 一次性读取整个矩阵，减少IO操作
    file.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(float));
    return matrix;
}

template <typename T>
T ModelLoader::readFromStream(std::ifstream& file) {
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file) {
        throw std::runtime_error("读取模型文件时出错");
    }
    return value;
}

} // namespace dhinference 