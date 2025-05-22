#include "../include/model/model.h"
#include "../include/utils/logger.h"
#include <iostream>
#include <fstream>
#include <random>
#include <string>

using namespace dhinference;

// 创建一个测试模型文件
void createDummyModelFile(const std::string& filepath, int n_layers, int n_heads, 
                         int hidden_dim, int ffn_expansion) {
    // 复用model_loader_test.cpp中的函数
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法创建模型文件: " << filepath << std::endl;
        return;
    }
    
    // 使用固定的种子
    std::mt19937 gen(42);  // 使用固定的种子42
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    // 写入模型配置
    file.write(reinterpret_cast<const char*>(&n_layers), sizeof(int));
    file.write(reinterpret_cast<const char*>(&n_heads), sizeof(int));
    
    int act_func = 0;  // RELU
    file.write(reinterpret_cast<const char*>(&act_func), sizeof(int));
    file.write(reinterpret_cast<const char*>(&hidden_dim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&ffn_expansion), sizeof(int));
    
    int input_dim = hidden_dim;         // input_dim == hidden_dim
    
    // 为每一层写入参数
    for (int layer = 0; layer < n_layers; ++layer) {
        int curr_input_dim = (layer == 0) ? input_dim : hidden_dim;

        // 写入LayerNorm参数
        // 第一层的attention_norm使用input_dim，其他层使用hidden_dim
        for (int i = 0; i < curr_input_dim; ++i) {
            float val = 1.0f + dist(gen) * 0.01f;
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        for (int i = 0; i < curr_input_dim; ++i) {
            float val = dist(gen) * 0.01f;
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        
        // 写入注意力层参数
        for (int i = 0; i < curr_input_dim * hidden_dim; ++i) {
            float val = dist(gen);
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        
        for (int i = 0; i < curr_input_dim * hidden_dim; ++i) {
            float val = dist(gen);
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        
        for (int i = 0; i < curr_input_dim * hidden_dim; ++i) {
            float val = dist(gen);
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        
        for (int i = 0; i < hidden_dim * hidden_dim; ++i) {
            float val = dist(gen);
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }

        // 写入FFN LayerNorm参数
        for (int i = 0; i < hidden_dim; ++i) {
            float val = 1.0f + dist(gen) * 0.01f;
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        for (int i = 0; i < hidden_dim; ++i) {
            float val = dist(gen) * 0.01f;
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        
        // 写入FFN参数
        int expanded_dim = hidden_dim * ffn_expansion;
        for (int i = 0; i < hidden_dim * expanded_dim; ++i) {
            float val = dist(gen);
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        for (int i = 0; i < expanded_dim * hidden_dim; ++i) {
            float val = dist(gen);
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }
    
    file.close();
    std::cout << "已创建测试模型文件: " << filepath << std::endl;
}

int main() {
    // 设置日志
    Logger::getInstance().setLogLevel(LogLevel::DEBUG);
    
    // 创建测试模型文件
    std::string model_path = std::string(CMAKE_BINARY_DIR) + "/bin/dummy_model.bin";
    int n_layers = 2;
    int n_heads = 4;
    int hidden_dim = 64;
    int ffn_expansion = 4;
    
    createDummyModelFile(model_path, n_layers, n_heads, hidden_dim, ffn_expansion);
    
    // 加载模型
    try {
        std::cout << "开始加载模型..." << std::endl;
        std::cout << "模型文件路径: " << model_path << std::endl;
        auto model = Model::loadFromFile(model_path);
        
        // 获取模型配置
        const ModelConfig& config = model->getConfig();
        
        std::cout << "模型加载成功：" << std::endl;
        std::cout << "  层数: " << config.n_layers << std::endl;
        std::cout << "  头数: " << config.n_heads << std::endl;
        std::cout << "  激活函数: " << static_cast<int>(config.act_func) << std::endl;
        std::cout << "  隐藏维度: " << config.layer_hidden_dim << std::endl;
        std::cout << "  FFN扩展系数: " << config.ffn_expansion << std::endl;
        
        std::cout << "模型参数加载功能测试成功！" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 