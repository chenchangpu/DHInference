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
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法创建模型文件: " << filepath << std::endl;
        return;
    }
    
    // 设置随机生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    // 写入模型配置
    file.write(reinterpret_cast<const char*>(&n_layers), sizeof(int));
    file.write(reinterpret_cast<const char*>(&n_heads), sizeof(int));
    
    // 写入激活函数类型 (GELU = 1)
    int act_func = 1;
    file.write(reinterpret_cast<const char*>(&act_func), sizeof(int));
    
    // 写入隐藏层维度
    file.write(reinterpret_cast<const char*>(&hidden_dim), sizeof(int));
    
    // 写入FFN扩展系数
    file.write(reinterpret_cast<const char*>(&ffn_expansion), sizeof(int));
    
    // 输入维度固定为128
    int input_dim = 128;
    
    // 为每一层写入参数
    for (int layer = 0; layer < n_layers; ++layer) {
        std::cout << "生成第 " << (layer+1) << " 层参数..." << std::endl;
        
        // 当前层的输入维度，第一层是input_dim，其他层是hidden_dim
        int curr_input_dim = (layer == 0) ? input_dim : hidden_dim;
        
        // 写入Wq (curr_input_dim x hidden_dim)
        for (int i = 0; i < curr_input_dim; ++i) {
            for (int j = 0; j < hidden_dim; ++j) {
                float val = dist(gen);
                file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        
        // 写入Wk (curr_input_dim x hidden_dim)
        for (int i = 0; i < curr_input_dim; ++i) {
            for (int j = 0; j < hidden_dim; ++j) {
                float val = dist(gen);
                file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        
        // 写入Wv (curr_input_dim x hidden_dim)
        for (int i = 0; i < curr_input_dim; ++i) {
            for (int j = 0; j < hidden_dim; ++j) {
                float val = dist(gen);
                file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        
        // 写入Wo (hidden_dim x hidden_dim)
        for (int i = 0; i < hidden_dim; ++i) {
            for (int j = 0; j < hidden_dim; ++j) {
                float val = dist(gen);
                file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        
        // 写入注意力LayerNorm参数
        for (int i = 0; i < hidden_dim; ++i) {
            float val = 1.0f + dist(gen) * 0.01f;  // gamma接近1
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        
        for (int i = 0; i < hidden_dim; ++i) {
            float val = dist(gen) * 0.01f;  // beta接近0
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        
        // 写入FFN参数
        int expanded_dim = hidden_dim * ffn_expansion;
        
        // 写入W1 (hidden_dim x expanded_dim)
        for (int i = 0; i < hidden_dim; ++i) {
            for (int j = 0; j < expanded_dim; ++j) {
                float val = dist(gen);
                file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        
        // 写入W2 (expanded_dim x hidden_dim)
        for (int i = 0; i < expanded_dim; ++i) {
            for (int j = 0; j < hidden_dim; ++j) {
                float val = dist(gen);
                file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        
        // 写入FFN LayerNorm参数
        for (int i = 0; i < hidden_dim; ++i) {
            float val = 1.0f + dist(gen) * 0.01f;  // gamma接近1
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        
        for (int i = 0; i < hidden_dim; ++i) {
            float val = dist(gen) * 0.01f;  // beta接近0
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
    std::string model_path = "dummy_model.bin";
    int n_layers = 2;
    int n_heads = 4;
    int hidden_dim = 64;
    int ffn_expansion = 4;
    
    createDummyModelFile(model_path, n_layers, n_heads, hidden_dim, ffn_expansion);
    
    // 加载模型
    try {
        std::cout << "开始加载模型..." << std::endl;
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