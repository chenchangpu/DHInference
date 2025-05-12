#include "../include/model/model.h"
#include "../include/utils/logger.h"
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

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
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    // 写入模型配置
    file.write(reinterpret_cast<const char*>(&n_layers), sizeof(int));
    file.write(reinterpret_cast<const char*>(&n_heads), sizeof(int));
    
    int act_func = 1;  // GELU
    file.write(reinterpret_cast<const char*>(&act_func), sizeof(int));
    file.write(reinterpret_cast<const char*>(&hidden_dim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&ffn_expansion), sizeof(int));
    
    int input_dim = hidden_dim;         // input_dim == hidden_dim
    
    // 为每一层写入参数
    for (int layer = 0; layer < n_layers; ++layer) {
        int curr_input_dim = (layer == 0) ? input_dim : hidden_dim;
        
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
        
        // 写入LayerNorm参数
        // 第一层的attention_norm使用input_dim，其他层使用hidden_dim
        int attention_norm_dim =  hidden_dim;
        for (int i = 0; i < attention_norm_dim; ++i) {
            float val = 1.0f + dist(gen) * 0.01f;
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        for (int i = 0; i < attention_norm_dim; ++i) {
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
        
        // 写入FFN LayerNorm参数
        for (int i = 0; i < hidden_dim; ++i) {
            float val = 1.0f + dist(gen) * 0.01f;
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
        for (int i = 0; i < hidden_dim; ++i) {
            float val = dist(gen) * 0.01f;
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }
    
    file.close();
    std::cout << "已创建测试模型文件: " << filepath << std::endl;
}

// 生成随机输入
std::vector<float> generateRandomInput(int seq_len, int input_dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // 创建2维输入，展平为1维vector
    std::vector<float> input(seq_len * input_dim);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            input[i * input_dim + j] = dist(gen);
        }
    }
    return input;
}

// 检查输出是否有效
bool isValidOutput(const std::vector<float>& output, int seq_len, int output_dim) {
    // 检查维度
    if (output.empty()) {
        std::cerr << "错误：输出为空" << std::endl;
        return false;
    }
    
    if (output.size() != seq_len * output_dim) {
        std::cerr << "错误：输出维度不匹配，期望 " << seq_len * output_dim 
                  << "，实际 " << output.size() << std::endl;
        return false;
    }
    
    // 检查是否包含NaN或无穷大
    for (float val : output) {
        if (std::isnan(val) || std::isinf(val)) {
            std::cerr << "错误：输出包含NaN或无穷大" << std::endl;
            return false;
        }
    }
    
    // 检查数值范围
    float max_val = *std::max_element(output.begin(), output.end());
    float min_val = *std::min_element(output.begin(), output.end());
    if (std::abs(max_val) > 1e6 || std::abs(min_val) > 1e6) {
        std::cerr << "错误：输出值范围异常" << std::endl;
        return false;
    }
    
    return true;
}

int main() {
    // 设置日志
    Logger::getInstance().setLogLevel(LogLevel::DEBUG);
    
    // 创建测试模型文件
    std::string model_path = "dummy_model.bin";
    int n_layers = 2;
    int n_heads = 4;
    int hidden_dim = 128;      // 默认128
    int ffn_expansion = 4;
    
    // 设置序列长度和输入维度
    const int seq_len = 1024;
    const int input_dim = 128; // 默认128
    
    createDummyModelFile(model_path, n_layers, n_heads, hidden_dim, ffn_expansion);
    
    try {
        // 加载模型
        std::cout << "开始加载模型..." << std::endl;
        auto model = Model::loadFromFile(model_path);
        
        // 生成随机输入
        std::vector<float> input = generateRandomInput(seq_len, input_dim);
        
        // 执行推理
        std::cout << "开始执行推理..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();  // 开始时间
        std::vector<float> output = model->forward(input);
        auto end = std::chrono::high_resolution_clock::now();    // 结束时间
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
        std::cout << "推理时间: " << duration.count() << " ms" << std::endl;
        
        // 验证输出
        if (!isValidOutput(output, seq_len, input_dim)) {
            std::cerr << "推理测试失败" << std::endl;
            return 1;
        }
        
        std::cout << "推理测试成功！" << std::endl;
        std::cout << "输入维度: [" << seq_len << " x " << input_dim << "]" << std::endl;
        std::cout << "输出维度: [" << seq_len << " x " << input_dim << "]" << std::endl;
        
        // 打印一些统计信息
        float max_val = *std::max_element(output.begin(), output.end());
        float min_val = *std::min_element(output.begin(), output.end());
        float mean = 0.0f;
        for (float val : output) {
            mean += val;
        }
        mean /= output.size();
        
        std::cout << "输出统计信息：" << std::endl;
        std::cout << "  最大值: " << max_val << std::endl;
        std::cout << "  最小值: " << min_val << std::endl;
        std::cout << "  平均值: " << mean << std::endl;
        
        // 打印每个序列位置的统计信息
        std::cout << "\n每个序列位置的统计信息：" << std::endl;
        for (int i = 0; i < seq_len; i += seq_len/10) {  // 只打印10个位置的信息
            float pos_max = output[i * input_dim];
            float pos_min = output[i * input_dim];
            float pos_sum = 0.0f;
            
            for (int j = 0; j < input_dim; ++j) {
                float val = output[i * input_dim + j];
                pos_max = std::max(pos_max, val);
                pos_min = std::min(pos_min, val);
                pos_sum += val;
            }
            
            std::cout << "  位置 " << i << ":" << std::endl;
            std::cout << "    最大值: " << pos_max << std::endl;
            std::cout << "    最小值: " << pos_min << std::endl;
            std::cout << "    平均值: " << (pos_sum / input_dim) << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}