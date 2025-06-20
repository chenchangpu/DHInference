#include "../include/backend/utils/backend_model_loader.hpp"
#include "../include/backend/utils/backend_logger.hpp"
#include "../include/backend/utils/backend_tensor.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <filesystem>

// using namespace dhinference::backend;

// 创建一个测试模型文件
void createDummyModelFile(const std::string& filepath, int n_layers, int n_heads, 
                         int hidden_dim, int ffn_expansion) {
    // 若权重文件已存在，直接返回
    try {
        if (std::filesystem::exists(filepath) && std::filesystem::is_regular_file(filepath)) {
            std::cout << "dummy_model权重文件已存在" << std::endl;
            return;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "文件系统错误: " << e.what() << std::endl;
    }

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

// 生成随机输入
std::vector<float> generateRandomInput(int seq_len, int input_dim) {
    // 使用固定的种子
    std::mt19937 gen(42);  // 使用固定的种子42
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
bool isValidOutput(dhinference::backend::Tensor* result, int seq_len, int output_dim) {
    // 检查维度
    if (result->getrank() != 2) {
        std::cerr << "错误：输出维度不为2" << std::endl;
        return false;
    }
    
    if (result->shape(0) != seq_len || result->shape(1) != output_dim) {
        std::cerr << "错误：输出维度不匹配" << std::endl;
        return false;
    }
    
    // 检查是否包含NaN或无穷大
    float* result_ptr = result->data();
    for (int i = 0; i < result->size(); ++i) {
        if (std::isnan(result_ptr[i]) || std::isinf(result_ptr[i])) {
            std::cerr << "错误：输出包含NaN或无穷大" << std::endl;
            return false;
        }
    }
    
    return true;
}

int main() {
    // 设置日志
    dhinference::backend::Logger::getInstance().setLogLevel(dhinference::backend::LogLevel::DEBUG);
    
    // 创建测试模型文件
    std::string model_path = "dummy_model.bin";
    int n_layers = 2;
    int n_heads = 4;
    int hidden_dim = 4096;
    int ffn_expansion = 4;
    
    const int seq_len = 8192;
    const int input_dim = 4096;
    
    createDummyModelFile(model_path, n_layers, n_heads, hidden_dim, ffn_expansion);
    
    try {
        std::cout << "开始加载模型..." << std::endl;
        std::cout << "模型文件路径: " << model_path << std::endl;
        
        // 创建CUDA模型
        int MAX_LEAFS = 1024;
        int MAX_NODES = 1024;
        dhinference::backend::graph_model_cuda* model = new dhinference::backend::graph_model_cuda(MAX_LEAFS, MAX_NODES);
        model->alloc_extra_buff(32);  // 分配transpose空间

        // 创建输入tensor（初始在CPU上）
        std::vector<float> input_vec = generateRandomInput(seq_len, input_dim);
        int shape[2] = {seq_len, hidden_dim};
        dhinference::backend::Tensor* input_tensor = new dhinference::backend::Tensor(2, shape, input_vec.data(), dhinference::backend::BackendType::CPU);
        // 注意，input_tensor的owned属性为false，数据不属于自己！

        // 加载模型
        dhinference::backend::ModelLoader model_loader(model_path, dhinference::backend::ModelBackend::CUDA);
        model_loader.load_build_model_from_file(model, input_tensor);
        model->to_device();         // CPU->GPU
        
        // 预热
        for(int i = 0; i < 10; ++i){
            model->forward();
            model->reset_nodes();
        }

        // 执行推理
        std::cout << "开始执行推理..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        model->forward();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "推理时间: " << duration.count() << " ms" << std::endl;
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // std::cout << "推理时间: " << duration.count() << " us" << std::endl;
        
        // 获取结果并复制回CPU
        dhinference::backend::Tensor* result_gpu = model->get_result_tensor();
        float* result_cpu = (float*)malloc(result_gpu->size() * sizeof(float));
        copy_gpu_to_cpu(result_cpu, result_gpu->data(), result_gpu->size() * sizeof(float));
        
        // 创建CPU tensor用于验证
        dhinference::backend::Tensor* result_tensor = new dhinference::backend::Tensor(2, shape, result_cpu, dhinference::backend::BackendType::CPU);

        // 验证输出
        if (!isValidOutput(result_tensor, seq_len, input_dim)) {
            std::cerr << "推理测试失败" << std::endl;
            return 1;
        }
        
        // 统计结果
        std::vector<float> output_vec(result_cpu, result_cpu + result_gpu->size());
        std::cout << "推理测试成功！" << std::endl;
        std::cout << "输入维度: [" << seq_len << " x " << input_dim << "]" << std::endl;
        std::cout << "输出维度: [" << seq_len << " x " << input_dim << "]" << std::endl;
        
        // 打印统计信息
        float max_val = *std::max_element(output_vec.begin(), output_vec.end());
        float min_val = *std::min_element(output_vec.begin(), output_vec.end());
        float mean = 0.0f;
        for (float val : output_vec) {
            mean += val;
        }
        mean /= output_vec.size();
        
        std::cout << "输出统计信息：" << std::endl;
        std::cout << "  最大值: " << max_val << std::endl;
        std::cout << "  最小值: " << min_val << std::endl;
        std::cout << "  平均值: " << mean << std::endl;
        
        // 打印序列位置统计信息
        std::cout << "\n每个序列位置的统计信息：" << std::endl;
        for (int i = 0; i < seq_len; i += seq_len/10) {
            float pos_max = output_vec[i * input_dim];
            float pos_min = output_vec[i * input_dim];
            float pos_sum = 0.0f;
            
            for (int j = 0; j < input_dim; ++j) {
                float val = output_vec[i * input_dim + j];
                pos_max = std::max(pos_max, val);
                pos_min = std::min(pos_min, val);
                pos_sum += val;
            }
            
            std::cout << "  位置 " << i << ":" << std::endl;
            std::cout << "    最大值: " << pos_max << std::endl;
            std::cout << "    最小值: " << pos_min << std::endl;
            std::cout << "    平均值: " << (pos_sum / input_dim) << std::endl;
        }

        // 释放资源
        delete model;         // 这会释放所有tensor和extra_buff
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}