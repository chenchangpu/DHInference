#ifndef DHINFERENCE_MODEL_LOADER_H
#define DHINFERENCE_MODEL_LOADER_H

#include <string>
#include <memory>
#include <fstream>
#include <vector>
#include <cstring>
#include "backend_graph_model.hpp"

namespace dhinference {
    namespace backend{

// 模型配置结构
struct ModelConfig {
    int n_layers;           // 层数
    int n_heads;            // 注意力头数
    int input_dim;          // 输入维度
    int layer_hidden_dim;   // 隐藏层维度
    int ffn_expansion;      // FFN扩展系数
    ActivationType act_func; // 激活函数类型
};

// 模型加载器类
class ModelLoader {
public:
    // 构造函数
    ModelLoader(std::string model_file_path, ModelBackend model_backend);
    
    // 从文件加载模型
    void load_build_model_from_file(graph_model* model, Tensor* input_tensor);
    
private:
    ModelConfig model_config_;          // model config 成员
    std::ifstream model_file_;          // model file 成员
    std::string model_file_path_;       // model文件路径
    ModelBackend  model_backend_;       // backend类型

    // 解析模型配置
    ModelConfig parseModelConfig(std::ifstream& file);
    
    // 加载层参数
    Tensor* loadLayerParameters(graph_model* model, Tensor* input,
                            int input_dim, int hidden_dim, int ffn_expansion);
    
    // 加载注意力层参数
    Tensor* loadAttentionParams(graph_model* model, Tensor* input, int input_dim, int hidden_dim);
    
    // 加载LayerNorm参数
    Tensor* loadLayerNormParams(graph_model* model, Tensor* input, int hidden_dim);
    
    // 加载FFN参数
    Tensor* loadFFNParams(graph_model* model, Tensor* input,
                      int hidden_dim, int expanded_dim);
    
    // 从文件读取数组数据
    void readFloatArray(std::ifstream& file, float* array, size_t size);
    
    // 从文件读取矩阵数据
    void readFloatMatrix(std::ifstream& file, float* matrix, size_t rows, size_t cols);
    
    // 实用函数：从流中读取特定类型的数据
    template <typename T>
    T readFromStream(std::ifstream& file);
};

    }   // namespace backend
} // namespace dhinference

#endif // DHINFERENCE_MODEL_LOADER_H