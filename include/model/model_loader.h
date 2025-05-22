#ifndef DHINFERENCE_MODEL_LOADER_H
#define DHINFERENCE_MODEL_LOADER_H

#include <string>
#include <memory>
#include <fstream>
#include "model.h"

namespace dhinference {

// 模型加载器类
class ModelLoader {
public:
    // 构造函数
    ModelLoader();
    
    // 从文件加载模型
    std::unique_ptr<Model> loadModel(const std::string& filepath);
    
private:
    // 解析模型配置
    ModelConfig parseModelConfig(std::ifstream& file);
    
    // 加载层参数
    void loadLayerParameters(std::ifstream& file, Model& model, int layer_idx, 
                            int input_dim, int hidden_dim, int ffn_expansion);
    
    // 加载注意力层参数
    void loadAttentionParams(std::ifstream& file, Layer* layer, 
                            int input_dim, int hidden_dim);
    
    // 加载LayerNorm参数
    void loadLayerNormParams(std::ifstream& file, Layer* layer, int hidden_dim, bool is_ffn);
    
    // 加载FFN参数
    void loadFFNParams(std::ifstream& file, Layer* layer, 
                      int hidden_dim, int expanded_dim);
    
    // 从文件读取数组数据
    std::vector<float> readFloatArray(std::ifstream& file, size_t size);
    
    // 从文件读取矩阵数据
    std::vector<float> readFloatMatrix(std::ifstream& file, size_t rows, size_t cols);
    
    // 实用函数：从流中读取特定类型的数据
    template <typename T>
    T readFromStream(std::ifstream& file);
};

} // namespace dhinference

#endif // DHINFERENCE_MODEL_LOADER_H 