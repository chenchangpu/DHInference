#include "backend/utils/backend_logger.hpp"
#include "backend/utils/backend_model_loader.hpp"
#include "backend/utils/backend_graph_model.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>

namespace dhinference {
    namespace backend{

ModelLoader::ModelLoader(std::string model_file_path, ModelBackend model_backend) {
    model_backend_ = model_backend;
    model_file_path_ = model_file_path;
    // 初始化日志记录器
    Logger::getInstance().setLogLevel(LogLevel::INFO);
    // 打开文件
    model_file_.open(model_file_path, std::ios::binary | std::ios::in);
    if (!model_file_.is_open()) {
        throw std::runtime_error("无法打开模型文件: " + model_file_path);
    }
    // 读取文件，获得model config
    model_config_ = parseModelConfig(model_file_);
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

void ModelLoader::load_build_model_from_file(graph_model* model, Tensor* input_tensor) {
    LOG_INFO("开始加载模型: " + model_file_path_);

    // 设置input tensor
    model->set_input_tensor(input_tensor);
    input_tensor->setisleaf(true);
    
    // 读取每一层的参数
    int input_dim = model_config_.input_dim;
    int hidden_dim = model_config_.layer_hidden_dim;
    int ffn_expansion = model_config_.ffn_expansion;
    
    Tensor* layer_in_tensor = input_tensor;
    Tensor* output_tensor;
    
    for (int i = 0; i < model_config_.n_layers; ++i) {
        LOG_INFO("加载第 " + std::to_string(i+1) + " 层参数");
        int layer_in_dim = (i == 0) ? input_dim : hidden_dim;
        output_tensor = loadLayerParameters(model, layer_in_tensor, layer_in_dim, hidden_dim, ffn_expansion);
        layer_in_tensor = output_tensor;
        if(i == model_config_.n_layers - 1){
            model->set_result_tesnor(output_tensor);
        }
    }
    
    // 将模型参数移动到对应设备
    // model->to_device();
    
    LOG_INFO("模型加载完成");
}

// one layer
Tensor* ModelLoader::loadLayerParameters(graph_model* model, Tensor* input, 
                                       int input_dim, int hidden_dim, int ffn_expansion) {
    Tensor* norm1_t = loadLayerNormParams(model, input, input_dim);
    norm1_t->set_owns_data(true);
    Tensor* attn_t = loadAttentionParams(model, norm1_t, input_dim, hidden_dim);
    attn_t->set_owns_data(true);

    // residual
    size_t size = input->size() * sizeof(float);
    int shape[2] = {input->shape(0), input->shape(1)};
    float* residual1_ptr = (float*)malloc(size);
    Tensor* residual1_t = new Tensor(2, shape, residual1_ptr);
    model->add_op_elementwise_add(input, attn_t, residual1_t);
    residual1_t->set_owns_data(true);

    Tensor* norm2_t = loadLayerNormParams(model, residual1_t, hidden_dim);
    norm2_t->set_owns_data(true);
    Tensor* ffn_t = loadFFNParams(model, norm2_t, hidden_dim, ffn_expansion * hidden_dim);
    ffn_t->set_owns_data(true);

    // residual
    float* residual2_ptr = (float*)malloc(size);
    Tensor* residual2_t = new Tensor(2, shape, residual2_ptr);
    model->add_op_elementwise_add(residual1_t, ffn_t, residual2_t);
    residual2_t->set_owns_data(true);

    return residual2_t;
}

// 返回attention的输出Tensor
Tensor* ModelLoader::loadAttentionParams(graph_model* model, Tensor* input, int input_dim, int hidden_dim) {
    LOG_INFO("加载注意力层参数");

    int head_dim = model_config_.layer_hidden_dim / model_config_.n_heads;
    float scale = 1 / sqrtf(head_dim);
    
    // 读取Wq矩阵 (input_dim x hidden_dim)
    float* wq_ptr = (float*)malloc(input_dim * hidden_dim * sizeof(float));
    if (!wq_ptr) {
        throw std::runtime_error("Failed to allocate memory for Wq");
    }
    
    // 读取Wk矩阵 (input_dim x hidden_dim)
    float* wk_ptr = (float*)malloc(input_dim * hidden_dim * sizeof(float));
    if (!wk_ptr) {
        free(wq_ptr);
        throw std::runtime_error("Failed to allocate memory for Wk");
    }
    
    // 读取Wv矩阵 (input_dim x hidden_dim)
    float* wv_ptr = (float*)malloc(input_dim * hidden_dim * sizeof(float));
    if (!wv_ptr) {
        free(wq_ptr);
        free(wk_ptr);
        throw std::runtime_error("Failed to allocate memory for Wv");
    }
    
    // 读取Wo矩阵 (hidden_dim x hidden_dim)
    float* wo_ptr = (float*)malloc(input_dim * hidden_dim * sizeof(float));
    if (!wo_ptr) {
        free(wq_ptr);
        free(wk_ptr);
        free(wv_ptr);
        throw std::runtime_error("Failed to allocate memory for Wo");
    }
    
    // 设置tensor
    int shape[2] = {input_dim, hidden_dim};
    Tensor* Wq_t = new Tensor(2, shape, wq_ptr);    // input_dim x hidden_dim
    readFloatMatrix(model_file_, wq_ptr, input_dim, hidden_dim);
    Wq_t->setisleaf(true);                          // 参数节点设置为leaf

    Tensor* Wk_t = new Tensor(2, shape, wk_ptr);
    readFloatMatrix(model_file_, wk_ptr, input_dim, hidden_dim);
    Wk_t->setisleaf(true);

    Tensor* Wv_t = new Tensor(2, shape, wv_ptr);
    readFloatMatrix(model_file_, wv_ptr, input_dim, hidden_dim);
    Wv_t->setisleaf(true);

    shape[0] = hidden_dim;
    Tensor* Wo_t = new Tensor(2, shape, wo_ptr);    // hidden_dim x hidden_dim
    readFloatMatrix(model_file_, wo_ptr, hidden_dim, hidden_dim);
    Wo_t->setisleaf(true);

    // ============== build ==============
    try {
        // 申请tensor空间
        shape[0] = input->shape(0);
        shape[1] = hidden_dim;
        size_t size = shape[0] * shape[1] * sizeof(float);
        float* Q_ptr = nullptr;                                 // 中间tensor的host可以先设置为null，后面直接分配显存空间
        Tensor* Q_t = new Tensor(2, shape, Q_ptr);  
        Q_t->setisleaf(false);
        
        float* K_ptr = nullptr;
        Tensor* K_t = new Tensor(2, shape, K_ptr);  
        K_t->setisleaf(false);
        
        float* V_ptr = nullptr;
        Tensor* V_t = new Tensor(2, shape, V_ptr);  
        V_t->setisleaf(false);

        int shape_3d[3] = { input->shape(0), model_config_.n_heads, head_dim};      // [seq_len, n_head, head_dim]
        Tensor* Q_reshape_t = new Tensor(3, shape_3d, Q_ptr);      // 解释shape为3维
        Q_reshape_t->setisleaf(false);
        Q_reshape_t->set_owns_data(false);                  // reshape其他的tensor, 并不自己拥有数据
        Tensor* K_reshape_t = new Tensor(3, shape_3d, K_ptr);      // 解释shape为3维
        K_reshape_t->setisleaf(false);
        K_reshape_t->set_owns_data(false);                  // reshape其他的tensor, 并不自己拥有数据
        Tensor* V_reshape_t = new Tensor(3, shape_3d, V_ptr);      // 解释shape为3维
        V_reshape_t->setisleaf(false);
        V_reshape_t->set_owns_data(false);                  // reshape其他的tensor, 并不自己拥有数据

        shape_3d[0] = model_config_.n_heads;
        shape_3d[1] = input->shape(0);
        shape_3d[2] = head_dim;
        float* Q_perm_ptr = nullptr;
        Tensor* Q_perm_t = new Tensor(3, shape_3d, Q_perm_ptr);

        float* V_perm_ptr = nullptr;
        Tensor* V_perm_t = new Tensor(3, shape_3d, V_perm_ptr);

        shape_3d[0] = model_config_.n_heads;
        shape_3d[1] = head_dim; 
        shape_3d[2] = input->shape(0);
        float* K_perm_ptr = nullptr;
        Tensor* K_perm_t = new Tensor(3, shape_3d, K_perm_ptr);

        shape_3d[0] = model_config_.n_heads;
        shape_3d[1] = shape_3d[2] = input->shape(0); // seq_len
        size = shape_3d[0] * shape_3d[1] * shape_3d[2] * sizeof(float);
        float* QK_ptr = nullptr;
        Tensor* QK_t = new Tensor(3, shape_3d, QK_ptr);    
        QK_t->setisleaf(false);
        QK_t->setscale(scale);          // QK有scale，需要设置

        float* P_ptr = nullptr;
        Tensor* P_t = new Tensor(3, shape_3d, P_ptr);      
        P_t->setisleaf(false);

        shape_3d[0] = model_config_.n_heads;
        shape_3d[1] = input->shape(0); // seq_len
        shape_3d[2] = head_dim;
        size = shape_3d[0] * shape_3d[1] * shape_3d[2] * sizeof(float);
        float* PV_ptr = nullptr;
        Tensor* PV_t = new Tensor(3, shape_3d, PV_ptr);    
        PV_t->setisleaf(false);

        shape_3d[0] = input->shape(0);
        shape_3d[1] = model_config_.n_heads;
        shape_3d[2] = head_dim;
        size = shape_3d[0] * shape_3d[1] * shape_3d[2] * sizeof(float);
        float* PV_perm_ptr = nullptr;
        Tensor* PV_perm_t = new Tensor(3, shape_3d, PV_perm_ptr);    
        PV_perm_t->setisleaf(false);

        shape[0] = input->shape(0);
        shape[1] = hidden_dim;
        Tensor* PV_perm_reshape_t = new Tensor(2, shape);      // 解释shape为2维
        PV_perm_reshape_t->setisleaf(false);
        PV_perm_reshape_t->set_owns_data(false);                  // reshape其他的tensor, 并不自己拥有数据

        float* O_ptr = nullptr;
        Tensor* O_t = new Tensor(2, shape, O_ptr);      
        O_t->setisleaf(false);

        /////////////////////////////////////////

        // 构建计算
        model->add_op_matrix_matrix_mul(input, Wq_t, Q_t);
        model->add_op_matrix_matrix_mul(input, Wk_t, K_t);
        model->add_op_matrix_matrix_mul(input, Wv_t, V_t);   // Q, K, V

        model->add_op_matrix_reshape(Q_t, Q_reshape_t);     // reshape
        model->add_op_matrix_reshape(K_t, K_reshape_t);
        model->add_op_matrix_reshape(V_t, V_reshape_t);

        model->add_op_matrix_permutation_102(Q_reshape_t, Q_perm_t);    // [seq_len, n_head, h_dim] -> [n_head, seq_len, h_dim]
        model->add_op_matrix_permutation_120(K_reshape_t, K_perm_t);    // [seq_len, n_head, h_dim] -> [n_head, h_dim, seq_len]
        model->add_op_matrix_permutation_102(V_reshape_t, V_perm_t);    // [seq_len, n_head, h_dim] -> [n_head, seq_len, h_dim]

        model->add_op_batch_matrix_matrix_mul(Q_perm_t, K_perm_t, QK_t);   // batch gemm: Q_perm_t x K_perm_t
        model->add_op_matrix_softmax(QK_t, P_t);             // softmax

        model->add_op_batch_matrix_matrix_mul(P_t, V_perm_t, PV_t);     // batch gemm: P x V
        model->add_op_matrix_permutation_102(PV_t, PV_perm_t);     // batch gemm: P x V
        model->add_op_matrix_reshape(PV_perm_t, PV_perm_reshape_t); // reshape
        model->add_op_matrix_matrix_mul(PV_perm_reshape_t, Wo_t, O_t);    // PV

        return O_t;

    } catch (const std::exception& e) {
        // 清理资源
        delete Wq_t;
        delete Wk_t;
        delete Wv_t;
        delete Wo_t;
        throw;
    }
    
    return nullptr;
}

Tensor* ModelLoader::loadLayerNormParams(graph_model* model, Tensor* input,
                                       int hidden_dim) {
    LOG_INFO("加载LayerNorm参数");
    
    // 读取gamma向量 (hidden_dim)
    float* gamma_ptr = (float*)malloc(hidden_dim * sizeof(float));
    if (!gamma_ptr) {
        throw std::runtime_error("Failed to allocate memory for gamma");
    }
    
    // 读取beta向量 (hidden_dim)
    float* beta_ptr = (float*)malloc(hidden_dim * sizeof(float));
    if (!beta_ptr) {
        free(gamma_ptr);  // 修正：释放gamma_ptr而不是beta_ptr
        throw std::runtime_error("Failed to allocate memory for beta");
    }
    
    // 设置tensor(rank=1, shape=hidden_dim)
    int shape[2] = {hidden_dim, 1};
    Tensor* gamma_t = new Tensor(1, shape, gamma_ptr);
    gamma_t->setisleaf(true);                          // 参数节点设置为leaf
    readFloatArray(model_file_, gamma_ptr, hidden_dim);

    Tensor* beta_t = new Tensor(1, shape, beta_ptr);
    beta_t->setisleaf(true);
    readFloatArray(model_file_, beta_ptr, hidden_dim);

    // =========== build model ==============
    try {        
        // output: rank = input.rank, shape = input.shape()
        // 申请空间
        int out_rank = input->getrank();
        int out_shape[4];
        size_t out_size = sizeof(float);
        for(int rk = 0; rk < out_rank; ++rk){
            out_shape[rk] = input->shape(rk);
            out_size *= out_shape[rk];
        }
        float* output_ptr = nullptr;
        Tensor* output = new Tensor(out_rank, out_shape, output_ptr);
        // 进行计算
        model->add_op_matrix_layernorm(input, output, gamma_t, beta_t);

        return output;
    } catch (const std::exception& e) {
        // 清理资源
        delete gamma_t;
        delete beta_t;
        throw;
    }
    
    return nullptr;
}

Tensor* ModelLoader::loadFFNParams(graph_model* model, Tensor* input, 
                                int hidden_dim, int expanded_dim) { // relu(input * W1) * W2
    LOG_INFO("加载FFN参数");

    // 读取W1矩阵 (hidden_dim x expanded_dim)
    size_t size = hidden_dim * expanded_dim * sizeof(float);
    float* w1_ptr = (float*)malloc(size);
    if (!w1_ptr) {
        throw std::runtime_error("Failed to allocate memory for W1");
    }
    
    // 读取W2矩阵 (expanded_dim x hidden_dim)
    float* w2_ptr = (float*)malloc(size);
    if (!w2_ptr) {
        free(w1_ptr);
        throw std::runtime_error("Failed to allocate memory for W2");
    }
    
    // 设置tensor(rank=1, shape=hidden_dim)
    int shape[2] = {hidden_dim, expanded_dim};
    Tensor* w1_t = new Tensor(2, shape, w1_ptr);
    readFloatMatrix(model_file_, w1_ptr, hidden_dim, expanded_dim);
    w1_t->setisleaf(true);                          // 参数节点设置为leaf

    shape[0] = expanded_dim;
    shape[1] = hidden_dim;
    Tensor* w2_t = new Tensor(2, shape, w2_ptr);
    readFloatMatrix(model_file_, w2_ptr, expanded_dim, hidden_dim);
    w2_t->setisleaf(true);

    // ========== build model ==========
    try {
        // o1 = input * w1, o2 = relu(o1), o = o2 * w2
        // 申请空间
        shape[0] = input->shape(0);
        shape[1] = expanded_dim;
        size_t size = shape[0] * shape[1] * sizeof(float);
        float* o1_ptr = nullptr;
        Tensor* o1_t = new Tensor(2, shape, o1_ptr);        // o1

        float* o2_ptr = nullptr;
        Tensor* o2_t = new Tensor(2, shape, o2_ptr);        // o2

        shape[0] = input->shape(0);
        shape[1] = hidden_dim;
        size = shape[0] * shape[1] * sizeof(float);
        float* o_ptr = nullptr;
        Tensor* o_t = new Tensor(2, shape, o_ptr);        // o

        // 进行计算
        model->add_op_matrix_matrix_mul(input, w1_t, o1_t);
        // activate
        switch (model_config_.act_func)
        {
        case ActivationType::RELU:
            model->add_op_elementwise_relu(o1_t, o2_t);  // 修正：使用o1_t而不是input
            break;
        
        default:
            free(o1_ptr);
            free(o2_ptr);
            free(o_ptr);
            throw std::runtime_error("目前只实现了Relu激活函数");
            break;
        }
        model->add_op_matrix_matrix_mul(o2_t, w2_t, o_t);

        return o_t;

    } catch (const std::exception& e) {
        // 清理资源
        delete w1_t;
        delete w2_t;
        throw;
    }
    
    return nullptr;
}

void ModelLoader::readFloatArray(std::ifstream& file, float* array, size_t size) {
    // 一次性读取整个数组，减少IO操作
    file.read(reinterpret_cast<char*>(array), size * sizeof(float));
}

void ModelLoader::readFloatMatrix(std::ifstream& file, float* matrix, size_t rows, size_t cols) {
    file.read(reinterpret_cast<char*>(matrix), rows * cols * sizeof(float));
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

    }   // namespace backend
} // namespace dhinference 