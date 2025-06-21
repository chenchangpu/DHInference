#ifndef DHINFERENCE_BACKEND_GRAPH_MODEL_H
#define DHINFERENCE_BACKEND_GRAPH_MODEL_H

#include <memory>
#include <cstddef>
#include <stdexcept>
#include "backend_tensor.hpp"
#include "../cuda/cuda_utils.hpp"

namespace dhinference {
namespace backend{

// 激活函数（ffn层）
enum class ActivationType {
    RELU,
    GELU
};

enum class ModelBackend{
    CUDA,
    OPENMP
};

// graph_model
class graph_model{
public:
    graph_model();
    graph_model(int MAX_LEAFS, int MAX_NODES);  // 分配leaf_pools和node_pools

    virtual ~graph_model();

    // 设置/获得 input tensor
    void set_input_tensor(Tensor* t);
    Tensor* get_input_tensor();

    // 设置/获得 result tensor
    void set_result_tesnor(Tensor* t);
    Tensor* get_result_tensor();

    // c[i] = a[i] + b[i]
    void add_op_elementwise_add(Tensor* a, Tensor* b, Tensor* c);
    // c[i] = a[i] - b[i]
    void add_op_elementwise_sub(Tensor* a, Tensor* b, Tensor* c);   
    // c[i] = a[i] * b[i]
    void add_op_elementwise_mul(Tensor* a, Tensor* b, Tensor* c);
    // c[i] = a[i] / b[i]
    void add_op_elementwise_div(Tensor* a, Tensor* b, Tensor* c);
    // b = relu(a)
    void add_op_elementwise_relu(Tensor* a, Tensor* b);
    // b = layernorm(a)  -  row
    void add_op_matrix_layernorm(Tensor* a, Tensor* b, Tensor* gamma, Tensor* beta);
    // b = rmsnorm(a)    -  row
    void add_op_matrix_rmsnorm(Tensor* a, Tensor* b, Tensor* gamma);
    // c = ab
    void add_op_matrix_matrix_mul(Tensor* a, Tensor* b, Tensor* c);
    // c[i] = a[i]b[i], batched sgemm, i=0,1,...,batchsize-1
    void add_op_batch_matrix_matrix_mul(Tensor* a, Tensor* b, Tensor* c);
    // c = ab
    void add_op_matrix_vector_mul(Tensor* a, Tensor* b, Tensor* c);
    // b = softmax(a)    -  row
    void add_op_matrix_softmax(Tensor* a, Tensor* b);
    // b = a^T
    void add_op_matrix_transpose(Tensor* a, Tensor* b);
    // matrix permutation
    void add_op_matrix_permutation_102(Tensor* a, Tensor* b);
    void add_op_matrix_permutation_210(Tensor* a, Tensor* b);
    void add_op_matrix_permutation_021(Tensor* a, Tensor* b);
    void add_op_matrix_permutation_120(Tensor* a, Tensor* b);
    void add_op_matrix_permutation_0213(Tensor* a, Tensor* b);

    // reshape
    void add_op_matrix_reshape(Tensor* a, Tensor* b);

    // reset nodes, 全部设为 未遍历
    void reset_nodes();

    // 释放tensor pools空间
    void free_tensor_pools();   // 注意：result也会被free

    // 释放单个tensor空间
    // static void free_tensor(Tensor* t); // Tensor类已添加析构函数

    // 遍历model_graph，进行inference前向计算
    virtual void forward() = 0;

    // 将模型参数移动到对应设备
    virtual void to_device() = 0;

protected:
    void add_tensor(Tensor* t);                          // 加入leaf_pools或node_pools
    int n_leafs;                    // 参数叶子节点
    int n_nodes;                    // 中间计算节点
    Tensor** leaf_pools;            // 最多n_leafs个leaf节点
    Tensor** node_pools;            // 最多n_nodes个node节点
    Tensor* input;                  // model的输入tensor, 对于transformer, [seq_len, hidden_dim]
    Tensor* result;                 // model的输出tensor, 对于transformer, [seq_len, hidden_dim]
};

class graph_model_cuda: public graph_model{
public:
    graph_model_cuda();
    graph_model_cuda(int MAX_LEAFS, int MAX_NODES);  // 分配leaf_pools和node_pools
    void forward() override;
    void free_tensor_pools();
    void to_device() override;  // 实现CUDA设备的数据迁移
    // static void free_tensor(Tensor* t);
    void alloc_extra_buff(size_t extr_buff_size);       // 分配额外显存空间，比如transpose的shape
    static void move_data_to_gpu(Tensor* t);        // 把tensor的data移到GPU（并释放cpu的内存）
    static void move_data_to_cpu(Tensor* t);         // 把tensor的data移到CPU

private:
    void* extra_buff;              // 额外显存指针
};

}
}

#endif