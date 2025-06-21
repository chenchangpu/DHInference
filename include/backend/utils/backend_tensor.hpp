#ifndef DHINFERENCE_BACKEND_TENSOR_H
#define DHINFERENCE_BACKEND_TENSOR_H

#include <memory>
#include <cstddef>
#include <stdexcept>

namespace dhinference {
namespace backend{

// backend支持的op
enum class Ops{
    NONE,                   // None
    ELEMENTWISE_ADD,        // 逐元素加法
    ELEMENTWISE_SUB,        // 逐元素减法
    ELEMENTWISE_MUL,        // 逐元素乘法
    ELEMENTWISE_DIV,        // 逐元素除法
    ELEMENTWISE_RELU,       // 激活函数relu
    MATRIX_LAYERNORM,       // 2-D layernorm，其他维度的可以转换为2-D的形式
    MATRIX_RMSNORM,         // 2-D rmsnorm，其他维度的可以转换为2-D的形式
    MATRIX_MATRIX_MUL,      // 2-D matrix-martix multiply(sgemm)，其他维度的可以转化为2-D的形式
    BATCH_MATRIX_MATRIX_MUL,// batched gemm, a_BxMxK, b_BxKxN, c_BxMxN
    MATRIX_VECTOR_MUL,      // matrix-vector multiply(sgemv)
    MATRIX_SOFTMAX,         // 2-D softmax，其他维度的可以转换为2-D的形式
    MATRIX_TRANSPOSE,       // 2-D 矩阵转置 / permutation
    MATRIX_PERMUTATION_102, // 3-D permutation [0, 1, 2] -> [1, 0, 2]
    MATRIX_PERMUTATION_210, // 3-D permutation [0, 1, 2] -> [2, 1, 0]
    MATRIX_PERMUTATION_021, // 3-D permutation [0, 1, 2] -> [0, 2, 1]
    MATRIX_PERMUTATION_120, // 3-D permutation [0, 1, 2] -> [1, 2, 0]
    MATRIX_PERMUTATION_0213,// 4-D permutation 暂时只支持 [0, 1, 2, 3] -> [0, 2, 1, 3] <=> [B, L, H, D] -> [B, H, L, D]
    MATRIX_RESHAPE          // 在不改变data数据排布的情况下，将矩阵的形状reinterpret成另一个形状
};

enum class BackendType{
    CPU,
    CUDA
};

// backend的tensor类
class Tensor {
public:
    // 构造函数
    Tensor(BackendType backend_type = BackendType::CPU);
    Tensor(const int rank, const int* shape, BackendType backend_type = BackendType::CPU);
    Tensor(const int rank, const int* shape, const float init_val, BackendType backend_type = BackendType::CPU);
    Tensor(const int rank, const int* shape, float *data, BackendType backend_type = BackendType::CPU);
    
    // 拷贝/移动构造函数
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    
    // 拷贝/移动赋值运算符
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // 析构函数
    ~Tensor();
    
    // 设置data
    void setdata(float* data);

    // 重设形状
    void reshape(const int* new_shape);
    
    // 获取形状
    int shape(int idx) const;
    
    // 获取元素数量
    size_t size() const;
    
    // 获取数据指针
    float* data();
    const float* data() const;

    // 设置op
    void setrank(int rank);

    // 获得op
    int getrank();
    
    // 设置op
    void setop(Ops op);

    // 获得op
    Ops getop();

    // 获取src数量
    int getnsrc();

    // 获得第idx个src
    Tensor* getsrc(int idx);

    // 设置低idx个src
    void setsrc(int idx, Tensor* t);

    // 获得是否是leaf
    bool getisleaf();

    // 设置是否是leaf
    void setisleaf(bool is_leaf);

    // 获得是否是在pool中
    bool getinpool();

    // 设置是否是leaf
    void setinpool(bool in_pool);

    // 获得是否updated
    bool getupdated();

    // 设置updated是否更新
    void setupdated(bool updated);

    // 获得scale
    float getscale();

    // 设置scale
    void setscale(float scale);

    // 获取后端类型
    BackendType get_backend_type() const;
    
    // 设置后端类型
    void set_backend_type(BackendType backend_type);

    // 设置数据所有权
    void set_owns_data(bool owns_data);
    
    // 获取数据所有权
    bool get_owns_data() const;

    // 根据指针和指针类型设置tensor的值
    void setvalue(float* src_data, BackendType src_type);

private:
    int rank_;      // 最高支持4维的tensor
    int shape_[4];
    int stride_[4];
    float* data_;
    Ops op_;         // Ops::op
    int n_src_;      // 
    Tensor* src_[5]; // 最多支持5个src，op(src) -> this tensor
    bool is_leaf_;    // 是否是leaf节点(parametaers或input)
    bool in_pools_;    // 是否放入了graph的leaf_pools或node_pools
    bool updated_;     // 是否已经遍历计算过
    float scale_;      // 算子融合后的scale
    BackendType backend_type_;  // 后端类型
    bool owns_data_;   // 是否拥有数据所有权

    // 辅助函数：释放内存
    void free_data();
};

}   // namespace backend
} // namespace dhinference

#endif // DHINFERENCE_BACKEND_TENSOR_H 