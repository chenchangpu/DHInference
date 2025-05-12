#ifndef DHINFERENCE_TENSOR_H
#define DHINFERENCE_TENSOR_H

#include <vector>
#include <memory>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

namespace dhinference {

// 简单的张量类，支持多维数组操作
class Tensor {
public:
    // 构造函数
    Tensor();
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, float init_val);
    Tensor(const float *data, const std::vector<size_t>& shape);
    
    // 拷贝/移动构造函数
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    
    // 拷贝/移动赋值运算符
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // 析构函数
    ~Tensor() = default;
    
    // 重设形状
    void reshape(const std::vector<size_t>& new_shape);
    
    // 获取形状
    const std::vector<size_t>& shape() const;
    
    // 获取元素数量
    size_t size() const;
    
    // 获取数据指针
    float* data();
    const float* data() const;
    
    // 索引操作
    float& at(const std::initializer_list<size_t>& indices);
    const float& at(const std::initializer_list<size_t>& indices) const;
    
    // 运算符重载
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const; // 逐元素相乘
    
    // 与标量相乘
    Tensor operator*(float scalar) const;
    friend Tensor operator*(float scalar, const Tensor& tensor);
    
    // 矩阵乘法
    Tensor matmul(const Tensor& other) const;
    
    // 矩阵转置
    Tensor transpose() const;

private:
    std::vector<size_t> shape_;
    std::vector<float> data_;
    
    // 计算线性索引
    size_t compute_index(const std::initializer_list<size_t>& indices) const;
};

// 标量左乘张量
Tensor operator*(float scalar, const Tensor& tensor);

} // namespace dhinference

#endif // DHINFERENCE_TENSOR_H 