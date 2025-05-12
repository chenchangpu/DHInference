#include "../../include/utils/tensor.h"
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <cstring>      // memcpy

namespace dhinference {

Tensor::Tensor() : shape_(), data_() {}

Tensor::Tensor(const std::vector<size_t>& shape)
    : shape_(shape) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 
                                      size_t(1), std::multiplies<size_t>());
    data_.resize(total_size, 0.0f);
}

Tensor::Tensor(const std::vector<size_t>& shape, float init_val)
    : shape_(shape) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 
                                      size_t(1), std::multiplies<size_t>());
    data_.resize(total_size, init_val);
}

Tensor::Tensor(const float *data, const std::vector<size_t>& shape)
    : shape_(shape) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 
                                      size_t(1), std::multiplies<size_t>());
    data_.resize(total_size);

    std::memcpy(data_.data(), data, total_size * sizeof(float));
}

// dahu: 浅拷贝
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), data_(other.data_) {}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), data_(std::move(other.data_)) {}

// dahu: 浅拷贝
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        data_ = other.data_;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        data_ = std::move(other.data_);
    }
    return *this;
}

void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 
                                     size_t(1), std::multiplies<size_t>());
    if (new_size != data_.size()) {
        throw std::invalid_argument("新形状的元素数量必须与原形状相同");
    }
    shape_ = new_shape;
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

size_t Tensor::size() const {
    return data_.size();
}

float* Tensor::data() {
    return data_.data();
}

const float* Tensor::data() const {
    return data_.data();
}

float& Tensor::at(const std::initializer_list<size_t>& indices) {
    return data_[compute_index(indices)];
}

const float& Tensor::at(const std::initializer_list<size_t>& indices) const {
    return data_[compute_index(indices)];
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("张量形状不匹配");
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("张量形状不匹配");
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("张量形状不匹配");
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    // 简化为2D矩阵乘法
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("矩阵乘法要求两个2D张量");
    }
    
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("矩阵维度不兼容: (" + 
                                  std::to_string(shape_[0]) + "," + std::to_string(shape_[1]) + ") x (" + 
                                  std::to_string(other.shape_[0]) + "," + std::to_string(other.shape_[1]) + ")");
    }
    
    Tensor result({shape_[0], other.shape_[1]}, 0.0f);
    
    // 朴素的矩阵乘法实现
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < other.shape_[1]; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < shape_[1]; ++k) {
                sum += data_[i * shape_[1] + k] * other.data_[k * other.shape_[1] + j];
            }
            result.data_[i * other.shape_[1] + j] = sum;
        }
    }
    
    return result;
}

size_t Tensor::compute_index(const std::initializer_list<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("索引维度与张量维度不匹配");
    }
    
    size_t flat_index = 0;
    size_t stride = 1;
    
    auto shape_it = shape_.rbegin();
    auto indices_it = std::rbegin(indices);
    
    for (; shape_it != shape_.rend() && indices_it != std::rend(indices); ++shape_it, ++indices_it) {
        if (*indices_it >= *shape_it) {
            throw std::out_of_range("索引超出范围");
        }
        flat_index += (*indices_it) * stride;
        stride *= (*shape_it);
    }
    
    return flat_index;
}

Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("转置操作只支持2维张量");
    }
    
    // 创建新的形状
    std::vector<size_t> new_shape = {shape_[1], shape_[0]};
    Tensor result(new_shape);
    
    // 执行转置
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            result.at({j, i}) = at({i, j});
        }
    }
    
    return result;
}

} // namespace dhinference 