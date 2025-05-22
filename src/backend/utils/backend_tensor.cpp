#include "backend/utils/backend_tensor.hpp"
#include <cstring>
#include <algorithm>

namespace dhinference {
namespace backend {
// 辅助函数：计算stride
static void compute_stride(const int rank, const int* shape, int* stride) {
    stride[0] = 1;
    for (int i = 1; i < rank; ++i) {
        stride[i] = stride[i - 1] * shape[i - 1];
    }
}

// 辅助函数：计算总大小
static size_t compute_size(const int rank, const int* shape) {
    size_t size = 1;
    for (int i = 0; i < rank; ++i) {
        size *= shape[i];
    }
    return size;
}

Tensor::Tensor() : rank_(0), data_(nullptr) {
    is_leaf_ = false;
    in_pools_ = false;
    updated_ = false;
    scale_ = 1.0f;
    std::fill(shape_, shape_ + 4, 0);
    std::fill(stride_, stride_ + 4, 0);
}

Tensor::Tensor(const int rank, const int* shape)
    : rank_(rank) {
    is_leaf_ = false;
    in_pools_ = false;
    updated_ = false;
    scale_ = 1.0f;
    if (rank > 4) {
        throw std::runtime_error("Tensor rank exceeds maximum supported rank (4)");
    }
    
    // 复制shape
    std::copy(shape, shape + rank, shape_);
    
    // 默认行主序计算stride
    compute_stride(rank, shape_, stride_);
    
    // 初始data_ = null
    data_ = nullptr;

}

Tensor::Tensor(const int rank, const int* shape, const float init_val)
    : Tensor(rank, shape) {
    updated_ = false;
    scale_ = 1.0f;
    size_t total_size = compute_size(rank, shape_);
    data_ = new float[total_size];
    std::fill(data_, data_ + total_size, static_cast<float>(init_val));
}

Tensor::Tensor(const int rank, const int* shape, float* data)
    : Tensor(rank, shape) {
        updated_ = false;
        scale_ = 1.0f;
        data_ = data;
}

Tensor::Tensor(const Tensor& other) : rank_(other.rank_) {
    is_leaf_ = other.is_leaf_;               
    in_pools_ = other.in_pools_;          
    updated_ = other.updated_;   
    scale_ = other.scale_;

    std::copy(other.shape_, other.shape_ + 4, shape_);
    std::copy(other.stride_, other.stride_ + 4, stride_);
    
    size_t total_size = compute_size(rank_, shape_);
    data_ = new float[total_size];
    std::copy(other.data_, other.data_ + total_size, data_);    // 只能复制CPU的data！
}

Tensor::Tensor(Tensor&& other) noexcept
    : rank_(other.rank_)
    , data_(other.data_) {
    is_leaf_ = other.is_leaf_;
    in_pools_ = other.in_pools_;
    updated_ = other.updated_;
    scale_ = other.scale_;
    std::copy(other.shape_, other.shape_ + 4, shape_);
    std::copy(other.stride_, other.stride_ + 4, stride_);
    
    // delete other
    other.rank_ = 0;
    other.data_ = nullptr;
    std::fill(other.shape_, other.shape_ + 4, 0);
    std::fill(other.stride_, other.stride_ + 4, 0);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        delete[] data_;
        
        rank_ = other.rank_;
        updated_ = other.updated_;
        is_leaf_ = other.is_leaf_;
        in_pools_ = other.in_pools_;
        scale_ = other.scale_;
        std::copy(other.shape_, other.shape_ + 4, shape_);
        std::copy(other.stride_, other.stride_ + 4, stride_);
        
        size_t total_size = compute_size(rank_, shape_);
        data_ = new float[total_size];
        std::copy(other.data_, other.data_ + total_size, data_);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] data_;
        
        rank_ = other.rank_;
        updated_ = other.updated_;
        is_leaf_ = other.is_leaf_;
        in_pools_ = other.in_pools_;
        scale_ = other.scale_;
        std::copy(other.shape_, other.shape_ + 4, shape_);
        std::copy(other.stride_, other.stride_ + 4, stride_);
        data_ = other.data_;
        
        other.rank_ = 0;
        other.data_ = nullptr;
        std::fill(other.shape_, other.shape_ + 4, 0);
        std::fill(other.stride_, other.stride_ + 4, 0);
    }
    return *this;
}

void Tensor::setdata(float* data) {
    delete[] data_;
    data_ = data;
}

void Tensor::reshape(const int* new_shape) {
    size_t new_size = 1;
    for (int i = 0; i < rank_; ++i) {
        new_size *= new_shape[i];
    }
    
    if (new_size != compute_size(rank_, shape_)) {
        throw std::runtime_error("New shape must have the same total size as the original shape");
    }
    
    std::copy(new_shape, new_shape + rank_, shape_);
    compute_stride(rank_, shape_, stride_);
}

int Tensor::shape(int idx) const {
    if(idx < rank_)
        return shape_[idx];
    throw std::runtime_error("Idx exceeds this tensor's rank");
}

size_t Tensor::size() const {
    return compute_size(rank_, shape_);
}

float* Tensor::data() {
    return data_;
}

const float* Tensor::data() const {
    return data_;
}

void Tensor::setrank(int rank){
    rank_ = rank;
}

int Tensor::getrank(){
    return rank_;
}

void Tensor::setop(Ops op){
    op_ = op;
    switch (op_)
    {
    case Ops::NONE:                 n_src_ = 2;      break;
    case Ops::ELEMENTWISE_ADD:      n_src_ = 2;      break;
    case Ops::ELEMENTWISE_SUB:      n_src_ = 2;      break;
    case Ops::ELEMENTWISE_MUL:      n_src_ = 2;      break;
    case Ops::ELEMENTWISE_DIV:      n_src_ = 2;      break;
    case Ops::ELEMENTWISE_RELU:     n_src_ = 1;      break;
    case Ops::MATRIX_LAYERNORM:     n_src_ = 3;      break;             // b = layernorm(a, gamma, beta);
    case Ops::MATRIX_RMSNORM:       n_src_ = 2;      break;             // b = rmsnorm(a, gamma);
    case Ops::MATRIX_MATRIX_MUL:    n_src_ = 2;      break;
    case Ops::MATRIX_VECTOR_MUL:    n_src_ = 2;      break;
    case Ops::MATRIX_SOFTMAX:       n_src_ = 1;      break;
    case Ops::MATRIX_TRANSPOSE:     n_src_ = 1;      break;
    
    default:
        n_src_ = 1;     // other MATRIX_PERMUTATION ops
        break;
    }

    // src初始化为null
    for(int i = 0; i < n_src_; ++i){
        src_[i] = nullptr;
    }
}

Ops Tensor::getop(){
    return op_;
}

int Tensor::getnsrc(){
    return n_src_;
};

Tensor* Tensor::getsrc(int idx){
    return src_[idx];
}

void Tensor::setsrc(int idx, Tensor* t){
    src_[idx] = t;
}

bool Tensor::getisleaf(){
    return is_leaf_;
}

void Tensor::setisleaf(bool is_leaf){
    is_leaf_ = is_leaf;
}

bool Tensor::getinpool(){
    return in_pools_;
}

void Tensor::setinpool(bool in_pool){
    in_pools_ = in_pool;
}

bool Tensor::getupdated(){
    return updated_;
};

void Tensor::setupdated(bool updated){
    updated_ = updated;
}

float Tensor::getscale(){
    return scale_;
}

void Tensor::setscale(float scale){
    scale_ = scale;
}


}   // namespace backend
} // namespace dhinference
