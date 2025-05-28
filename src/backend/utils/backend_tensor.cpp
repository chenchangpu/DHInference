#include "backend/utils/backend_tensor.hpp"
#include "backend/cuda/cuda_utils.hpp"
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

void Tensor::free_data() {
    if (data_ != nullptr && owns_data_) {  // 只有拥有所有权时才释放
        if (backend_type_ == BackendType::CUDA) {
            gpu_free(data_);
        } else {
            free(data_);
        }
        data_ = nullptr;
    }
}

Tensor::Tensor(BackendType backend_type) 
    : rank_(0), data_(nullptr), backend_type_(backend_type), owns_data_(true) {
    is_leaf_ = false;
    in_pools_ = false;
    updated_ = false;
    scale_ = 1.0f;
    std::fill(shape_, shape_ + 4, 0);
    std::fill(stride_, stride_ + 4, 0);
}

Tensor::Tensor(const int rank, const int* shape, BackendType backend_type)
    : rank_(rank), backend_type_(backend_type), owns_data_(true) {
    is_leaf_ = false;
    in_pools_ = false;
    updated_ = false;
    scale_ = 1.0f;
    if (rank > 4) {
        throw std::runtime_error("Tensor rank exceeds maximum supported rank (4)");
    }
    
    std::copy(shape, shape + rank, shape_);
    compute_stride(rank, shape_, stride_);
    data_ = nullptr;
}

Tensor::Tensor(const int rank, const int* shape, const float init_val, BackendType backend_type)
    : Tensor(rank, shape, backend_type) {
    if (backend_type == BackendType::CUDA) {
        throw std::runtime_error("Cannot directly initialize CUDA tensor with value. Please create a CPU tensor first, then use setvalue() or set_backend_type().");
    }
    owns_data_ = true;
    size_t total_size = compute_size(rank, shape_);
    data_ = static_cast<float*>(malloc(total_size * sizeof(float)));
    std::fill(data_, data_ + total_size, init_val);
}

Tensor::Tensor(const int rank, const int* shape, float* data, BackendType backend_type)
    : Tensor(rank, shape, backend_type) {
    data_ = data;
    owns_data_ = false;  // 使用外部数据时不拥有所有权
}

Tensor::~Tensor() {
    free_data();
}

Tensor::Tensor(const Tensor& other) 
    : rank_(other.rank_), backend_type_(other.backend_type_), owns_data_(true) {
    is_leaf_ = other.is_leaf_;               
    in_pools_ = other.in_pools_;          
    updated_ = other.updated_;   
    scale_ = other.scale_;

    std::copy(other.shape_, other.shape_ + 4, shape_);
    std::copy(other.stride_, other.stride_ + 4, stride_);
    
    size_t total_size = compute_size(rank_, shape_);
    if (backend_type_ == BackendType::CUDA) {
        data_ = static_cast<float*>(gpu_malloc(total_size * sizeof(float)));
        copy_gpu_to_gpu(data_, other.data_, total_size * sizeof(float));
    } else {
        data_ = static_cast<float*>(malloc(total_size * sizeof(float)));
        std::memcpy(data_, other.data_, total_size * sizeof(float));
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : rank_(other.rank_)
    , data_(other.data_)
    , backend_type_(other.backend_type_)
    , owns_data_(other.owns_data_) {
    is_leaf_ = other.is_leaf_;
    in_pools_ = other.in_pools_;
    updated_ = other.updated_;
    scale_ = other.scale_;
    std::copy(other.shape_, other.shape_ + 4, shape_);
    std::copy(other.stride_, other.stride_ + 4, stride_);
    
    other.rank_ = 0;
    other.data_ = nullptr;
    other.owns_data_ = false;
    std::fill(other.shape_, other.shape_ + 4, 0);
    std::fill(other.stride_, other.stride_ + 4, 0);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        free_data();
        
        rank_ = other.rank_;
        backend_type_ = other.backend_type_;
        updated_ = other.updated_;
        is_leaf_ = other.is_leaf_;
        in_pools_ = other.in_pools_;
        scale_ = other.scale_;
        owns_data_ = true;
        std::copy(other.shape_, other.shape_ + 4, shape_);
        std::copy(other.stride_, other.stride_ + 4, stride_);
        
        size_t total_size = compute_size(rank_, shape_);
        if (backend_type_ == BackendType::CUDA) {
            data_ = static_cast<float*>(gpu_malloc(total_size * sizeof(float)));
            copy_gpu_to_gpu(data_, other.data_, total_size * sizeof(float));
        } else {
            data_ = static_cast<float*>(malloc(total_size * sizeof(float)));
            std::memcpy(data_, other.data_, total_size * sizeof(float));
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free_data();
        
        rank_ = other.rank_;
        backend_type_ = other.backend_type_;
        updated_ = other.updated_;
        is_leaf_ = other.is_leaf_;
        in_pools_ = other.in_pools_;
        scale_ = other.scale_;
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        std::copy(other.shape_, other.shape_ + 4, shape_);
        std::copy(other.stride_, other.stride_ + 4, stride_);
        
        other.rank_ = 0;
        other.data_ = nullptr;
        other.owns_data_ = false;
        std::fill(other.shape_, other.shape_ + 4, 0);
        std::fill(other.stride_, other.stride_ + 4, 0);
    }
    return *this;
}

void Tensor::setdata(float* data) {
    free_data();    // 拥有权才能释放
    data_ = data;
}

BackendType Tensor::get_backend_type() const {
    return backend_type_;
}

void Tensor::set_backend_type(BackendType new_backend_type) {
    backend_type_ = new_backend_type;
}

void Tensor::set_owns_data(bool owns_data) {
    owns_data_ = owns_data;
}

bool Tensor::get_owns_data() const {
    return owns_data_;
}

void Tensor::setvalue(float* src_data, BackendType src_type) {
    if (src_data == nullptr) {
        throw std::runtime_error("Source data pointer is null");
    }

    size_t total_size = compute_size(rank_, shape_);
    if (total_size == 0) {
        throw std::runtime_error("Tensor has zero size");
    }

    // 根据源数据和目标数据的类型选择合适的拷贝方式
    if (backend_type_ == BackendType::CUDA) {
        if (src_type == BackendType::CUDA) {
            // GPU -> GPU
            copy_gpu_to_gpu(data_, src_data, total_size * sizeof(float));
        } else {
            // CPU -> GPU
            copy_cpu_to_gpu(data_, src_data, total_size * sizeof(float));
        }
    } else {
        if (src_type == BackendType::CUDA) {
            // GPU -> CPU
            copy_gpu_to_cpu(data_, src_data, total_size * sizeof(float));
        } else {
            // CPU -> CPU
            std::memcpy(data_, src_data, total_size * sizeof(float));
        }
    }
}

}   // namespace backend
} // namespace dhinference
