#include <cstring>
#include <algorithm>
#include "backend/utils/backend_graph_model.hpp"
#include "backend/cuda/cuda_utils.hpp"

namespace dhinference {
namespace backend {

// ====================== graph_model ======================
graph_model::graph_model(){     // 分配tensor指针，并没有分配实际tensor，更没有分配实际data
    leaf_pools = new Tensor*[1024]; // 默认分配1024个tensor指针
    node_pools = new Tensor*[1024]; // 默认分配1024个tensor指针
    n_leafs = 0;                    // n_leafs和n_nodes初始化为0
    n_nodes = 0;
    input = nullptr;
    result = nullptr;               // result初始化为空
}

graph_model::graph_model(int MAX_LEAFS, int MAX_NODES){     // 分配tensor指针，并没有分配实际tensor，更没有分配实际data
    leaf_pools = new Tensor*[MAX_LEAFS];
    node_pools = new Tensor*[MAX_NODES];
    n_leafs = 0;                    // n_leafs和n_nodes初始化为0
    n_nodes = 0;
    input = nullptr;
    result = nullptr;               // result初始化为空
}

graph_model::~graph_model(){
    free_tensor_pools();
    // input和result也包含在leaf_pools和node_pools中，所以不需要单独释放
}

void graph_model::set_input_tensor(Tensor* t){
    input = t;
}

Tensor* graph_model::get_input_tensor(){
    return input;
}

void graph_model::set_result_tesnor(Tensor* t){
    result = t;
}

Tensor* graph_model::get_result_tensor(){
    return result;
}

void graph_model::add_tensor(Tensor* t){
    if(!t->getinpool()){
        t->setinpool(true);
        if(t->getisleaf()){
            leaf_pools[n_leafs++] = t;          // 放入leaf_pools
        }
        else{
            node_pools[n_nodes++] = t;          // 放入node_pools
        }
    }
}

// c = a + b
void graph_model::add_op_elementwise_add(Tensor* a, Tensor* b, Tensor* c){
    c->setop(Ops::ELEMENTWISE_ADD);
    c->setsrc(0, a);
    c->setsrc(1, b);
    add_tensor(a);
    add_tensor(b);
    add_tensor(c);
}

// c = a - b
void graph_model::add_op_elementwise_sub(Tensor* a, Tensor* b, Tensor* c){
    c->setop(Ops::ELEMENTWISE_SUB);
    c->setsrc(0, a);
    c->setsrc(1, b); 
    add_tensor(a);
    add_tensor(b);
    add_tensor(c);
}

// c = a * b
void graph_model::add_op_elementwise_mul(Tensor* a, Tensor* b, Tensor* c){
    c->setop(Ops::ELEMENTWISE_MUL);
    c->setsrc(0, a);
    c->setsrc(1, b); 
    add_tensor(a);
    add_tensor(b);
    add_tensor(c);
}

// c = a / b
void graph_model::add_op_elementwise_div(Tensor* a, Tensor* b, Tensor* c){
    c->setop(Ops::ELEMENTWISE_DIV);
    c->setsrc(0, a);
    c->setsrc(1, b); 
    add_tensor(a);
    add_tensor(b);
    add_tensor(c);
}

// b = relu(a)
void graph_model::add_op_elementwise_relu(Tensor* a, Tensor* b){
    b->setop(Ops::ELEMENTWISE_RELU);
    b->setsrc(0, a);
    add_tensor(a);
    add_tensor(b);
}

// b = layernorm(a, gamma, beta)
void graph_model::add_op_matrix_layernorm(Tensor* a, Tensor* b, Tensor* gamma, Tensor* beta){
    b->setop(Ops::MATRIX_LAYERNORM);
    b->setsrc(0, a);
    b->setsrc(1, gamma);
    b->setsrc(2, beta);
    add_tensor(a);
    add_tensor(b);
    add_tensor(gamma);
    add_tensor(beta);
}

// b = rmsnorm(a)
void graph_model::add_op_matrix_rmsnorm(Tensor* a, Tensor* b, Tensor* gamma){
    b->setop(Ops::MATRIX_RMSNORM);
    b->setsrc(0, a);
    b->setsrc(1, gamma);
    add_tensor(a);
    add_tensor(b);
    add_tensor(gamma);
}

// c = ab
void graph_model::add_op_matrix_matrix_mul(Tensor* a, Tensor* b, Tensor* c){
    c->setop(Ops::MATRIX_MATRIX_MUL);
    c->setsrc(0, a);
    c->setsrc(1, b); 
    add_tensor(a);
    add_tensor(b);
    add_tensor(c);
}

// c = a b
void graph_model::add_op_matrix_vector_mul(Tensor* a, Tensor* b, Tensor* c){
    c->setop(Ops::MATRIX_VECTOR_MUL);
    c->setsrc(0, a);
    c->setsrc(1, b); 
    add_tensor(a);
    add_tensor(b);
    add_tensor(c);
}

// b = softmax(a)
void graph_model::add_op_matrix_softmax(Tensor* a, Tensor* b){
    b->setop(Ops::MATRIX_SOFTMAX);
    b->setsrc(0, a);
    add_tensor(a);
    add_tensor(b);
}

// b = a^T
void graph_model::add_op_matrix_transpose(Tensor* a, Tensor* b){
    b->setop(Ops::MATRIX_TRANSPOSE);
    b->setsrc(0, a);
    add_tensor(a);
    add_tensor(b);
}

// permutation [0, 1, 2] -> [1, 0, 2]
void graph_model::add_op_matrix_permutation_102(Tensor* a, Tensor* b){
    b->setop(Ops::MATRIX_PERMUTATION_102);
    b->setsrc(0, a);
    add_tensor(a);
    add_tensor(b);
}

// permutation [0, 1, 2] -> [2, 1, 0]
void graph_model::add_op_matrix_permutation_210(Tensor* a, Tensor* b){
    b->setop(Ops::MATRIX_PERMUTATION_210);
    b->setsrc(0, a);
    add_tensor(a);
    add_tensor(b);
}

// permutation [0, 1, 2] -> [0, 2, 1]
void graph_model::add_op_matrix_permutation_021(Tensor* a, Tensor* b){
    b->setop(Ops::MATRIX_PERMUTATION_021);
    b->setsrc(0, a);
    add_tensor(a);
    add_tensor(b);
}

// permutation [0, 1, 2, 3] -> [0, 2, 1, 3]
void graph_model::add_op_matrix_permutation_0213(Tensor* a, Tensor* b){
    b->setop(Ops::MATRIX_PERMUTATION_0213);
    b->setsrc(0, a);
    add_tensor(a);
    add_tensor(b);
}

// cpu释放pools内存
void graph_model::free_tensor_pools(){
    for(int i = 0; i < n_leafs; ++i){   //释放leafs
        delete leaf_pools[i];           // 然后释放tensor
        leaf_pools[i] = nullptr;
    }
    for(int i = 0; i < n_nodes; ++i){
        delete node_pools[i];
        node_pools[i] = nullptr;
    }
    delete[] leaf_pools;                // 最后释放tensor pools
    delete[] node_pools;
}

void graph_model::reset_nodes(){
    for(int i = 0; i < n_nodes; ++i){
        node_pools[i]->setupdated(false);
    }
}


// ====================== graph_model_cuda ======================
graph_model_cuda::graph_model_cuda(): graph_model(){
    extra_buff = nullptr;           // extr_buff初始化为null
}

graph_model_cuda::graph_model_cuda(int MAX_LEAFS, int MAX_NODES): graph_model(MAX_LEAFS, MAX_NODES){
    extra_buff = nullptr;           // extr_buff初始化为null
}

void graph_model_cuda::alloc_extra_buff(size_t extr_buff_size){
    if(extr_buff_size > 0){
        extra_buff = (void*)gpu_malloc(extr_buff_size);
    }
}

// 循环node_pool进行计算
void graph_model_cuda::forward(){
    int rest_nodes = n_nodes;
    while(rest_nodes > 0){
        bool have_update = false;
        for(int i = 0; i < n_nodes; ++i){
            Tensor* now_tensor = node_pools[i];
            if(!now_tensor->getupdated()){          // 未更新，则检查是否可更新
                bool is_ready = true;
                for(int j = 0; j < now_tensor->getnsrc(); ++j){     // 检查src是否全为是leaf或已经updated
                    Tensor* src = now_tensor->getsrc(j);
                    if(src->getisleaf() || src->getupdated()){      // leaf 或 updated : 继续检查
                        continue;
                    }
                    else{
                        is_ready = false;                           // 否则，is_ready = false
                        break;
                    }
                }
                if(is_ready){
                    // 进行计算！
                    ///////////////////////////////////////////////////////
                    switch (now_tensor->getop())
                    {
                        // ======================================
                        case Ops::ELEMENTWISE_ADD: {
                            // 调用cuda的launch_elementwise_add
                            Tensor* src0 = now_tensor->getsrc(0);
                            Tensor* src1 = now_tensor->getsrc(1);
                            bool can_implement = true;
                            if(src0->getrank() != src1->getrank() || src0->getrank() != now_tensor->getrank()){
                                can_implement = false;
                            }
                            int n_rank = src0->getrank();
                            if(can_implement){
                                for(int idx = 0; idx < n_rank; ++idx){
                                    if(src0->shape(idx) != src1->shape(idx) || src0->shape(idx) != now_tensor->shape(idx)){
                                        can_implement = false;
                                        break;
                                    }
                                }
                            }
                            if(!can_implement){
                                throw std::runtime_error("Rank or Shape is not compatible, cannot implement elementwise add");
                            }
                            launch_elementwise_add(src0->data(), src1->data(), \
                                                    now_tensor->data(), src0->size());                 
                            break;
                        }

                        // ======================================
                        case Ops::ELEMENTWISE_SUB: {
                            // 调用cuda的launch_elementwise_sub
                            Tensor* src0 = now_tensor->getsrc(0);
                            Tensor* src1 = now_tensor->getsrc(1);
                            bool can_implement = true;
                            if(src0->getrank() != src1->getrank() || src0->getrank() != now_tensor->getrank()){
                                can_implement = false;
                            }
                            int n_rank = src0->getrank();
                            if(can_implement){
                                for(int idx = 0; idx < n_rank; ++idx){
                                    if(src0->shape(idx) != src1->shape(idx) || src0->shape(idx) != now_tensor->shape(idx)){
                                        can_implement = false;
                                        break;
                                    }
                                }
                            }
                            if(!can_implement){
                                throw std::runtime_error("Rank or Shape is not compatible, cannot implement elementwise sub");
                            }
                            launch_elementwise_sub(src0->data(), src1->data(), \
                                                    now_tensor->data(), src0->size());
                            break;
                        }

                        // ======================================
                        case Ops::ELEMENTWISE_MUL:{
                            // 调用cuda的launch_elementwise_mul
                            Tensor* src0 = now_tensor->getsrc(0);
                            Tensor* src1 = now_tensor->getsrc(1);
                            bool can_implement = true;
                            if(src0->getrank() != src1->getrank() || src0->getrank() != now_tensor->getrank()){
                                can_implement = false;
                            }
                            int n_rank = src0->getrank();
                            if(can_implement){
                                for(int idx = 0; idx < n_rank; ++idx){
                                    if(src0->shape(idx) != src1->shape(idx) || src0->shape(idx) != now_tensor->shape(idx)){
                                        can_implement = false;
                                        break;
                                    }
                                }
                            }
                            if(!can_implement){
                                throw std::runtime_error("Rank or Shape is not compatible, cannot implement elementwise mul");
                            }
                            launch_elementwise_mul(src0->data(), now_tensor->data(), \
                                                    now_tensor->data(), src0->size());
                            break;
                        }

                        // ======================================
                        case Ops::ELEMENTWISE_DIV: {
                            // 调用cuda的launch_elementwise_div
                            Tensor* src0 = now_tensor->getsrc(0);
                            Tensor* src1 = now_tensor->getsrc(1);
                            bool can_implement = true;
                            if(src0->getrank() != src1->getrank() || src0->getrank() != now_tensor->getrank()){
                                can_implement = false;
                            }
                            int n_rank = src0->getrank();
                            if(can_implement){
                                for(int idx = 0; idx < n_rank; ++idx){
                                    if(src0->shape(idx) != src1->shape(idx) || src0->shape(idx) != now_tensor->shape(idx)){
                                        can_implement = false;
                                        break;
                                    }
                                }
                            }
                            if(!can_implement){
                                throw std::runtime_error("Rank or Shape is not compatible, cannot implement elementwise div");
                            }
                            launch_elementwise_div(src0->data(), src1->data(), \
                                                    now_tensor->data(), src0->size());
                            break;
                        }

                        // ======================================
                        case Ops::ELEMENTWISE_RELU: {
                            // 调用cuda的launch_elementwise_relu
                            Tensor* src0 = now_tensor->getsrc(0);
                            bool can_implement = true;
                            if(src0->getrank() != now_tensor->getrank()){
                                can_implement = false;
                            }
                            // rank相同
                            int n_rank = src0->getrank();
                            // shape相同
                            if(can_implement){
                                for(int idx = 0; idx < n_rank; ++idx){
                                    if(src0->shape(idx) != now_tensor->shape(idx)){
                                        can_implement = false;
                                        break;
                                    }
                                }
                            }
                            if(!can_implement){
                                throw std::runtime_error("Rank or Shape is not compatible, cannot implement elementwise relu");
                            }
                            launch_elementwise_relu(src0->data(), now_tensor->data(), src0->size());
                            break;
                        }
                        
                        // ======================================
                        case Ops::MATRIX_LAYERNORM: {
                            // 调用cuda的launch_layernorm_128
                            Tensor* src0 = now_tensor->getsrc(0);       // b
                            Tensor* src1 = now_tensor->getsrc(1);       // gamma
                            Tensor* src2 = now_tensor->getsrc(2);       // beta
                            bool can_implement = true;
                            if(src1->getrank() != 1 || src2->getrank() != 1){   // gamma, beta 一维向量
                                can_implement = false;
                            }
                            if(src1->shape(0) != src2->shape(0)){
                                can_implement = false;                          // gamma, beta维度相同
                            }
                            int n_rank = src0->getrank();
                            if(src0->shape(n_rank - 1) != src1->shape(0)){
                                can_implement = false;                          // 最后一维维度相同
                            }
                            if(src0->getrank() != now_tensor->getrank()){       // 输入、输出rank相同
                                can_implement = false;
                            }
                            if(can_implement){                                  // 输入、输出shape相同
                                for(int idx = 0; idx < n_rank; ++idx){
                                    if(src0->shape(idx) != now_tensor->shape(idx)){
                                        can_implement = false;
                                        break;
                                    }
                                }
                            }
                            if(!can_implement){
                                throw std::runtime_error("Rank or Shape is not compatible, cannot implement matrix layernorm");
                            }
                            int n_rows = src0->size() / src0->shape(n_rank - 1);
                            launch_layernorm(src0->data(), now_tensor->data(), \
                                                    src1->data(), src2->data(), n_rows, src0->shape(n_rank - 1));
                            break;
                        }
                        
                        // =======================================
                        case Ops::MATRIX_RMSNORM: {
                                                        // 调用cuda的launch_rmsnorm_128
                            Tensor* src0 = now_tensor->getsrc(0);       // b
                            Tensor* src1 = now_tensor->getsrc(1);       // gamma
                            bool can_implement = true;
                            int n_rank = src0->getrank();
                            if(src0->shape(n_rank - 1) != src1->shape(0)){
                                can_implement = false;                          // 最后一维维度相同
                            }
                            if(src0->getrank() != now_tensor->getrank()){       // 输入、输出rank相同
                                can_implement = false;
                            }
                            if(can_implement){                                  // 输入、输出shape相同
                                for(int idx = 0; idx < n_rank; ++idx){
                                    if(src0->shape(idx) != now_tensor->shape(idx)){
                                        can_implement = false;
                                        break;
                                    }
                                }
                            }
                            if(!can_implement){
                                throw std::runtime_error("Rank or Shape is not compatible, cannot implement matrix rmsnorm");
                            }
                            int n_rows = src0->size() / src0->shape(n_rank - 1);
                            launch_rmsnorm(src0->data(), now_tensor->data(), \
                                                    src1->data(), n_rows, src0->shape(n_rank - 1));
                            break;
                        }
                        
                        // =====================================
                        case Ops::MATRIX_MATRIX_MUL: { // matrix-matrix multiply, 高维转换为2维，不支持batched gemm
                            Tensor* src0 = now_tensor->getsrc(0);   // A[M, K]
                            Tensor* src1 = now_tensor->getsrc(1);   // B[K, N]
                            bool can_implement = true;
                            // 至少2维
                            if(src0->getrank() < 2 || src1->getrank() < 2 || now_tensor->getrank() < 2){
                                can_implement = false;
                            }
                            // e.g. src0 [a,b,c], src1 [c,d,e,f], now_tensor: [a,b,d,e,f] 
                            // <-> [a*b, c] x [c, d*e*f] = [a*b, d*e*f]
                            // 检查rank是否符合
                            if(now_tensor->getrank() != src0->getrank() + src1->getrank() - 2){
                                can_implement = false;
                            }
                            if(can_implement){
                                int n_rank0 = src0->getrank();
                                int n_rank1 = src1->getrank();
                                // c = c
                                if(src0->shape(n_rank0 - 1) != src1->shape(0)) 
                                    can_implement = false;
                                // a,b = a,b
                                for(int rk = 0; rk < n_rank0 - 1; ++rk){
                                    if(src0->shape(rk) != now_tensor->shape(rk)){
                                        can_implement = false; break;
                                    }
                                }
                                // d,e,f = d,e,f
                                for(int rk = 0; rk < n_rank1 - 1; ++rk){
                                    if(src1->shape(1 + rk) != now_tensor->shape(n_rank0 - 1 + rk)){
                                        can_implement = false; break;
                                    }
                                }
                            }
                            if(!can_implement){
                                throw std::runtime_error("Rank or Shape is not compatible, cannot implement gemm");
                            }
                            int M = src0->size() / src0->shape(src0->getrank() - 1);
                            int N = src1->size() / src1->shape(0);
                            launch_sgemm_default(src0->data(), src1->data(), now_tensor->data(), \
                                        M, N, src1->shape(0), now_tensor->getscale());
                            break;
                        }   

                        // =======================================
                        case Ops::MATRIX_VECTOR_MUL: { // matrix-vector mul, 高维可以转换为2维
                            Tensor* src0 = now_tensor->getsrc(0);   // A[M, K]
                            Tensor* src1 = now_tensor->getsrc(1);   // B[K]
                            bool can_implement = true;
                            // A至少2维, B为1维
                            if(src0->getrank() < 2 || src1->getrank() != 1 || now_tensor->getrank() < 1){
                                can_implement = false;
                            }
                            // e.g. src0 [a,b,c], src1 [c], now_tensor: [a,b] 
                            // 检查rank
                            if(now_tensor->getrank() != src0->getrank() - 1)
                                can_implement = false;
                            // 检查shape, c = c, ab = ab
                            if(can_implement){
                                if(src0->shape(src0->getrank() - 1) != src1->shape(0))
                                    can_implement = false;
                                for(int rk = 0; rk < now_tensor->getrank(); ++rk){
                                    if(src0->shape(rk) != now_tensor->shape(rk)){
                                        can_implement = false;
                                        break;
                                    }
                                }
                            }
                            if(!can_implement){
                                throw std::runtime_error("Rank or Shape is not compatible, cannot implement gemv");
                            }
                            int M = src0->size() / src0->shape(src0->getrank() - 1);
                            launch_sgemv_128(src0->data(), src1->data(), now_tensor->data(),\
                                                    M, src1->shape(0));
                            break;
                        }

                        case Ops::MATRIX_SOFTMAX: {
                            // 调用cuda的launch_softmax_128
                            Tensor* src0 = now_tensor->getsrc(0);       //
                            bool can_implement = true;
                            if(src0->getrank() != now_tensor->getrank()){       // 输入、输出rank相同
                                can_implement = false;
                            }
                            int n_rank = src0->getrank();
                            if(can_implement){                                  // 输入、输出shape相同
                                for(int idx = 0; idx < n_rank; ++idx){
                                    if(src0->shape(idx) != now_tensor->shape(idx)){
                                        can_implement = false;
                                        break;
                                    }
                                }
                            }
                            if(!can_implement){
                                throw std::runtime_error("Rank or Shape is not compatible, cannot implement matrix softmax");
                            }
                            int n_rows = src0->size() / src0->shape(n_rank - 1);
                            launch_softmax(src0->data(), now_tensor->data(), \
                                                    n_rows, src0->shape(n_rank - 1));
                            break;
                        }
                        
                        case Ops::MATRIX_TRANSPOSE: {   // 二维矩阵转置
                            Tensor* src0 = now_tensor->getsrc(0);
                            bool can_implement = true;
                            // 输入、输出rank相同，且都是2维
                            if(src0->getrank() != now_tensor->getrank() || src0->getrank() != 2){       
                                can_implement = false;
                            }
                            // 检查shape A[m,n] B[n,m]
                            if(src0->shape(0) != now_tensor->shape(1) || src0->shape(1) != now_tensor->shape(0))
                                can_implement = false;
                            if(!can_implement){
                                throw std::runtime_error("Rank or Shape is not compatible, cannot implement matrix transpose");
                            }
                            // 利用extra_buff分配d_shape和d_perm
                            int h_shape[4] = {src0->shape(0), src0->shape(1), 0, 0};
                            int h_perm[4] = {1, 0, -1, -1};

                            // 调用launch_transpose
                            launch_transpose(src0->data(), now_tensor->data(), \
                                    h_shape, h_perm, extra_buff, 2);
                            break;
                        }
                        
                        case Ops::NONE:
                            break;  // Do nothing

                        default:{
                            throw std::runtime_error("This op is not implemented yet");
                            break;
                        }
                    }
                    cuda_device_sync();                       // 为保证正确性，CPU要和GPU同步
                    have_update = true;          // 更新一个node
                    --rest_nodes;                // rest_nodes - 1
                    now_tensor->setupdated(true);               // 将now_tensor更新
                }
            }
        }
        if(rest_nodes > 0 && have_update == false){             // 剩下的节点无法更新，有环，退出
            throw std::runtime_error("Model Graph has loops");
            break;
        }
    }
}

void graph_model_cuda::move_data_to_gpu(Tensor* t) {
    size_t size = t->size() * sizeof(float);
    float* d_ptr = (float*)gpu_malloc(size);    // 分配显存
    float* h_ptr = t->data();                   // 获取CPU数据指针
    copy_cpu_to_gpu(d_ptr, h_ptr, size);        // CPU复制到GPU
    
    t->setdata(d_ptr);                          // 更改tensor指向GPU显存
    t->set_backend_type(BackendType::CUDA);     // 设置后端类型为CUDA
}

void graph_model_cuda::move_data_to_cpu(Tensor* t){
    size_t size = t->size() * sizeof(float);
    float* h_ptr = (float*)malloc(size);    // 分配显存data空间
    float* d_ptr = t->data();                   
    copy_gpu_to_cpu(h_ptr, d_ptr, size);        // GPU复制到CPU

    t->setdata(h_ptr);                          // 更改tensor的data指向CPU
    t->set_backend_type(BackendType::CPU);     // 设置后端类型为CPU
}

// gpu释放内存
void graph_model_cuda::free_tensor_pools(){
    for(int i = 0; i < n_leafs; ++i){   //释放leafs
        delete leaf_pools[i];           // 然后释放tensor
        leaf_pools[i] = nullptr;
    }
    for(int i = 0; i < n_nodes; ++i){
        // gpu_free(node_pools[i]->data());    //  先释放tensor data，注意用gpu_free
        delete node_pools[i];
        node_pools[i] = nullptr;
    }
    delete[] leaf_pools;                // 最后释放tensor pools
    delete[] node_pools;
}

void graph_model_cuda::to_device() {
    // 将所有leaf节点（模型参数）移动到GPU
    for(int i = 0; i < n_leafs; ++i) {
        if(leaf_pools[i]->get_backend_type() == BackendType::CPU) {
            // input tensor不释放CPU内存，其他leaf节点释放
            // move_data_to_gpu(leaf_pools[i], leaf_pools[i] != input);     // dahu: ?? 报错多次释放内存
            move_data_to_gpu(leaf_pools[i]);
        }
    }
    
    // 为所有node节点分配GPU内存并移动数据
    for(int i = 0; i < n_nodes; ++i) {
        if(node_pools[i]->get_backend_type() == BackendType::CPU) {
            // move_data_to_gpu(node_pools[i], true);  // 中间node节点释放CPU内存
            move_data_to_gpu(node_pools[i]); 
        }
    }
}

}   // end of backend
}   // end of dfinference