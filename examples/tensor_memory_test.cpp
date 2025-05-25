#include "backend/utils/backend_tensor.hpp"
#include "backend/cuda/cuda_utils.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace dhinference::backend;

void test_cpu_tensor() {
    std::cout << "\n=== 测试CPU Tensor ===\n";
    
    // 测试基本构造和析构
    {
        std::cout << "测试1: 基本构造和析构" << std::endl;
        int shape[2] = {2, 3};
        Tensor t(2, shape, 1.0f, BackendType::CPU);
        float* data = t.data();
        // 验证内存是否正确初始化
        bool all_ones = true;
        for(size_t i = 0; i < 6; ++i) {
            if(data[i] != 1.0f) {
                all_ones = false;
                break;
            }
        }
        std::cout << "内存初始化测试: " << (all_ones ? "通过" : "失败") << std::endl;
        std::cout << "数据所有权测试: " << (t.get_owns_data() ? "通过" : "失败") << std::endl;
    }
    std::cout << "测试1：张量创建释放成功" << std::endl;

    // 测试使用外部分配的内存
    {
        std::cout << "\n测试2: 使用外部malloc分配的内存" << std::endl;
        int shape[2] = {2, 3};
        float* external_data = static_cast<float*>(malloc(6 * sizeof(float)));
        for(int i = 0; i < 6; ++i) {
            external_data[i] = i;
        }
        
        {
            Tensor t(2, shape, external_data, BackendType::CPU);
            float* tensor_data = t.data();
            bool data_correct = true;
            for(int i = 0; i < 6; ++i) {
                if(tensor_data[i] != i) {
                    data_correct = false;
                    break;
                }
            }
            std::cout << "外部内存测试: " << (data_correct ? "通过" : "失败") << std::endl;
            std::cout << "数据所有权测试: " << (!t.get_owns_data() ? "通过" : "失败") << std::endl;
        }  // Tensor析构，但不会释放external_data
        
        // 验证external_data仍然可用
        bool data_valid = true;
        for(int i = 0; i < 6; ++i) {
            if(external_data[i] != i) {
                data_valid = false;
                break;
            }
        }
        std::cout << "外部内存保持完整性: " << (data_valid ? "通过" : "失败") << std::endl;
        
        free(external_data);  // 现在安全地释放外部内存
    }
    std::cout << "测试2：外部内存测试成功" << std::endl;

    // 测试拷贝构造
    {
        std::cout << "\n测试3: 拷贝构造" << std::endl;
        int shape[2] = {2, 3};
        Tensor t1(2, shape, 1.0f, BackendType::CPU);
        Tensor t2(t1);
        // 验证内存是独立的
        float* data1 = t1.data();
        float* data2 = t2.data();
        data1[0] = 2.0f;
        bool memory_independent = (data2[0] == 1.0f);
        std::cout << "内存独立性测试: " << (memory_independent ? "通过" : "失败") << std::endl;
    }
    std::cout << "测试3：拷贝构造成功" << std::endl;
    // 测试移动构造
    {
        std::cout << "\n测试4: 移动构造" << std::endl;
        int shape[2] = {2, 3};
        Tensor t1(2, shape, 1.0f, BackendType::CPU);
        float* original_data = t1.data();
        Tensor t2(std::move(t1));
        bool moved_correctly = (t2.data() == original_data && t1.data() == nullptr);
        std::cout << "移动构造测试: " << (moved_correctly ? "通过" : "失败") << std::endl;
    }
    std::cout << "测试4：移动构造成功" << std::endl;
    // 测试赋值操作
    {
        std::cout << "\n测试5: 赋值操作" << std::endl;
        int shape[2] = {2, 3};
        Tensor t1(2, shape, 1.0f, BackendType::CPU);
        Tensor t2 = t1;
        float* data1 = t1.data();
        float* data2 = t2.data();
        data1[0] = 2.0f;
        bool assignment_independent = (data2[0] == 1.0f);
        std::cout << "赋值操作内存独立性测试: " << (assignment_independent ? "通过" : "失败") << std::endl;
    }
    std::cout << "测试5：赋值操作成功" << std::endl;

    // 测试setvalue函数
    {
        std::cout << "\n测试6: setvalue功能" << std::endl;
        int shape[2] = {2, 3};
        Tensor t(2, shape, 0.0f, BackendType::CPU);  // 初始化为0
        
        // 测试CPU->CPU
        float cpu_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        t.setvalue(cpu_data, BackendType::CPU);
        float* tensor_data = t.data();
        bool cpu_to_cpu_correct = true;
        for(int i = 0; i < 6; ++i) {
            if(abs(tensor_data[i] - cpu_data[i]) > 1e-6) {
                cpu_to_cpu_correct = false;
                break;
            }
        }
        std::cout << "CPU->CPU数据拷贝测试: " << (cpu_to_cpu_correct ? "通过" : "失败") << std::endl;
    }
    std::cout << "测试6：setvalue功能测试成功" << std::endl;
}

void test_cuda_tensor() {
    std::cout << "\n=== 测试CUDA Tensor ===\n";
    
    try {
        // 测试基本构造和析构
        {
            std::cout << "测试1: 基本构造和析构" << std::endl;
            int shape[2] = {2, 3};
            // 先创建CPU tensor并初始化
            Tensor cpu_tensor(2, shape, 1.0f, BackendType::CPU);
            // 转换为CUDA tensor
            cpu_tensor.set_backend_type(BackendType::CUDA);
        }
        std::cout << "测试1：CUDA张量创建释放成功" << std::endl;

        // 测试拷贝构造
        {
            std::cout << "\n测试2: 拷贝构造" << std::endl;
            int shape[2] = {2, 3};
            Tensor cpu_tensor(2, shape, 1.0f, BackendType::CPU);
            cpu_tensor.set_backend_type(BackendType::CUDA);
            Tensor t2(cpu_tensor);
        }
        std::cout << "测试2：CUDA张量拷贝构造成功" << std::endl;

        // 测试移动构造
        {
            std::cout << "\n测试3: 移动构造" << std::endl;
            int shape[2] = {2, 3};
            Tensor cpu_tensor(2, shape, 1.0f, BackendType::CPU);
            cpu_tensor.set_backend_type(BackendType::CUDA);
            Tensor t2(std::move(cpu_tensor));
        }
        std::cout << "测试3：CUDA张量移动构造成功" << std::endl;

        // 测试赋值操作
        {
            std::cout << "\n测试4: 赋值操作" << std::endl;
            int shape[2] = {2, 3};
            Tensor cpu_tensor(2, shape, 1.0f, BackendType::CPU);
            cpu_tensor.set_backend_type(BackendType::CUDA);
            Tensor t2 = cpu_tensor;
        }
        std::cout << "测试4：CUDA张量赋值操作成功" << std::endl;

        // 测试CPU和CUDA tensor之间的转换
        {
            std::cout << "\n测试5: CPU和CUDA tensor转换" << std::endl;
            int shape[2] = {2, 3};
            Tensor cpu_tensor(2, shape, 1.0f, BackendType::CPU);
            Tensor cuda_tensor(cpu_tensor);
            cuda_tensor.set_backend_type(BackendType::CUDA);
        }
        std::cout << "测试5：CPU和CUDA张量转换成功" << std::endl;

        // 测试setvalue函数
        {
            std::cout << "\n测试6: setvalue功能" << std::endl;
            int shape[2] = {2, 3};
            // 创建一个CPU tensor并初始化为0
            Tensor cpu_tensor(2, shape, 0.0f, BackendType::CPU);
            // 转换为CUDA tensor
            cpu_tensor.set_backend_type(BackendType::CUDA);
            Tensor& cuda_tensor = cpu_tensor;
            
            // 准备CPU数据
            float cpu_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            cuda_tensor.setvalue(cpu_data, BackendType::CPU);  // CPU->GPU
            
            // 验证数据
            float verify_data[6];
            copy_gpu_to_cpu(verify_data, cuda_tensor.data(), 6 * sizeof(float));
            bool cpu_to_gpu_correct = true;
            for(int i = 0; i < 6; ++i) {
                if(abs(verify_data[i] - cpu_data[i]) > 1e-6) {
                    cpu_to_gpu_correct = false;
                    break;
                }
            }
            std::cout << "CPU->GPU数据拷贝测试: " << (cpu_to_gpu_correct ? "通过" : "失败") << std::endl;

            // 准备另一个CUDA tensor用于GPU->GPU测试
            Tensor temp_cpu_tensor(2, shape, 7.0f, BackendType::CPU);
            temp_cpu_tensor.set_backend_type(BackendType::CUDA);
            cuda_tensor.setvalue(temp_cpu_tensor.data(), BackendType::CUDA);  // GPU->GPU
            
            // 验证数据
            copy_gpu_to_cpu(verify_data, cuda_tensor.data(), 6 * sizeof(float));
            bool gpu_to_gpu_correct = true;
            for(int i = 0; i < 6; ++i) {
                if(abs(verify_data[i] - 7.0f) > 1e-6) {
                    gpu_to_gpu_correct = false;
                    break;
                }
            }
            std::cout << "GPU->GPU数据拷贝测试: " << (gpu_to_gpu_correct ? "通过" : "失败") << std::endl;
        }
        std::cout << "测试6：CUDA setvalue功能测试成功" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CUDA test failed: " << e.what() << std::endl;
    }
}

int main() {
    try {
        test_cpu_tensor();
        test_cuda_tensor();
        std::cout << "\n所有测试完成！" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        return 1;
    }
    return 0;
} 