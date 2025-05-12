#ifndef DHINFERENCE_MEMORY_H
#define DHINFERENCE_MEMORY_H

#include <cstddef>
#include <vector>
#include <memory>

namespace dhinference {

// 内存分配和管理工具类
class Memory {
public:
    // 分配连续内存块
    static void* allocate(size_t size);

    // 释放内存
    static void free(void* ptr);

    // 创建指定大小的float数组
    static std::vector<float> createFloatArray(size_t size, float init_val = 0.0f);

    // 创建指定维度的2D float数组
    static std::vector<std::vector<float>> create2DFloatArray(size_t rows, size_t cols, float init_val = 0.0f);
};

} // namespace dhinference

#endif // DHINFERENCE_MEMORY_H 