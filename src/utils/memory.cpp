#include "../../include/utils/memory.h"
#include <cstdlib>
#include <stdexcept>

namespace dhinference {

void* Memory::allocate(size_t size) {
    if (size == 0) return nullptr;
    
    void* ptr = std::malloc(size);
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    return ptr;
}

void Memory::free(void* ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

// dahu: 效率有待提高
std::vector<float> Memory::createFloatArray(size_t size, float init_val) {
    return std::vector<float>(size, init_val);
}

// dahu: 效率有待提高
std::vector<std::vector<float>> Memory::create2DFloatArray(size_t rows, size_t cols, float init_val) {
    return std::vector<std::vector<float>>(rows, std::vector<float>(cols, init_val));
}

} // namespace dhinference 