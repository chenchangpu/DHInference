set(CUDA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/backend/cuda)

# 设置CUDA架构
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89)
find_package(CUDAToolkit REQUIRED)

# 创建CUDA库
add_library(backend_cuda SHARED
    elementwise.cu
    elementwise_oneflow.cu
    sgemm.cu
    sgemm_cublas.cu
    cuda_memory.cu
    sgemv.cu
    transpose.cu
    softmax.cu
    softmax_oneflow.cu
    rmsnorm.cu
    layernorm.cu
    layernorm_oneflow.cu
)

# 设置CUDA编译选项
set_target_properties(backend_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "backend_cuda"
)

# 设置包含目录
target_include_directories(backend_cuda
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 链接CUDA库
target_link_libraries(backend_cuda
    PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
)

# 设置编译选项
if(MSVC)
    target_compile_options(backend_cuda PRIVATE /W4)
else()
    target_compile_options(backend_cuda PRIVATE -Wall -Wextra)
endif()

# 设置安装规则
install(TARGETS backend_cuda
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)