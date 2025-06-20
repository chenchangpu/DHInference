cmake_minimum_required(VERSION 3.15)
project(DHInference VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
option(BUILD_OPENMP "Build with CUDA support" OFF)
option(BUILD_CUDA "Build with CUDA support" ON)
option(BUILD_TESTS "Build tests" ON)

option(USE_CUBLAS "Use cublas to implement GEMM" ON)
option(USE_ONEFLOW "Use oneflow CUDA ops" ON)

# 增加宏定义
if(BUILD_OPENMP)
    message(STATUS "OpenMP Support Enabled")
else()
    message(STATUS "OPENMP Support Disabled")
endif()

if(BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA Support Enabled")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    message(STATUS "CUDA Support Disabled")
endif()

if(USE_CUBLAS)
    add_definitions(-DUSE_CUBLAS=1)
endif()

if(USE_ONEFLOW)
    add_definitions(-DUSE_ONEFLOW=1)
endif()

# 定义头文件包含路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加子目录
add_subdirectory(src)

# 构建测试
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 示例
add_subdirectory(examples) 