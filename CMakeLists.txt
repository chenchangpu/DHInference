cmake_minimum_required(VERSION 3.10)
project(DHInference VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
option(BUILD_OPENMP "Build with CUDA support" OFF)
option(BUILD_CUDA "Build with CUDA support" OFF)
option(BUILD_TESTS "Build tests" ON)

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