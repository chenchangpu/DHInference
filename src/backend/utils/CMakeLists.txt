# 创建utils库
add_library(backend_utils STATIC
    backend_graph_model.cpp
    backend_logger.cpp
    backend_model_loader.cpp
    backend_tensor.cpp
)

# 设置包含目录
target_include_directories(backend_utils
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)

if(BUILD_CUDA)
    target_link_libraries(backend_utils PRIVATE backend_cuda)
endif()

# 设置编译选项
if(MSVC)
    target_compile_options(backend_utils PRIVATE /W4)
else()
    target_compile_options(backend_utils PRIVATE -Wall -Wextra)
endif()
