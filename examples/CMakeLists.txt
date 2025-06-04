set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加定义
add_definitions(-DCMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}")

# 模型加载测试示例
add_executable(model_loader_test model_loader_test.cpp)
add_executable(model_inference_test model_inference_test.cpp)
add_executable(tensor_memory_test tensor_memory_test.cpp)

if(BUILD_CUDA)
    add_executable(model_backend_cuda_test model_backend_cuda_test.cpp)
endif()

# 链接库
target_link_libraries(model_loader_test
    PRIVATE
    dhinference
)

target_link_libraries(model_inference_test
    PRIVATE
    dhinference
)

target_link_libraries(tensor_memory_test
    PRIVATE
    dhinference
)

if(BUILD_CUDA)
    target_link_libraries(model_backend_cuda_test
        PRIVATE
        dhinference
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
    )
    
    target_link_libraries(tensor_memory_test
        PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
    )
endif()

# 设置输出目录
set_target_properties(model_loader_test model_inference_test tensor_memory_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(BUILD_CUDA)
    set_target_properties(model_backend_cuda_test
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# 安装
install(TARGETS 
        model_loader_test 
        model_inference_test 
        tensor_memory_test
        RUNTIME DESTINATION bin)

if(BUILD_CUDA)
    install(TARGETS model_backend_cuda_test
            RUNTIME DESTINATION bin)
endif()

# 安装Python测试文件
install(FILES
    model_pytorch_test.py
    model_triton_test.py
    DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
) 
