set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 模型加载测试示例
add_executable(model_loader_test model_loader_test.cpp)
target_link_libraries(model_loader_test PRIVATE dhinference)

add_executable(model_inference_test model_inference_test.cpp)
target_link_libraries(model_inference_test PRIVATE dhinference)

# 安装
install(TARGETS model_loader_test model_inference_test
        RUNTIME DESTINATION bin)

# 设置包含目录
target_include_directories(model_loader_test
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
) 