set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加测试子目录
add_subdirectory(unit_tests)
add_subdirectory(integration_tests)

# 添加CTest支持
include(CTest)
enable_testing() 