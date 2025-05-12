# 添加子目录
add_subdirectory(model)
add_subdirectory(ops)
add_subdirectory(utils)
add_subdirectory(backend)

# 创建主库
set(LIBRARY_NAME dhinference)

# 创建接口库（不直接包含源文件，仅聚合其他库）
add_library(${LIBRARY_NAME} INTERFACE)

# 将子目录的源文件链接到主库
target_link_libraries(${LIBRARY_NAME}
    INTERFACE
    model
    ops
    utils
    backend
)

# 设置库的头文件包含
target_include_directories(${LIBRARY_NAME}
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# 导出库的版本信息
set_target_properties(${LIBRARY_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
) 