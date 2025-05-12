set(TARGET_NAME backend)

# 添加openmp后端子目录
if(BUILD_OPENMP)
    add_subdirectory(openmp)
endif()

# 如果启用CUDA，添加CUDA后端子目录
if(BUILD_CUDA)
    add_subdirectory(cuda)
endif()

# 创建接口库汇总所有后端
add_library(${TARGET_NAME} INTERFACE)

# 链接openmp后端
if(BUILD_OPENMP)
    target_link_libraries(${TARGET_NAME}
        INTERFACE
        backend_openmp
    )
endif()

# 如果启用CUDA，链接CUDA后端
if(BUILD_CUDA)
    target_link_libraries(${TARGET_NAME}
        INTERFACE
        backend_cuda
    )
endif()

# 设置包含目录
target_include_directories(${TARGET_NAME}
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
) 