# 添加子目录
add_subdirectory(utils)
if(BUILD_CUDA)
    add_subdirectory(cuda)
endif()
if(BUILD_OPENMP)
    add_subdirectory(openmp)
endif()

# 创建backend库
add_library(backend INTERFACE)

# 设置包含目录
target_include_directories(backend
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 链接utils
target_link_libraries(backend
    INTERFACE
    backend_utils
)

# 链接CUDA库
if(BUILD_CUDA)
    target_link_libraries(backend
        INTERFACE
        backend_cuda
        CUDA::cudart
    )
endif()

# 设置安装规则
install(TARGETS backend
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
) 