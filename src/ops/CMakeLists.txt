set(TARGET_NAME ops)

# 源文件
set(SOURCES
    layernorm.cpp
    matmul.cpp
    attention.cpp
    ffn.cpp
    activation.cpp
)

# 头文件
# set(HEADERS
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ops/layernorm.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ops/matmul.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ops/attention.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ops/ffn.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ops/activation.h
# )

# 创建库
add_library(${TARGET_NAME} STATIC ${SOURCES})

# 设置目标属性
target_include_directories(${TARGET_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 链接依赖
target_link_libraries(${TARGET_NAME}
    PUBLIC
    utils
) 