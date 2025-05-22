set(TARGET_NAME utils)

# 源文件
set(SOURCES
    tensor.cpp
    memory.cpp
    logger.cpp
)

# 创建库
add_library(${TARGET_NAME} STATIC ${SOURCES})

# 设置目标属性
target_include_directories(${TARGET_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
) 