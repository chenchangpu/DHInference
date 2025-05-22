set(TARGET_NAME model)

# 源文件
set(SOURCES
    model.cpp
    model_loader.cpp
)

# 头文件
# set(HEADERS
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include/model/model.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../include/model/model_loader.h
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