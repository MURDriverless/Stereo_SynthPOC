cmake_minimum_required(VERSION 3.18)

project("StereoCommon")

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories( ${PROJECT_NAME}
    INTERFACE ${${PROJECT_NAME}_SOURCE_DIR}/includes
    INTERFACE ${OpenCV_INCLUDE_DIRS}
)