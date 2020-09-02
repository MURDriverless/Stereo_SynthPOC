cmake_minimum_required(VERSION 3.18)

find_package(tkDNN)

project("DetectorsLib")

file(GLOB LOCAL_SRC "src/*.cpp")

add_library(${PROJECT_NAME}
    ${LOCAL_SRC}
)

target_link_libraries(${PROJECT_NAME} 
    ${tkDNN_LIBRARIES}
    ${OPENCV_LIBRARIES}
    nvonnxparser
    StereoCommon
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${tkDNN_INCLUDE_DIRS}
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes
)