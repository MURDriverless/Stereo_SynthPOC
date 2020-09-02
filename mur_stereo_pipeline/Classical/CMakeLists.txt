cmake_minimum_required(VERSION 3.18)

project("ClassicalLib")

add_library(${PROJECT_NAME} 
    src/ClassicalStereo.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OPENCV_LIBRARIES}
    StereoCommon
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes
)