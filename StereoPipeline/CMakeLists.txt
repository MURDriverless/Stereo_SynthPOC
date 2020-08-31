cmake_minimum_required(VERSION 3.18)

project("StereoPipelineLib")

add_subdirectory(StereoCommon)
add_subdirectory(Classical)
add_subdirectory(Detectors)

add_library(${PROJECT_NAME} INTERFACE)

target_link_libraries(${PROJECT_NAME} 
    INTERFACE ClassicalLib
    INTERFACE DetectorsLib
)
