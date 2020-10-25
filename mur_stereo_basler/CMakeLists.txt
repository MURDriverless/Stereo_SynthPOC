cmake_minimum_required(VERSION 3.18)
project("mur_stereo_basler")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindPylon.cmake")

add_executable( ${PROJECT_NAME}
    src/main.cpp
    src/GeniWrap.cpp
)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC ${Pylon_INCLUDE_DIRS}
    PRIVATE includes
)

target_link_libraries( ${PROJECT_NAME} 
    mur_stereo_pipeline
    ${OPENCV_LIBRARIES}
    ${OpenCV_LIBS}
    ${Pylon_LIBRARIES} 
)