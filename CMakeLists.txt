cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project( main )

file(GLOB LOCAL_SRC "src/*.cpp")
file(GLOB LOCAL_INC "includes/*.hpp")


find_package( OpenCV 4.0.0 REQUIRED )

add_subdirectory(StereoPipeline)

add_executable( main 
    src/main.cpp
    ${LOCAL_SRC}
    ${LOCAL_INC}
)

include_directories(
    includes
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries( main 
    StereoPipelineLib
    ${OPENCV_LIBRARIES}
    ${OpenCV_LIBS}
    Threads::Threads
)