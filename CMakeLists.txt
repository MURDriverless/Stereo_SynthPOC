cmake_minimum_required(VERSION 3.18)
project(mur_stereo)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package( OpenCV 4.0.0 REQUIRED )

add_subdirectory(StereoPipeline)
add_subdirectory(rosNodes)