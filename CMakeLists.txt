cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project( main )

# tkdnn

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/src/tkDNN/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wno-deprecated-declarations -Wno-unused-variable -pthread -DOPENCV_CUDACONTRIB -DCUDA_API_PER_THREAD_DEFAULT_STREAM")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/tkDNN/include/tkDNN)

add_definitions(-DTKDNN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/tkDNN")

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDA 9.0 REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 --compiler-options '-fPIC'")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32)

#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Local Sources
#-------------------------------------------------------------------------------
file(GLOB LOCAL_SRC "src/*.cpp")
file(GLOB LOCAL_INC "includes/*.hpp")

find_package( OpenCV 4.0.0 REQUIRED )

include_directories(
    includes
    ${OpenCV_INCLUDE_DIRS}
    )

add_executable(main 
    src/main.cpp
    ${LOCAL_SRC}
    ${LOCAL_INC}
    )

target_link_libraries( main 
    ${OPENCV_LIBRARIES}
    ${OpenCV_LIBS}
    Threads::Threads)