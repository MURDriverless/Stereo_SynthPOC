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

find_package(CUDNN REQUIRED)
include_directories(${CUDNN_INCLUDE_DIR})

# compile
file(GLOB tkdnn_CUSRC "src/tkDNN/src/kernels/*.cu" "src/tkDNN/src/sorting.cu")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/tkDNN/include ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS})
cuda_add_library(kernels SHARED ${tkdnn_CUSRC})

#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
file(GLOB tkdnn_SRC "src/tkDNN/src/*.cpp")
set(tkdnn_LIBS kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/tkDNN/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES})
add_library(tkDNN SHARED ${tkdnn_SRC})
target_link_libraries(tkDNN ${tkdnn_LIBS})

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

add_library(keypoint 
    src/Detectors.cpp
    src/KeypointDetector.cpp 
    src/Utils.cpp 
    src/EntropyCalibrator.cpp
)

target_link_libraries(keypoint ${OPENCV_LIBRARIES} nvonnxparser)

add_executable( main 
    src/main.cpp
    ${LOCAL_SRC}
    ${LOCAL_INC}
)

target_link_libraries(  main 
    tkDNN 
    keypoint
    ${OPENCV_LIBRARIES}
    ${OpenCV_LIBS}
    Threads::Threads
)