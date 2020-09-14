cmake_minimum_required(VERSION 3.18)
project("mur_stereo_pipeline")

if (NOT OpenCV_FOUND)
    message(WARNING "OpenCV not included in parent CMake project")
    find_package( OpenCV 4.0.0 REQUIRED )
endif()

if (DEFINED CATKIN_DEVEL_PREFIX)
  find_package(catkin REQUIRED)
  message(STATUS "Catkin env set, compiling ${PROJECT_NAME} with ROS")

  catkin_package(
    INCLUDE_DIRS includes/Classical includes/Detectors includes/StereoCommon
    LIBRARIES ${PROJECT_NAME}
  )
else()
  message(STATUS "Catkin env not set, compiling ${PROJECT_NAME} without ROS")
endif()

find_package(tkDNN)

add_library(${PROJECT_NAME}
    src/Classical/ClassicalStereo.cpp
    src/Detectors/EntropyCalibrator.cpp
    src/Detectors/Utils.cpp
    src/Detectors/Detectors.cpp
    src/Detectors/KeypointDetector.cpp
  )

target_link_libraries(${PROJECT_NAME} 
    ${tkDNN_LIBRARIES}
    ${OPENCV_LIBRARIES}
    nvonnxparser
)

target_include_directories(${PROJECT_NAME}
    PUBLIC ${tkDNN_INCLUDE_DIRS}
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC includes/Classical
    PUBLIC includes/Detectors
    PUBLIC includes/StereoCommon
)