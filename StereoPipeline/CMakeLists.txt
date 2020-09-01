cmake_minimum_required(VERSION 3.18)

if (NOT OpenCV_FOUND)
    message(WARNING "OpenCV not included in parent CMake project")
    find_package( OpenCV 4.0.0 REQUIRED )
endif()

project("StereoPipelineLib")

if (DEFINED CATKIN_DEVEL_PREFIX)
  find_package(catkin REQUIRED)
  message(STATUS "Catkin env set, compiling ${PROJECT_NAME} with ROS")

  catkin_package(
    LIBRARIES ${PROJECT_NAME}
  )
else()
  message(STATUS "Catkin env not set, compiling ${PROJECT_NAME} without ROS")
endif()

add_subdirectory(StereoCommon)
add_subdirectory(Classical)
add_subdirectory(Detectors)

add_library(${PROJECT_NAME} INTERFACE)

target_link_libraries(${PROJECT_NAME} 
    INTERFACE ClassicalLib
    INTERFACE DetectorsLib
)
