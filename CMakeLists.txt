cmake_minimum_required(VERSION 3.18)

project(mur_stereo)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package( OpenCV 4.0.0 REQUIRED )

add_subdirectory(mur_stereo_pipeline)

# mur_stereo_video
add_executable( mur_stereo_video 
    mur_stereo_video/src/main.cpp
)

target_include_directories( mur_stereo_video
    PUBLIC ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries( mur_stereo_video 
    mur_stereo_pipeline
    ${OPENCV_LIBRARIES}
    ${OpenCV_LIBS}
    Threads::Threads
)

# mur_stereo_images
add_executable( mur_stereo_image
    mur_stereo_image/src/main.cpp
)

target_include_directories( mur_stereo_image
    PUBLIC ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries( mur_stereo_image 
    mur_stereo_pipeline
    ${OPENCV_LIBRARIES}
    ${OpenCV_LIBS}
)

# mur_stereo_basler
add_subdirectory(mur_stereo_basler)