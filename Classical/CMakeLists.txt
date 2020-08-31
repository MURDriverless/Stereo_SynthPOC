add_library(ClassicalLib 
    src/ClassicalStereo.cpp
)

target_link_libraries(ClassicalLib
    ${OPENCV_LIBRARIES}
    StereoCommon
    DetectorsLib
)

target_include_directories(ClassicalLib
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes
)