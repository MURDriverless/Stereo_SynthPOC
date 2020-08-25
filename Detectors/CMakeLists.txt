find_package(OpenCV REQUIRED)

find_package(tkDNN)

file(GLOB LOCAL_SRC "src/*.cpp")

add_library(DetectorsLib 
    ${LOCAL_SRC}
)

target_link_libraries(DetectorsLib 
    ${tkDNN_LIBRARIES}
    ${OPENCV_LIBRARIES}
    nvonnxparser
)

target_include_directories(DetectorsLib
    PUBLIC ${tkDNN_INCLUDE_DIRS}
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/includes"
)