find_package(OpenCV REQUIRED)

file(GLOB LOCAL_SRC "src/*.cpp")

add_library(ClassicalLib 
    ${LOCAL_SRC}
)

target_link_libraries(ClassicalLib
    DetectorsLib
    ${OPENCV_LIBRARIES}
)

target_include_directories(ClassicalLib
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/includes"
)