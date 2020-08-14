project(stereo)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-g")

set(SOURCES stereomatch.cpp)

add_executable(stereo_match ${SOURCES})

set(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})
target_link_libraries(stereo_match ${OpenCV_LIBS})
