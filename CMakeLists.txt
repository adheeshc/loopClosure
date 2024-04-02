cmake_minimum_required(VERSION 3.0.0)
project(loopClosure)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fbow REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIR})

set(source_files
    src/featureTraining.cpp
)

add_executable(featureTraining ${source_files})

target_link_libraries(featureTraining ${OpenCV_LIBS})
target_link_libraries(featureTraining Eigen3::Eigen)
target_link_libraries(featureTraining fbow)

set(source_files
    src/loopClosure.cpp
)

add_executable(loopClosure ${source_files})

target_link_libraries(loopClosure ${OpenCV_LIBS})
target_link_libraries(loopClosure Eigen3::Eigen)
target_link_libraries(loopClosure fbow)

# Add the -pthread flag to the compiler and linker flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
