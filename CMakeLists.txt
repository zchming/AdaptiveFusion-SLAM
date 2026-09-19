cmake_minimum_required(VERSION 3.16)

project(AdaptiveFusion_SLAM VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_library(adaptivefusion_slam
    src/camera.cpp
)

target_include_directories(adaptivefusion_slam
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(adaptivefusion_slam
    PUBLIC
        Eigen3::Eigen
)

add_executable(run_slam
    app/run_slam.cpp
)

target_link_libraries(run_slam
    PRIVATE
        adaptivefusion_slam
)
