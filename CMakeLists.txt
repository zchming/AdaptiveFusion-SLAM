cmake_minimum_required(VERSION 3.16)

project(AdaptiveFusion_SLAM VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.6 REQUIRED COMPONENTS core imgcodecs)

add_library(adaptivefusion_slam
    src/camera.cpp
    src/tum_rgbd_dataset.cpp
)

target_include_directories(adaptivefusion_slam
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(adaptivefusion_slam
    PUBLIC
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

add_executable(run_slam
    app/run_slam.cpp
)

target_link_libraries(run_slam
    PRIVATE
        adaptivefusion_slam
)

add_executable(inspect_tum_dataset
    app/inspect_tum_dataset.cpp
)

target_link_libraries(inspect_tum_dataset
    PRIVATE
        adaptivefusion_slam
)

if(BUILD_TESTING)
    add_test(NAME camera_round_trip COMMAND run_slam)

    add_executable(test_tum_rgbd_dataset
        tests/test_tum_rgbd_dataset.cpp
    )

    target_compile_definitions(test_tum_rgbd_dataset
        PRIVATE
            TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data/tum_sample"
    )

    target_link_libraries(test_tum_rgbd_dataset
        PRIVATE
            adaptivefusion_slam
    )

    add_test(NAME tum_rgbd_dataset COMMAND test_tum_rgbd_dataset)
endif()
