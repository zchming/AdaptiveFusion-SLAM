cmake_minimum_required(VERSION 3.16)

project(AdaptiveFusion_SLAM VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.6 REQUIRED COMPONENTS core features2d imgcodecs imgproc video)

add_library(adaptivefusion_slam
    src/camera.cpp
    src/lk_optical_flow_tracker.cpp
    src/orb_feature_extractor.cpp
    src/orb_feature_matcher.cpp
    src/rgbd_correspondence_builder.cpp
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

add_executable(run_feature_frontend
    app/run_feature_frontend.cpp
)

target_link_libraries(run_feature_frontend
    PRIVATE
        adaptivefusion_slam
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

    add_executable(test_orb_feature_extractor
        tests/test_orb_feature_extractor.cpp
    )

    target_link_libraries(test_orb_feature_extractor
        PRIVATE
            adaptivefusion_slam
    )

    add_test(NAME orb_feature_extractor COMMAND test_orb_feature_extractor)

    add_executable(test_orb_feature_matcher
        tests/test_orb_feature_matcher.cpp
    )

    target_link_libraries(test_orb_feature_matcher
        PRIVATE
            adaptivefusion_slam
    )

    add_test(NAME orb_feature_matcher COMMAND test_orb_feature_matcher)

    add_executable(test_lk_optical_flow_tracker
        tests/test_lk_optical_flow_tracker.cpp
    )

    target_link_libraries(test_lk_optical_flow_tracker
        PRIVATE
            adaptivefusion_slam
    )

    add_test(NAME lk_optical_flow_tracker COMMAND test_lk_optical_flow_tracker)

    add_executable(test_rgbd_correspondence_builder
        tests/test_rgbd_correspondence_builder.cpp
    )

    target_link_libraries(test_rgbd_correspondence_builder
        PRIVATE
            adaptivefusion_slam
    )

    add_test(
        NAME rgbd_correspondence_builder
        COMMAND test_rgbd_correspondence_builder
    )
endif()
