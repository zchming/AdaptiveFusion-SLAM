cmake_minimum_required(VERSION 3.16)

project(AdaptiveFusion_SLAM VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.6 REQUIRED COMPONENTS calib3d core features2d imgcodecs imgproc video)
find_package(Ceres 2.2 REQUIRED)

add_library(adaptivefusion_slam
    src/camera.cpp
    src/failure_prediction_dataset.cpp
    src/geometric_health.cpp
    src/keyframe.cpp
    src/keyframe_policy.cpp
    src/lk_optical_flow_tracker.cpp
    src/local_bundle_adjuster.cpp
    src/map_point.cpp
    src/orb_feature_extractor.cpp
    src/orb_feature_matcher.cpp
    src/pnp_pose_estimator.cpp
    src/rgbd_correspondence_builder.cpp
    src/rgbd_odometry.cpp
    src/temporal_risk_predictor.cpp
    src/sparse_map.cpp
    src/trajectory.cpp
    src/tum_rgbd_dataset.cpp
)

target_include_directories(adaptivefusion_slam
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(adaptivefusion_slam
    PUBLIC
        Eigen3::Eigen
        Ceres::ceres
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

add_executable(run_rgbd_odometry
    app/run_rgbd_odometry.cpp
)

add_executable(simulate_failure_dataset
    app/simulate_failure_dataset.cpp
)

add_executable(simulate_risk_prediction
    app/simulate_risk_prediction.cpp
)

target_link_libraries(run_rgbd_odometry
    PRIVATE
        adaptivefusion_slam
)

target_link_libraries(simulate_failure_dataset
    PRIVATE
        adaptivefusion_slam
)

target_link_libraries(simulate_risk_prediction
    PRIVATE
        adaptivefusion_slam
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

    add_executable(test_pnp_pose_estimator
        tests/test_pnp_pose_estimator.cpp
    )

    target_link_libraries(test_pnp_pose_estimator
        PRIVATE
            adaptivefusion_slam
    )

    add_test(NAME pnp_pose_estimator COMMAND test_pnp_pose_estimator)

    add_executable(test_rgbd_odometry
        tests/test_rgbd_odometry.cpp
    )

    target_link_libraries(test_rgbd_odometry
        PRIVATE
            adaptivefusion_slam
    )

    add_test(NAME rgbd_odometry COMMAND test_rgbd_odometry)

    add_executable(test_sparse_map
        tests/test_sparse_map.cpp
    )

    target_link_libraries(test_sparse_map
        PRIVATE
            adaptivefusion_slam
    )

    add_test(NAME sparse_map COMMAND test_sparse_map)

    add_executable(test_local_bundle_adjuster
        tests/test_local_bundle_adjuster.cpp
    )

    target_link_libraries(test_local_bundle_adjuster
        PRIVATE
            adaptivefusion_slam
    )

    add_test(NAME local_bundle_adjuster COMMAND test_local_bundle_adjuster)

    add_executable(test_geometric_health
        tests/test_geometric_health.cpp
    )

    target_link_libraries(test_geometric_health
        PRIVATE
            adaptivefusion_slam
    )

    add_test(NAME geometric_health COMMAND test_geometric_health)

    add_executable(test_failure_prediction_dataset
        tests/test_failure_prediction_dataset.cpp
    )

    target_link_libraries(test_failure_prediction_dataset
        PRIVATE
            adaptivefusion_slam
    )

    add_test(
        NAME failure_prediction_dataset
        COMMAND test_failure_prediction_dataset
    )

    add_executable(test_temporal_risk_predictor
        tests/test_temporal_risk_predictor.cpp
    )

    target_link_libraries(test_temporal_risk_predictor
        PRIVATE
            adaptivefusion_slam
    )

    add_test(
        NAME temporal_risk_predictor
        COMMAND test_temporal_risk_predictor
    )
endif()
