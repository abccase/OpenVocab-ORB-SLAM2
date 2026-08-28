#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <vector>

#include <opencv2/core.hpp>

#include "semantic/OnlineDynamicState.h"
#include "semantic/IpcMaskProvider.h"

namespace {

ORB_SLAM2::semantic::OnlineDynamicConfig Config() {
    ORB_SLAM2::semantic::OnlineDynamicConfig config;
    config.min_confirming_observations = 3;
    config.base_motion_threshold_m = 0.10;
    config.robust_sigma_multiplier = 3.0;
    config.dynamic_enter_threshold = 0.70;
    config.dynamic_exit_threshold = 0.40;
    config.dynamic_evidence_increment = 0.35;
    config.static_evidence_decrement = 0.20;
    config.association_gate_m = 1.0;
    config.max_prediction_age_ns = 250000000ULL;
    config.min_valid_depth_pixels = 2;
    config.depth_scale = 5000.0;
    return config;
}

ORB_SLAM2::semantic::OnlineCamera Camera() {
    ORB_SLAM2::semantic::OnlineCamera camera;
    camera.fx = 100.0;
    camera.fy = 100.0;
    camera.cx = 50.0;
    camera.cy = 40.0;
    camera.image_width = 100;
    camera.image_height = 80;
    return camera;
}

ORB_SLAM2::semantic::OnlineObservation Observation(double x) {
    ORB_SLAM2::semantic::OnlineObservation observation;
    observation.label = "person";
    observation.centroid_world = cv::Vec3d(x, 0.0, 2.0);
    observation.mad_world = cv::Vec3d(0.01, 0.01, 0.01);
    observation.box_width_pixels = 12.0;
    observation.box_height_pixels = 20.0;
    return observation;
}

double Maximum(const cv::Mat& scores) {
    double maximum = 0.0;
    cv::minMaxLoc(scores, NULL, &maximum);
    return maximum;
}

TEST(OnlineDynamicState, FirstObservationNeverStronglyFilters) {
    ORB_SLAM2::semantic::OnlineDynamicState state(Config());
    state.updateCompletedFrame({Observation(0.0)}, 1000000000ULL);

    const ORB_SLAM2::semantic::OnlinePrediction prediction = state.predict(
        1050000000ULL, cv::Mat::eye(4, 4, CV_64F), Camera());

    EXPECT_EQ(0U, prediction.strong_track_count);
    EXPECT_EQ(1U, prediction.unconfirmed_track_count);
    EXPECT_FLOAT_EQ(0.25f, static_cast<float>(Maximum(prediction.score_map.scores_f32)));
}

TEST(OnlineDynamicState, ThreeMovingUpdatesCanAffectOnlyALaterFrame) {
    ORB_SLAM2::semantic::OnlineDynamicState state(Config());
    state.updateCompletedFrame({Observation(0.0)}, 1000000000ULL);
    state.updateCompletedFrame({Observation(0.2)}, 1050000000ULL);
    state.updateCompletedFrame({Observation(0.4)}, 1100000000ULL);
    state.updateCompletedFrame({Observation(0.6)}, 1150000000ULL);

    const ORB_SLAM2::semantic::OnlinePrediction prediction = state.predict(
        1200000000ULL, cv::Mat::eye(4, 4, CV_64F), Camera());

    EXPECT_EQ(1U, prediction.strong_track_count);
    EXPECT_GE(Maximum(prediction.score_map.scores_f32), 0.70);
    EXPECT_GT(prediction.score_map.source_timestamp, 1.15);
}

TEST(OnlineDynamicState, StaticTrackNeverBecomesStrongDynamic) {
    ORB_SLAM2::semantic::OnlineDynamicState state(Config());
    for (int index = 0; index < 5; ++index)
        state.updateCompletedFrame({Observation(0.0)}, 1000000000ULL + index * 50000000ULL);

    const ORB_SLAM2::semantic::OnlinePrediction prediction = state.predict(
        1250000000ULL, cv::Mat::eye(4, 4, CV_64F), Camera());

    EXPECT_EQ(0U, prediction.strong_track_count);
    EXPECT_LT(Maximum(prediction.score_map.scores_f32), 0.70);
}

TEST(OnlineDynamicState, PredictionsOlderThanTwoHundredFiftyMillisecondsAreDiscarded) {
    ORB_SLAM2::semantic::OnlineDynamicState state(Config());
    state.updateCompletedFrame({Observation(0.0)}, 1000000000ULL);

    const ORB_SLAM2::semantic::OnlinePrediction prediction = state.predict(
        1250000001ULL, cv::Mat::eye(4, 4, CV_64F), Camera());

    EXPECT_EQ(0U, prediction.strong_track_count);
    EXPECT_EQ(0U, prediction.unconfirmed_track_count);
    EXPECT_FLOAT_EQ(0.0f, static_cast<float>(Maximum(prediction.score_map.scores_f32)));
    EXPECT_EQ("PREDICTION_STALE", prediction.reason);
}

TEST(OnlineDynamicState, RejectsNonCausalCompletedFrameUpdates) {
    ORB_SLAM2::semantic::OnlineDynamicState state(Config());
    state.updateCompletedFrame({Observation(0.0)}, 1000000000ULL);

    EXPECT_THROW(state.updateCompletedFrame({Observation(0.1)}, 999999999ULL),
                 std::invalid_argument);
}

TEST(OnlineDynamicState, PacketDepthAndCompletedPoseProduceWorldObservation) {
    ORB_SLAM2::semantic::PacketExpectations expected;
    expected.run_id = "p06-fixture-run";
    expected.prompt_sha256 = std::string(64, '1');
    expected.model_manifest_sha256 = std::string(64, '2');
    expected.image_width = 3;
    expected.image_height = 2;
    expected.current_timestamp_ns = 1200000000ULL;
    expected.max_age_ns = 250000000ULL;
    std::ifstream stream(
        std::string(IPC_FIXTURE_DIR) + "/valid_semantic_packet.msgpack",
        std::ios::binary);
    const std::vector<unsigned char> bytes(
        (std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    const ORB_SLAM2::semantic::SemanticPacket packet =
        ORB_SLAM2::semantic::decodeSemanticPacket(bytes, expected);
    cv::Mat depth(2, 3, CV_32FC1, cv::Scalar(2.0f));
    ORB_SLAM2::semantic::OnlineCamera camera = Camera();
    camera.fx = 1.0;
    camera.fy = 1.0;
    camera.cx = 0.0;
    camera.cy = 0.0;
    camera.image_width = 3;
    camera.image_height = 2;
    ORB_SLAM2::semantic::OnlineDynamicState state(Config());

    state.updateCompletedFrame(
        packet, depth, cv::Mat::eye(4, 4, CV_64F), camera);
    const ORB_SLAM2::semantic::OnlinePrediction prediction = state.predict(
        1050000000ULL, cv::Mat::eye(4, 4, CV_64F), camera);

    EXPECT_EQ(1U, prediction.unconfirmed_track_count);
    EXPECT_FLOAT_EQ(0.25f, static_cast<float>(Maximum(prediction.score_map.scores_f32)));
}

}  // namespace
