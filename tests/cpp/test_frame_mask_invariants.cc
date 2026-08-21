#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "Frame.h"
#include "ORBextractor.h"
#include "semantic/DynamicScoreMap.h"
#include "semantic/Telemetry.h"

namespace {

cv::Mat Checkerboard() {
    cv::Mat image(480, 640, CV_8UC1);
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
            image.at<unsigned char>(y, x) = ((x / 12 + y / 12) % 2) ? 255 : 0;
    return image;
}

ORB_SLAM2::Frame MakeFrame(const cv::Mat& image,
                           ORB_SLAM2::ORBextractor* extractor,
                           const ORB_SLAM2::semantic::DynamicScoreMap* scores,
                           ORB_SLAM2::semantic::FrameTelemetry* telemetry) {
    cv::Mat depth(image.rows, image.cols, CV_32F, cv::Scalar(1.0f));
    cv::Mat K = (cv::Mat_<float>(3, 3) << 525.0f, 0.0f, 319.5f,
                 0.0f, 525.0f, 239.5f, 0.0f, 0.0f, 1.0f);
    cv::Mat distortion = cv::Mat::zeros(4, 1, CV_32F);
    return ORB_SLAM2::Frame(image, depth, 1.25, extractor, NULL, K,
                            distortion, 40.0f, 3.0f, scores, telemetry);
}

TEST(FrameMaskInvariants, NullMapPreservesOriginalExtractedArrays) {
    cv::Mat image = Checkerboard();
    ORB_SLAM2::ORBextractor extractor(500, 1.2f, 8, 20, 7);
    ORB_SLAM2::semantic::FrameTelemetry telemetry;
    ORB_SLAM2::Frame frame = MakeFrame(image, &extractor, NULL, &telemetry);
    EXPECT_GT(frame.N, 0);
    EXPECT_EQ(frame.N, frame.mDescriptors.rows);
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvKeys.size()));
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvKeysUn.size()));
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvDepth.size()));
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvuRight.size()));
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvpMapPoints.size()));
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvbOutlier.size()));
    EXPECT_FALSE(telemetry.semantic_accessed);
    EXPECT_EQ(telemetry.raw_keypoints, telemetry.used_keypoints);
}

TEST(FrameMaskInvariants, FilteringPreservesAllAlignedIndexesAndGridPlacement) {
    cv::Mat image = Checkerboard();
    ORB_SLAM2::ORBextractor extractor(500, 1.2f, 8, 20, 7);
    ORB_SLAM2::semantic::DynamicScoreMap scores;
    scores.source_timestamp = 1.25;
    scores.frame_key = 99;
    scores.manifest_sha256 = std::string(64, 'a');
    scores.scores_f32 = cv::Mat(image.rows, image.cols, CV_32FC1, cv::Scalar(0.55f));
    scores.scores_f32(cv::Rect(0, 0, image.cols / 2, image.rows)).setTo(1.0f);
    ORB_SLAM2::semantic::FrameTelemetry telemetry;
    ORB_SLAM2::Frame frame = MakeFrame(image, &extractor, &scores, &telemetry);
    EXPECT_GT(telemetry.raw_keypoints, frame.N);
    EXPECT_EQ(frame.N, telemetry.used_keypoints);
    EXPECT_EQ(frame.N, frame.mDescriptors.rows);
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvKeys.size()));
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvKeysUn.size()));
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvDepth.size()));
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvuRight.size()));
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvpMapPoints.size()));
    EXPECT_EQ(frame.N, static_cast<int>(frame.mvbOutlier.size()));
    for (int i = 0; i < frame.N; ++i) {
        EXPECT_GE(cvRound(frame.mvKeys[i].pt.x), image.cols / 2);
        int grid_x = -1;
        int grid_y = -1;
        EXPECT_TRUE(frame.PosInGrid(frame.mvKeysUn[i], grid_x, grid_y));
        EXPECT_GE(frame.mvDepth[i], 0.0f);
    }
}

TEST(FrameMaskInvariants, HighScoreMapProducesNoMapCandidates) {
    cv::Mat image = Checkerboard();
    ORB_SLAM2::ORBextractor extractor(500, 1.2f, 8, 20, 7);
    ORB_SLAM2::semantic::DynamicScoreMap scores;
    scores.source_timestamp = 1.25;
    scores.frame_key = 99;
    scores.manifest_sha256 = std::string(64, 'b');
    scores.scores_f32 = cv::Mat(image.rows, image.cols, CV_32FC1, cv::Scalar(1.0f));
    ORB_SLAM2::semantic::FrameTelemetry telemetry;
    ORB_SLAM2::Frame frame = MakeFrame(image, &extractor, &scores, &telemetry);
    EXPECT_EQ(0, frame.N);
    EXPECT_TRUE(frame.mvpMapPoints.empty());
    EXPECT_TRUE(frame.mvDepth.empty());
    EXPECT_EQ(telemetry.raw_keypoints, telemetry.removed_dynamic);
}

}  // namespace
