#include <gtest/gtest.h>

#include <type_traits>

#include "System.h"

namespace {

using TrackRgbdSignature = cv::Mat (ORB_SLAM2::System::*)(
    const cv::Mat&, const cv::Mat&, const double&);

TEST(RgbdApi, PreservesOriginalTrackSignatureInHeadlessBuild) {
    auto method = static_cast<TrackRgbdSignature>(&ORB_SLAM2::System::TrackRGBD);
    EXPECT_TRUE((std::is_same<decltype(method), TrackRgbdSignature>::value));
    EXPECT_EQ(ORB_SLAM2_BUILD_VIEWER, 0);
}

}  // namespace
