#include <gtest/gtest.h>

#include "RunTelemetry.h"

TEST(RunTelemetry, FormatsCompleteDeterministicJsonLine) {
    EXPECT_EQ(
        ORB_SLAM2::FormatFrameTelemetry(3, 1.25, 2, true, 0.005),
        "{\"frame_index\":3,\"timestamp\":1.25,\"tracking_state\":2,"
        "\"pose_valid\":true,\"tracking_time_seconds\":0.005}");
}
