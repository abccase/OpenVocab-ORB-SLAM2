#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "semantic/FeatureMaskPolicy.h"
#include "semantic/Telemetry.h"

namespace {

using ORB_SLAM2::semantic::FeatureReason;
using ORB_SLAM2::semantic::PolicyConfig;
using ORB_SLAM2::semantic::decideFeature;
using ORB_SLAM2::semantic::makeFrameKey;
using ORB_SLAM2::semantic::parsePolicySeed;

TEST(FeatureMaskPolicy, AppliesFrozenThresholdBoundaries) {
    const PolicyConfig config = {0.70f, 0.40f, 0.50f, 23011};
    EXPECT_TRUE(decideFeature(0.39999f, 10, 20, 99, config).keep);
    EXPECT_EQ(FeatureReason::LOW_SCORE_KEEP,
              decideFeature(0.39999f, 10, 20, 99, config).reason);
    EXPECT_EQ(0.5f, decideFeature(0.40f, 10, 20, 99, config).semantic_weight);
    EXPECT_FALSE(decideFeature(0.70f, 10, 20, 99, config).keep);
    EXPECT_EQ(FeatureReason::HIGH_SCORE_REMOVE,
              decideFeature(0.71f, 10, 20, 99, config).reason);
}

TEST(PacingDecision, DeductsAllPostLoadProcessingAndReportsLateness) {
    const ORB_SLAM2::semantic::PacingDecision early =
        ORB_SLAM2::semantic::decidePacing(0.05, 0.02);
    EXPECT_DOUBLE_EQ(early.sleep_seconds, 0.03);
    EXPECT_DOUBLE_EQ(early.lateness_seconds, 0.0);
    const ORB_SLAM2::semantic::PacingDecision late =
        ORB_SLAM2::semantic::decidePacing(0.05, 0.08);
    EXPECT_DOUBLE_EQ(late.sleep_seconds, 0.0);
    EXPECT_DOUBLE_EQ(late.lateness_seconds, 0.03);
}

TEST(TelemetryCsv, RoundTripsRealTumTimestampAndAllDoubleFields) {
    const double timestamp = 1341845820.751833;
    const double tracking = 0.012345678901234567;
    ORB_SLAM2::semantic::FrameTelemetry telemetry;
    telemetry.cache_load_seconds = 0.023456789012345678;
    telemetry.policy_seconds = 0.034567890123456789;
    telemetry.pacing_lateness_seconds = 0.045678901234567891;
    telemetry.ipc_call_seconds = 0.0012345678901234567;
    telemetry.ipc_reason = "NO_PACKET";
    telemetry.request_attempted = true;
    telemetry.request_sent = false;
    telemetry.packet_age_ms = 200.0;
    telemetry.inference_ms = 45.0;
    telemetry.strong_track_count = 2;
    telemetry.unconfirmed_track_count = 1;
    const std::string csv = ORB_SLAM2::semantic::formatTelemetryCsv(
        0, timestamp, 2, true, tracking, telemetry);
    std::vector<std::string> fields;
    std::istringstream stream(csv);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    ASSERT_EQ(23u, fields.size());
    EXPECT_EQ(timestamp, std::stod(fields[1]));
    EXPECT_EQ(tracking, std::stod(fields[4]));
    EXPECT_EQ(telemetry.cache_load_seconds, std::stod(fields[12]));
    EXPECT_EQ(telemetry.policy_seconds, std::stod(fields[13]));
    EXPECT_EQ(telemetry.pacing_lateness_seconds, std::stod(fields[14]));
    EXPECT_EQ(telemetry.ipc_call_seconds, std::stod(fields[15]));
    EXPECT_EQ("NO_PACKET", fields[16]);
    EXPECT_EQ("1", fields[17]);
    EXPECT_EQ("0", fields[18]);
    EXPECT_EQ(telemetry.packet_age_ms, std::stod(fields[19]));
    EXPECT_EQ(telemetry.inference_ms, std::stod(fields[20]));
    EXPECT_EQ("2", fields[21]);
    EXPECT_EQ("1", fields[22]);
}

TEST(FeatureMaskPolicy, UncertainDecisionIsStableAndSequenceBound) {
    const PolicyConfig config = {0.70f, 0.40f, 0.50f, 23011};
    const std::uint64_t a_key = makeFrameKey("fr3_sitting_xyz", "1341845820.751833");
    const std::uint64_t b_key = makeFrameKey("fr3_walking_xyz", "1341845820.751833");
    EXPECT_NE(a_key, b_key);
    const auto first = decideFeature(0.55f, 10, 20, a_key, config);
    const auto second = decideFeature(0.55f, 10, 20, a_key, config);
    EXPECT_EQ(first.keep, second.keep);
    EXPECT_EQ(first.reason, second.reason);
    EXPECT_FLOAT_EQ(0.5f, first.semantic_weight);
}

TEST(FeatureMaskPolicy, UsesFrozenFnvByteEncodingAndLittleEndianIntegers) {
    const PolicyConfig config = {0.70f, 0.40f, 0.50f, 23011};
    const std::uint64_t frame_key =
        makeFrameKey("fr3_sitting_xyz", "1341845820.751833");
    EXPECT_EQ(UINT64_C(0xdeada92a12f8bffc), frame_key);
    const auto decision = decideFeature(0.55f, 10, 20, frame_key, config);
    EXPECT_FALSE(decision.keep);  // hash 0x8795fe95c0eed056, normalized > 0.5.
    EXPECT_EQ(FeatureReason::UNCERTAIN_HASH_REMOVE, decision.reason);
}

TEST(FeatureMaskPolicy, UncertainRetentionConvergesToFrozenFraction) {
    const PolicyConfig config = {0.70f, 0.40f, 0.50f, 23011};
    int kept = 0;
    for (int i = 0; i < 10000; ++i)
        kept += decideFeature(0.55f, i % 640, i / 640, 42, config).keep ? 1 : 0;
    EXPECT_GT(kept, 4700);
    EXPECT_LT(kept, 5300);
}

TEST(FeatureMaskPolicy, ParsesFullUint64SeedAndRejectsAmbiguousValues) {
    EXPECT_EQ(UINT64_MAX, parsePolicySeed("18446744073709551615"));
    EXPECT_EQ(UINT64_C(0), parsePolicySeed("0"));
    EXPECT_THROW(parsePolicySeed(""), std::invalid_argument);
    EXPECT_THROW(parsePolicySeed("-1"), std::invalid_argument);
    EXPECT_THROW(parsePolicySeed("+1"), std::invalid_argument);
    EXPECT_THROW(parsePolicySeed("18446744073709551616"), std::invalid_argument);
}

}  // namespace
