#ifndef ORB_SLAM2_SEMANTIC_FEATURE_MASK_POLICY_H
#define ORB_SLAM2_SEMANTIC_FEATURE_MASK_POLICY_H

#include <cstdint>
#include <string>

namespace ORB_SLAM2 {
namespace semantic {

enum class FeatureReason {
    LOW_SCORE_KEEP,
    HIGH_SCORE_REMOVE,
    UNCERTAIN_HASH_KEEP,
    UNCERTAIN_HASH_REMOVE
};

struct PolicyConfig {
    float high_dynamic_threshold;
    float low_dynamic_threshold;
    float uncertain_retention_fraction;
    std::uint64_t seed;
};

struct FeatureDecision {
    bool keep;
    float semantic_weight;
    FeatureReason reason;
};

std::uint64_t makeFrameKey(const std::string& sequence_id,
                           const std::string& exact_timestamp);
std::uint64_t parsePolicySeed(const std::string& decimal_seed);

FeatureDecision decideFeature(float score, int x, int y,
                              std::uint64_t frame_key,
                              const PolicyConfig& config);

const char* featureReasonName(FeatureReason reason);

}  // namespace semantic
}  // namespace ORB_SLAM2

#endif
