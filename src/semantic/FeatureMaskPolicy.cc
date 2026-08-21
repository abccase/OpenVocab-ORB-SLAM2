#include "semantic/FeatureMaskPolicy.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ORB_SLAM2 {
namespace semantic {
namespace {

const std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
const std::uint64_t kFnvPrime = UINT64_C(1099511628211);

void HashByte(std::uint64_t* hash, unsigned char byte) {
    *hash ^= byte;
    *hash *= kFnvPrime;
}

template <typename T>
void HashLittleEndian(std::uint64_t* hash, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i)
        HashByte(hash, static_cast<unsigned char>((value >> (8 * i)) & 0xff));
}

}  // namespace

std::uint64_t makeFrameKey(const std::string& sequence_id,
                           const std::string& exact_timestamp) {
    if (sequence_id.empty() || exact_timestamp.empty())
        throw std::invalid_argument("sequence ID and exact timestamp are required");
    std::uint64_t hash = kFnvOffset;
    for (std::size_t i = 0; i < sequence_id.size(); ++i)
        HashByte(&hash, static_cast<unsigned char>(sequence_id[i]));
    HashByte(&hash, 0);
    for (std::size_t i = 0; i < exact_timestamp.size(); ++i)
        HashByte(&hash, static_cast<unsigned char>(exact_timestamp[i]));
    return hash;
}

std::uint64_t parsePolicySeed(const std::string& decimal_seed) {
    if (decimal_seed.empty())
        throw std::invalid_argument("policy seed must be an unsigned decimal integer");
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < decimal_seed.size(); ++i) {
        const char character = decimal_seed[i];
        if (character < '0' || character > '9')
            throw std::invalid_argument("policy seed must be an unsigned decimal integer");
        const std::uint64_t digit = static_cast<std::uint64_t>(character - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
            throw std::invalid_argument("policy seed exceeds uint64 range");
        value = value * 10 + digit;
    }
    return value;
}

FeatureDecision decideFeature(float score, int x, int y,
                              std::uint64_t frame_key,
                              const PolicyConfig& config) {
    if (!std::isfinite(score) || score < 0.0f || score > 1.0f)
        throw std::invalid_argument("dynamic score must be finite and in [0,1]");
    if (!(config.low_dynamic_threshold >= 0.0f &&
          config.low_dynamic_threshold < config.high_dynamic_threshold &&
          config.high_dynamic_threshold <= 1.0f &&
          config.uncertain_retention_fraction >= 0.0f &&
          config.uncertain_retention_fraction <= 1.0f))
        throw std::invalid_argument("invalid feature mask policy configuration");

    if (score < config.low_dynamic_threshold)
        return FeatureDecision{true, 1.0f, FeatureReason::LOW_SCORE_KEEP};
    if (score >= config.high_dynamic_threshold)
        return FeatureDecision{false, 0.0f, FeatureReason::HIGH_SCORE_REMOVE};

    std::uint64_t hash = kFnvOffset;
    HashLittleEndian(&hash, config.seed);
    HashLittleEndian(&hash, frame_key);
    HashLittleEndian(&hash, static_cast<std::uint32_t>(x));
    HashLittleEndian(&hash, static_cast<std::uint32_t>(y));
    const long double normalized = static_cast<long double>(hash) /
        (static_cast<long double>(std::numeric_limits<std::uint64_t>::max()) + 1.0L);
    const bool keep = normalized < config.uncertain_retention_fraction;
    return FeatureDecision{keep, 0.5f,
                           keep ? FeatureReason::UNCERTAIN_HASH_KEEP
                                : FeatureReason::UNCERTAIN_HASH_REMOVE};
}

const char* featureReasonName(FeatureReason reason) {
    switch (reason) {
        case FeatureReason::LOW_SCORE_KEEP: return "LOW_SCORE_KEEP";
        case FeatureReason::HIGH_SCORE_REMOVE: return "HIGH_SCORE_REMOVE";
        case FeatureReason::UNCERTAIN_HASH_KEEP: return "UNCERTAIN_HASH_KEEP";
        case FeatureReason::UNCERTAIN_HASH_REMOVE: return "UNCERTAIN_HASH_REMOVE";
    }
    return "UNKNOWN";
}

}  // namespace semantic
}  // namespace ORB_SLAM2
