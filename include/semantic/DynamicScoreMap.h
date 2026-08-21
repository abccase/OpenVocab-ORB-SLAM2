#ifndef ORB_SLAM2_SEMANTIC_DYNAMIC_SCORE_MAP_H
#define ORB_SLAM2_SEMANTIC_DYNAMIC_SCORE_MAP_H

#include <cstdint>
#include <string>

#include <opencv2/core.hpp>

namespace ORB_SLAM2 {
namespace semantic {

struct DynamicScoreMap {
    DynamicScoreMap() : source_timestamp(0.0), frame_key(0), policy_seed(23011) {}

    double source_timestamp;
    std::uint64_t frame_key;
    std::uint64_t policy_seed;
    cv::Mat scores_f32;
    std::string manifest_sha256;
    std::string semantic_packet_sha256;
};

}  // namespace semantic
}  // namespace ORB_SLAM2

#endif
