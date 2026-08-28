#ifndef ORB_SLAM2_SEMANTIC_ONLINE_DYNAMIC_STATE_H
#define ORB_SLAM2_SEMANTIC_ONLINE_DYNAMIC_STATE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "semantic/DynamicScoreMap.h"
#include "semantic/IpcMaskProvider.h"

namespace ORB_SLAM2 {
namespace semantic {

struct OnlineDynamicConfig {
    OnlineDynamicConfig();
    int min_confirming_observations;
    double base_motion_threshold_m;
    double robust_sigma_multiplier;
    double dynamic_enter_threshold;
    double dynamic_exit_threshold;
    double dynamic_evidence_increment;
    double static_evidence_decrement;
    double association_gate_m;
    std::uint64_t max_prediction_age_ns;
    int min_valid_depth_pixels;
    double depth_scale;
};

struct OnlineCamera {
    OnlineCamera();
    double fx;
    double fy;
    double cx;
    double cy;
    int image_width;
    int image_height;
};

struct OnlineObservation {
    OnlineObservation();
    std::string label;
    cv::Vec3d centroid_world;
    cv::Vec3d mad_world;
    double box_width_pixels;
    double box_height_pixels;
};

struct OnlinePrediction {
    OnlinePrediction();
    DynamicScoreMap score_map;
    std::size_t strong_track_count;
    std::size_t unconfirmed_track_count;
    std::string reason;
};

class OnlineDynamicState {
public:
    explicit OnlineDynamicState(const OnlineDynamicConfig& config);
    ~OnlineDynamicState();
    void updateCompletedFrame(const std::vector<OnlineObservation>& observations,
                              std::uint64_t timestamp_ns);
    void updateCompletedFrame(const SemanticPacket& packet,
                              const cv::Mat& depth,
                              const cv::Mat& T_world_camera,
                              const OnlineCamera& camera);
    OnlinePrediction predict(std::uint64_t target_timestamp_ns,
                             const cv::Mat& last_T_world_camera,
                             const OnlineCamera& camera) const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace semantic
}  // namespace ORB_SLAM2

#endif
