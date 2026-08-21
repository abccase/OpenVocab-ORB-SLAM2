#include "semantic/Telemetry.h"

#include <iomanip>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace ORB_SLAM2 {
namespace semantic {

FrameTelemetry::FrameTelemetry()
    : raw_keypoints(0), used_keypoints(0), removed_dynamic(0),
      retained_uncertain(0), removed_uncertain(0), semantic_accessed(false),
      semantic_state(SemanticState::BASELINE), cache_load_seconds(0.0),
      policy_seconds(0.0), pacing_lateness_seconds(0.0) {}

PacingDecision decidePacing(double interval_seconds, double elapsed_seconds) {
    if (!std::isfinite(interval_seconds) || interval_seconds < 0.0 ||
        !std::isfinite(elapsed_seconds) || elapsed_seconds < 0.0)
        throw std::invalid_argument("pacing inputs must be finite and nonnegative");
    PacingDecision decision;
    decision.sleep_seconds = std::max(0.0, interval_seconds - elapsed_seconds);
    decision.lateness_seconds = std::max(0.0, elapsed_seconds - interval_seconds);
    return decision;
}

const char* semanticStateName(SemanticState state) {
    switch (state) {
        case SemanticState::BASELINE: return "BASELINE";
        case SemanticState::CACHE_VALID: return "CACHE_VALID";
        case SemanticState::ONLINE_VALID: return "ONLINE_VALID";
        case SemanticState::DEGRADED_TO_BASELINE: return "DEGRADED_TO_BASELINE";
    }
    return "UNKNOWN";
}

std::string telemetryCsvHeader() {
    return "frame_index,timestamp,tracking_state,pose_valid,tracking_time_seconds,"
           "raw_keypoints,used_keypoints,removed_dynamic,retained_uncertain,"
           "removed_uncertain,semantic_accessed,semantic_state,cache_load_seconds,"
           "policy_seconds,pacing_lateness_seconds";
}

std::string formatTelemetryCsv(std::size_t frame_index, double timestamp,
                               int tracking_state, bool pose_valid,
                               double tracking_time_seconds,
                               const FrameTelemetry& telemetry) {
    std::ostringstream output;
    output << std::setprecision(15) << frame_index << ',' << timestamp << ','
           << tracking_state << ',' << (pose_valid ? 1 : 0) << ','
           << tracking_time_seconds << ',' << telemetry.raw_keypoints << ','
           << telemetry.used_keypoints << ',' << telemetry.removed_dynamic << ','
           << telemetry.retained_uncertain << ',' << telemetry.removed_uncertain << ','
           << (telemetry.semantic_accessed ? 1 : 0) << ','
           << semanticStateName(telemetry.semantic_state) << ','
           << telemetry.cache_load_seconds << ',' << telemetry.policy_seconds << ','
           << telemetry.pacing_lateness_seconds;
    return output.str();
}

}  // namespace semantic
}  // namespace ORB_SLAM2
