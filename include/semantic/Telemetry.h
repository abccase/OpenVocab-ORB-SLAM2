#ifndef ORB_SLAM2_SEMANTIC_TELEMETRY_H
#define ORB_SLAM2_SEMANTIC_TELEMETRY_H

#include <cstddef>
#include <string>

namespace ORB_SLAM2 {
namespace semantic {

enum class SemanticState {
    BASELINE,
    CACHE_VALID,
    ONLINE_VALID,
    DEGRADED_TO_BASELINE
};

struct FrameTelemetry {
    FrameTelemetry();

    std::size_t raw_keypoints;
    std::size_t used_keypoints;
    std::size_t removed_dynamic;
    std::size_t retained_uncertain;
    std::size_t removed_uncertain;
    bool semantic_accessed;
    SemanticState semantic_state;
    double cache_load_seconds;
    double policy_seconds;
    double pacing_lateness_seconds;
    double ipc_call_seconds;
    std::string ipc_reason;
    bool request_attempted;
    bool request_sent;
    double packet_age_ms;
    double inference_ms;
    std::size_t strong_track_count;
    std::size_t unconfirmed_track_count;
};

struct PacingDecision {
    double sleep_seconds;
    double lateness_seconds;
};

PacingDecision decidePacing(double interval_seconds, double elapsed_seconds);

const char* semanticStateName(SemanticState state);
std::string telemetryCsvHeader();
std::string formatTelemetryCsv(std::size_t frame_index, double timestamp,
                               int tracking_state, bool pose_valid,
                               double tracking_time_seconds,
                               const FrameTelemetry& telemetry);

}  // namespace semantic
}  // namespace ORB_SLAM2

#endif
