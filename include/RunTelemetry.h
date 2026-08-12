#ifndef RUN_TELEMETRY_H
#define RUN_TELEMETRY_H

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>

namespace ORB_SLAM2 {

inline std::string FormatFrameTelemetry(std::size_t frame_index,
                                        double timestamp,
                                        int tracking_state,
                                        bool pose_valid,
                                        double tracking_time_seconds) {
    std::ostringstream output;
    output << std::setprecision(15)
           << "{\"frame_index\":" << frame_index
           << ",\"timestamp\":" << timestamp
           << ",\"tracking_state\":" << tracking_state
           << ",\"pose_valid\":" << (pose_valid ? "true" : "false")
           << ",\"tracking_time_seconds\":" << tracking_time_seconds
           << "}";
    return output.str();
}

}  // namespace ORB_SLAM2

#endif  // RUN_TELEMETRY_H
