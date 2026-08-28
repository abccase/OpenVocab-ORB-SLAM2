/** Non-blocking P06 online semantic-feedback TUM RGB-D runner. */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <opencv2/imgcodecs.hpp>

#include "System.h"
#include "semantic/FeatureMaskPolicy.h"
#include "semantic/IpcMaskProvider.h"
#include "semantic/OnlineDynamicState.h"
#include "semantic/Telemetry.h"

namespace {

struct AssociatedFrame {
    double timestamp;
    std::string rgb_path;
    std::string depth_path;
};

struct CompletedFrame {
    std::uint64_t frame_id;
    cv::Mat depth;
    cv::Mat T_world_camera;
};

struct OnlineCounters {
    OnlineCounters()
        : request_attempts(0), requests_sent(0), unique_packets(0), online_frames(0),
          degraded_frames(0), packet_age_sum_ms(0.0), inference_sum_ms(0.0),
          maximum_ipc_call_ms(0.0) {}
    std::size_t request_attempts;
    std::size_t requests_sent;
    std::size_t unique_packets;
    std::size_t online_frames;
    std::size_t degraded_frames;
    double packet_age_sum_ms;
    double inference_sum_ms;
    double maximum_ipc_call_ms;
};

std::vector<AssociatedFrame> LoadAssociations(const std::string& path) {
    std::ifstream stream(path.c_str());
    if (!stream) throw std::runtime_error("cannot open association file: " + path);
    std::vector<AssociatedFrame> frames;
    std::string line;
    double previous = -std::numeric_limits<double>::infinity();
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        AssociatedFrame frame;
        double depth_timestamp = 0.0;
        if (!(row >> frame.timestamp >> frame.rgb_path >> depth_timestamp >> frame.depth_path))
            throw std::runtime_error("malformed association row");
        std::string trailing;
        if (row >> trailing) throw std::runtime_error("association row has trailing fields");
        if (!std::isfinite(frame.timestamp) || frame.timestamp <= previous)
            throw std::runtime_error("association timestamps must be finite and increasing");
        previous = frame.timestamp;
        frames.push_back(frame);
    }
    if (frames.empty()) throw std::runtime_error("association file has no frames");
    return frames;
}

void WriteAtomic(const std::string& path, const std::string& contents) {
    const std::string temporary = "." + path + ".partial";
    {
        std::ofstream stream(temporary.c_str(), std::ios::out | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot open output: " + temporary);
        stream << contents;
        stream.flush();
        if (!stream) throw std::runtime_error("cannot write output: " + temporary);
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
        throw std::runtime_error("cannot publish output: " + path);
}

const CompletedFrame* FindCompleted(const std::deque<CompletedFrame>& history,
                                    std::uint64_t frame_id) {
    for (std::deque<CompletedFrame>::const_reverse_iterator item = history.rbegin();
         item != history.rend(); ++item)
        if (item->frame_id == frame_id) return &*item;
    return NULL;
}

ORB_SLAM2::semantic::OnlineCamera LoadCamera(const std::string& settings_path,
                                             double* depth_scale) {
    cv::FileStorage settings(settings_path, cv::FileStorage::READ);
    if (!settings.isOpened()) throw std::runtime_error("cannot open camera settings");
    ORB_SLAM2::semantic::OnlineCamera camera;
    camera.fx = static_cast<double>(settings["Camera.fx"]);
    camera.fy = static_cast<double>(settings["Camera.fy"]);
    camera.cx = static_cast<double>(settings["Camera.cx"]);
    camera.cy = static_cast<double>(settings["Camera.cy"]);
    *depth_scale = static_cast<double>(settings["DepthMapFactor"]);
    if (!std::isfinite(*depth_scale) || *depth_scale <= 0.0) *depth_scale = 1.0;
    return camera;
}

double ParsePositiveDouble(const char* raw, const std::string& label) {
    const std::string text(raw);
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid " + label);
    }
    if (consumed != text.size() || !std::isfinite(value) || value <= 0.0)
        throw std::runtime_error("invalid " + label);
    return value;
}

void WriteSummary(const OnlineCounters& counters, std::size_t frame_count,
                  double wall_seconds, const std::string& final_state,
                  const std::string& error, double request_rate_cap_hz,
                  double max_mask_age_ms) {
    const double semantic_hz = wall_seconds > 0.0 ? counters.unique_packets / wall_seconds : 0.0;
    const double drop_fraction = counters.request_attempts > 0
        ? 1.0 - static_cast<double>(counters.unique_packets) / counters.request_attempts : 0.0;
    const double degraded_fraction = frame_count > 0
        ? static_cast<double>(counters.degraded_frames) / frame_count : 0.0;
    const double mean_age = counters.unique_packets > 0
        ? counters.packet_age_sum_ms / counters.unique_packets : -1.0;
    const double mean_inference = counters.unique_packets > 0
        ? counters.inference_sum_ms / counters.unique_packets : -1.0;
    std::ostringstream output;
    output << std::setprecision(17)
           << "{\n  \"actual_semantic_hz\": " << semantic_hz
           << ",\n  \"degraded_frame_fraction\": " << degraded_fraction
           << ",\n  \"degraded_frames\": " << counters.degraded_frames
           << ",\n  \"error\": ";
    if (error.empty()) output << "null";
    else output << "\"" << error << "\"";
    output << ",\n  \"final_state\": \"" << final_state
           << "\",\n  \"frame_count\": " << frame_count
           << ",\n  \"maximum_ipc_call_ms\": " << counters.maximum_ipc_call_ms
           << ",\n  \"mean_inference_ms\": " << mean_inference
           << ",\n  \"mean_packet_age_ms\": " << mean_age
           << ",\n  \"online_frames\": " << counters.online_frames
           << ",\n  \"request_rate_cap_hz\": " << request_rate_cap_hz
           << ",\n  \"max_mask_age_ms\": " << max_mask_age_ms
           << ",\n  \"request_attempts\": " << counters.request_attempts
           << ",\n  \"request_drop_fraction\": " << std::max(0.0, drop_fraction)
           << ",\n  \"requests_sent\": " << counters.requests_sent
           << ",\n  \"unique_valid_packets\": " << counters.unique_packets
           << ",\n  \"wall_seconds\": " << wall_seconds << "\n}\n";
    WriteAtomic("online_summary.json", output.str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 14) {
        std::cerr << "Usage: rgbd_tum_online vocabulary settings sequence association "
                     "sequence_id seed run_id prompt_sha256 model_manifest_sha256 "
                     "request_endpoint result_endpoint request_rate_cap_hz "
                     "max_mask_age_ms\n";
        return 1;
    }
    std::size_t completed_frames = 0;
    OnlineCounters counters;
    double request_rate_cap_hz = 0.0;
    double max_mask_age_ms = 0.0;
    const std::chrono::steady_clock::time_point run_start = std::chrono::steady_clock::now();
    try {
        const std::string sequence_root = argv[3];
        const std::uint64_t seed = ORB_SLAM2::semantic::parsePolicySeed(argv[6]);
        const std::vector<AssociatedFrame> frames = LoadAssociations(argv[4]);
        double depth_scale = 5000.0;
        ORB_SLAM2::semantic::OnlineCamera camera = LoadCamera(argv[2], &depth_scale);
        request_rate_cap_hz = ParsePositiveDouble(argv[12], "request rate cap");
        max_mask_age_ms = ParsePositiveDouble(argv[13], "maximum mask age");
        if (request_rate_cap_hz > 5.0)
            throw std::runtime_error("request rate cap exceeds 5 Hz");
        if (max_mask_age_ms >
            static_cast<double>(std::numeric_limits<std::uint64_t>::max()) / 1000000.0)
            throw std::runtime_error("maximum mask age is too large");
        ORB_SLAM2::semantic::IpcProviderConfig provider_config;
        provider_config.run_id = argv[7];
        provider_config.prompt_sha256 = argv[8];
        provider_config.model_manifest_sha256 = argv[9];
        provider_config.request_rate_cap_hz = request_rate_cap_hz;
        provider_config.max_age_ns = static_cast<std::uint64_t>(
            std::llround(max_mask_age_ms * 1000000.0));
        ORB_SLAM2::semantic::SystemIpcClock clock;
        ORB_SLAM2::semantic::IpcMaskProvider provider(
            provider_config,
            ORB_SLAM2::semantic::makeZmqIpcTransport(argv[10], argv[11]), clock);
        ORB_SLAM2::semantic::OnlineDynamicConfig dynamic_config;
        dynamic_config.depth_scale = depth_scale;
        dynamic_config.max_prediction_age_ns = provider_config.max_age_ns;
        ORB_SLAM2::semantic::OnlineDynamicState dynamic_state(dynamic_config);
        bool dynamic_state_ready = false;
        bool has_applied_packet = false;
        std::uint64_t last_applied_packet = 0;
        std::deque<CompletedFrame> history;

        std::ofstream telemetry("frame_telemetry.csv", std::ios::out | std::ios::trunc);
        if (!telemetry) throw std::runtime_error("cannot open frame_telemetry.csv");
        telemetry << ORB_SLAM2::semantic::telemetryCsvHeader() << '\n';
        ORB_SLAM2::System slam(argv[1], argv[2], ORB_SLAM2::System::RGBD, true);
        std::string failure;

        for (std::size_t index = 0; index < frames.size(); ++index) {
            const AssociatedFrame& frame = frames[index];
            cv::Mat rgb = cv::imread(sequence_root + "/" + frame.rgb_path, cv::IMREAD_COLOR);
            cv::Mat depth = cv::imread(sequence_root + "/" + frame.depth_path, cv::IMREAD_UNCHANGED);
            if (rgb.empty() || depth.empty()) {
                failure = "RGB or depth image is missing";
                break;
            }
            if (index == 0) {
                camera.image_width = rgb.cols;
                camera.image_height = rgb.rows;
            } else if (rgb.cols != camera.image_width || rgb.rows != camera.image_height) {
                failure = "online stream dimensions changed";
                break;
            }
            const std::chrono::steady_clock::time_point frame_start =
                std::chrono::steady_clock::now();
            std::vector<unsigned char> jpeg;
            const std::vector<int> jpeg_options{cv::IMWRITE_JPEG_QUALITY, 85};
            if (!cv::imencode(".jpg", rgb, jpeg, jpeg_options)) {
                failure = "JPEG request encoding failed";
                break;
            }
            const std::uint64_t timestamp_ns = static_cast<std::uint64_t>(
                std::llround(frame.timestamp * 1000000000.0));
            ORB_SLAM2::semantic::IpcPollResult ipc = provider.poll(
                index, timestamp_ns, jpeg, rgb.cols, rgb.rows);
            if (ipc.request_attempted) ++counters.request_attempts;
            if (ipc.request_sent) ++counters.requests_sent;
            counters.maximum_ipc_call_ms = std::max(
                counters.maximum_ipc_call_ms, ipc.call_duration_ms);

            if (ipc.has_packet &&
                (!has_applied_packet || ipc.packet.frame_id > last_applied_packet)) {
                const CompletedFrame* source = FindCompleted(history, ipc.packet.frame_id);
                if (source) {
                    dynamic_state.updateCompletedFrame(
                        ipc.packet, source->depth, source->T_world_camera, camera);
                    dynamic_state_ready = true;
                    has_applied_packet = true;
                    last_applied_packet = ipc.packet.frame_id;
                    ++counters.unique_packets;
                    counters.packet_age_sum_ms += ipc.packet.age_ns / 1000000.0;
                    counters.inference_sum_ms += ipc.packet.inference_ms;
                } else if (ipc.packet.frame_id < index) {
                    ipc.state = ORB_SLAM2::semantic::SemanticState::DEGRADED_TO_BASELINE;
                    ipc.reason = "SOURCE_FRAME_NOT_AVAILABLE";
                }
            }

            ORB_SLAM2::semantic::OnlinePrediction prediction;
            ORB_SLAM2::semantic::DynamicScoreMap* score_map = NULL;
            if (dynamic_state_ready && ipc.state == ORB_SLAM2::semantic::SemanticState::ONLINE_VALID &&
                !history.empty()) {
                prediction = dynamic_state.predict(
                    timestamp_ns, history.back().T_world_camera, camera);
                if (prediction.reason == "ONLINE_PREDICTION_VALID") {
                    prediction.score_map.source_timestamp = frame.timestamp;
                    prediction.score_map.policy_seed = seed;
                    score_map = &prediction.score_map;
                } else {
                    ipc.state = ORB_SLAM2::semantic::SemanticState::DEGRADED_TO_BASELINE;
                    ipc.reason = prediction.reason;
                }
            }

            ORB_SLAM2::semantic::FrameTelemetry frame_telemetry;
            const std::chrono::steady_clock::time_point tracking_start =
                std::chrono::steady_clock::now();
            cv::Mat T_camera_world = slam.TrackRGBD(
                rgb, depth, frame.timestamp, score_map, &frame_telemetry);
            const double tracking_seconds =
                std::chrono::duration_cast<std::chrono::duration<double> >(
                    std::chrono::steady_clock::now() - tracking_start).count();
            frame_telemetry.semantic_accessed = true;
            frame_telemetry.semantic_state = ipc.state;
            frame_telemetry.ipc_call_seconds = ipc.call_duration_ms / 1000.0;
            frame_telemetry.ipc_reason = ipc.reason;
            frame_telemetry.request_attempted = ipc.request_attempted;
            frame_telemetry.request_sent = ipc.request_sent;
            if (ipc.has_packet) {
                frame_telemetry.packet_age_ms = ipc.packet.age_ns / 1000000.0;
                frame_telemetry.inference_ms = ipc.packet.inference_ms;
            }
            frame_telemetry.strong_track_count = prediction.strong_track_count;
            frame_telemetry.unconfirmed_track_count = prediction.unconfirmed_track_count;
            if (ipc.state == ORB_SLAM2::semantic::SemanticState::ONLINE_VALID)
                ++counters.online_frames;
            else
                ++counters.degraded_frames;

            if (!T_camera_world.empty()) {
                CompletedFrame completed;
                completed.frame_id = index;
                completed.depth = depth.clone();
                completed.T_world_camera = T_camera_world.inv();
                history.push_back(completed);
                while (history.size() > 64) history.pop_front();
            }
            double interval = 0.0;
            if (index + 1 < frames.size()) interval = frames[index + 1].timestamp - frame.timestamp;
            else if (index > 0) interval = frame.timestamp - frames[index - 1].timestamp;
            const double elapsed = std::chrono::duration_cast<std::chrono::duration<double> >(
                std::chrono::steady_clock::now() - frame_start).count();
            const ORB_SLAM2::semantic::PacingDecision pacing =
                ORB_SLAM2::semantic::decidePacing(interval, elapsed);
            frame_telemetry.pacing_lateness_seconds = pacing.lateness_seconds;
            telemetry << ORB_SLAM2::semantic::formatTelemetryCsv(
                index, frame.timestamp, slam.GetTrackingState(), !T_camera_world.empty(),
                tracking_seconds, frame_telemetry) << '\n';
            telemetry.flush();
            if (!telemetry) {
                failure = "cannot write online telemetry";
                break;
            }
            ++completed_frames;
            if (pacing.sleep_seconds > 0.0)
                usleep(static_cast<useconds_t>(pacing.sleep_seconds * 1000000.0));
        }

        slam.Shutdown();
        const double wall_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(
            std::chrono::steady_clock::now() - run_start).count();
        if (!failure.empty()) {
            WriteSummary(counters, completed_frames, wall_seconds, "FAILED", failure,
                         request_rate_cap_hz, max_mask_age_ms);
            std::cerr << failure << '\n';
            return 2;
        }
        slam.SaveTrajectoryTUM("CameraTrajectory.txt");
        slam.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
        WriteSummary(counters, completed_frames, wall_seconds, "COMPLETED", "",
                     request_rate_cap_hz, max_mask_age_ms);
        return 0;
    } catch (const std::exception& error) {
        const double wall_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(
            std::chrono::steady_clock::now() - run_start).count();
        try {
            WriteSummary(counters, completed_frames, wall_seconds, "FAILED", error.what(),
                         request_rate_cap_hz, max_mask_age_ms);
        }
        catch (...) {}
        std::cerr << error.what() << '\n';
        return 2;
    }
}
