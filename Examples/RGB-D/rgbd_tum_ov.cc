/**
 * Offline dual-mode TUM RGB-D runner for causal semantic feedback.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <opencv2/imgcodecs.hpp>

#include "System.h"
#include "semantic/MaskProvider.h"
#include "semantic/FeatureMaskPolicy.h"
#include "semantic/Telemetry.h"

namespace {

using std::string;
using std::vector;

struct AssociatedFrame {
    double timestamp;
    string rgb_path;
    string depth_path;
};

string JsonEscape(const string& value) {
    std::ostringstream output;
    for (std::size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (c == '"' || c == '\\') output << '\\' << c;
        else if (c == '\n') output << "\\n";
        else if (c == '\r') output << "\\r";
        else if (c == '\t') output << "\\t";
        else if (c < 0x20) output << "?";
        else output << c;
    }
    return output.str();
}

void WriteAtomic(const string& path, const string& contents) {
    const string temporary = "." + path + ".partial";
    {
        std::ofstream stream(temporary.c_str(), std::ios::out | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot open output: " + temporary);
        stream << contents;
        stream.flush();
        if (!stream) throw std::runtime_error("cannot write output: " + temporary);
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
        throw std::runtime_error("cannot atomically publish output: " + path);
}

void WriteFinalState(const string& state, const string& mode,
                     std::size_t frame_count, const string& error) {
    std::ostringstream json;
    json << "{\n  \"error\": ";
    if (error.empty()) json << "null";
    else json << "\"" << JsonEscape(error) << "\"";
    json << ",\n  \"frame_count\": " << frame_count
         << ",\n  \"mode\": \"" << mode
         << "\",\n  \"state\": \"" << state << "\"\n}\n";
    WriteAtomic("final_state.json", json.str());
}

vector<AssociatedFrame> LoadAssociations(const string& path) {
    std::ifstream stream(path.c_str());
    if (!stream) throw std::runtime_error("cannot open association file: " + path);
    vector<AssociatedFrame> frames;
    string line;
    std::size_t line_number = 0;
    double previous = -std::numeric_limits<double>::infinity();
    while (std::getline(stream, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        AssociatedFrame frame;
        double depth_timestamp = 0.0;
        if (!(row >> frame.timestamp >> frame.rgb_path >> depth_timestamp >> frame.depth_path))
            throw std::runtime_error("malformed association row " + std::to_string(line_number));
        string trailing;
        if (row >> trailing)
            throw std::runtime_error("association row has trailing fields " + std::to_string(line_number));
        if (!std::isfinite(frame.timestamp) || frame.timestamp <= previous)
            throw std::runtime_error("association timestamps are not finite and increasing");
        previous = frame.timestamp;
        frames.push_back(frame);
    }
    if (frames.empty()) throw std::runtime_error("association file has no frames");
    return frames;
}

void WriteTimings(const vector<double>& tracking, double wall_seconds) {
    vector<double> sorted = tracking;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (std::size_t i = 0; i < tracking.size(); ++i) sum += tracking[i];
    const double mean = tracking.empty() ? 0.0 : sum / tracking.size();
    const double median = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
    std::ostringstream json;
    json << std::setprecision(15)
         << "{\n  \"frame_count\": " << tracking.size()
         << ",\n  \"mean_tracking_seconds\": " << mean
         << ",\n  \"median_tracking_seconds\": " << median
         << ",\n  \"wall_seconds\": " << wall_seconds << "\n}\n";
    WriteAtomic("timings.json", json.str());
}

}  // namespace

int main(int argc, char** argv) {
    const bool baseline_shape = argc == 8;
    const bool semantic_shape = argc == 12;
    if (!baseline_shape && !semantic_shape) {
        std::cerr << "Usage: rgbd_tum_ov vocabulary settings sequence association "
                     "baseline sequence_id seed\n"
                     "   or: rgbd_tum_ov vocabulary settings sequence association "
                     "semantic-feedback sequence_id seed cache_root manifest_sha256 "
                     "completion_sha256 index_sha256\n";
        return 1;
    }

    const string mode = argv[5];
    const bool semantic = mode == "semantic-feedback";
    if ((mode == "baseline" && !baseline_shape) || (semantic && !semantic_shape) ||
        (mode != "baseline" && !semantic)) {
        std::cerr << "mode and argument shape do not match\n";
        return 1;
    }

    std::size_t completed_frames = 0;
    try {
        const string sequence_root = argv[3];
        const string sequence_id = argv[6];
        const std::uint64_t seed = ORB_SLAM2::semantic::parsePolicySeed(argv[7]);
        const vector<AssociatedFrame> frames = LoadAssociations(argv[4]);

        // This object is never constructed in baseline mode.
        std::unique_ptr<ORB_SLAM2::semantic::CacheMaskProvider> provider;
        if (semantic) {
            provider.reset(new ORB_SLAM2::semantic::CacheMaskProvider(
                argv[8], sequence_id, argv[9], argv[10], argv[11]));
        }

        std::ofstream telemetry("frame_telemetry.csv", std::ios::out | std::ios::trunc);
        if (!telemetry) throw std::runtime_error("cannot open frame_telemetry.csv");
        telemetry << ORB_SLAM2::semantic::telemetryCsvHeader() << '\n';

        ORB_SLAM2::System slam(argv[1], argv[2], ORB_SLAM2::System::RGBD, true);
        vector<double> tracking_times;
        tracking_times.reserve(frames.size());
        const std::chrono::steady_clock::time_point run_start =
            std::chrono::steady_clock::now();
        string failure;

        for (std::size_t i = 0; i < frames.size(); ++i) {
            const AssociatedFrame& frame = frames[i];
            const cv::Mat rgb = cv::imread(sequence_root + "/" + frame.rgb_path,
                                           cv::IMREAD_UNCHANGED);
            const cv::Mat depth = cv::imread(sequence_root + "/" + frame.depth_path,
                                             cv::IMREAD_UNCHANGED);
            if (rgb.empty() || depth.empty()) {
                failure = "RGB or depth image is missing at frame " + std::to_string(i);
                break;
            }

            ORB_SLAM2::semantic::DynamicScoreMap score_map;
            const ORB_SLAM2::semantic::DynamicScoreMap* score_pointer = NULL;
            double cache_seconds = 0.0;
            if (semantic) {
                try {
                    const std::chrono::steady_clock::time_point cache_start =
                        std::chrono::steady_clock::now();
                    score_map = provider->load(frame.timestamp, rgb.cols, rgb.rows);
                    cache_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(
                        std::chrono::steady_clock::now() - cache_start).count();
                    score_map.policy_seed = static_cast<std::uint64_t>(seed);
                    score_pointer = &score_map;
                } catch (const std::exception& error) {
                    failure = string("cache validation failed before tracking frame ") +
                              std::to_string(i) + ": " + error.what();
                    break;
                }
            }

            ORB_SLAM2::semantic::FrameTelemetry frame_telemetry;
            const std::chrono::steady_clock::time_point track_start =
                std::chrono::steady_clock::now();
            const cv::Mat pose = slam.TrackRGBD(rgb, depth, frame.timestamp,
                                                score_pointer, &frame_telemetry);
            const double tracking_seconds =
                std::chrono::duration_cast<std::chrono::duration<double> >(
                    std::chrono::steady_clock::now() - track_start).count();
            frame_telemetry.cache_load_seconds = cache_seconds;
            telemetry << ORB_SLAM2::semantic::formatTelemetryCsv(
                i, frame.timestamp, slam.GetTrackingState(), !pose.empty(),
                tracking_seconds, frame_telemetry) << '\n';
            telemetry.flush();
            if (!telemetry) {
                failure = "cannot write frame telemetry";
                break;
            }
            tracking_times.push_back(tracking_seconds);
            ++completed_frames;

            double interval = 0.0;
            if (i + 1 < frames.size()) interval = frames[i + 1].timestamp - frame.timestamp;
            else if (i > 0) interval = frame.timestamp - frames[i - 1].timestamp;
            if (tracking_seconds < interval)
                usleep(static_cast<useconds_t>((interval - tracking_seconds) * 1e6));
        }

        slam.Shutdown();
        const double wall_seconds =
            std::chrono::duration_cast<std::chrono::duration<double> >(
                std::chrono::steady_clock::now() - run_start).count();
        WriteTimings(tracking_times, wall_seconds);
        if (!failure.empty()) {
            WriteFinalState("FAILED", mode, completed_frames, failure);
            std::cerr << failure << '\n';
            return 2;
        }
        slam.SaveTrajectoryTUM("CameraTrajectory.txt");
        slam.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
        WriteFinalState("COMPLETED", mode, completed_frames, "");
        return 0;
    } catch (const std::exception& error) {
        try {
            WriteFinalState("FAILED", mode, completed_frames, error.what());
        } catch (...) {
        }
        std::cerr << error.what() << '\n';
        return 2;
    }
}
