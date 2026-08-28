#include "semantic/OnlineDynamicState.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <opencv2/video/tracking.hpp>

namespace ORB_SLAM2 {
namespace semantic {
namespace {

bool finiteVector(const cv::Vec3d& value) {
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("median requires values");
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    if (values.size() % 2 == 1) return values[middle];
    const double upper = values[middle];
    std::nth_element(values.begin(), values.begin() + middle - 1, values.begin() + middle);
    return (values[middle - 1] + upper) / 2.0;
}

double norm(const cv::Vec3d& value) {
    return std::sqrt(value.dot(value));
}

cv::Vec3d matrixPosition(const cv::Mat& state) {
    return cv::Vec3d(state.at<double>(0), state.at<double>(1), state.at<double>(2));
}

struct Track {
    Track(int identity, const OnlineObservation& observation, std::uint64_t timestamp)
        : id(identity), label(observation.label), filter(6, 3, 0, CV_64F),
          timestamp_ns(timestamp), last_observation_ns(timestamp),
          confirming_observations(0), dynamic_probability(0.0), strong_dynamic(false),
          misses(0), box_width(observation.box_width_pixels),
          box_height(observation.box_height_pixels), predicted_displacement(0.0) {
        filter.transitionMatrix = cv::Mat::eye(6, 6, CV_64F);
        filter.measurementMatrix = cv::Mat::zeros(3, 6, CV_64F);
        for (int axis = 0; axis < 3; ++axis) filter.measurementMatrix.at<double>(axis, axis) = 1.0;
        cv::setIdentity(filter.processNoiseCov, cv::Scalar::all(0.01));
        cv::setIdentity(filter.measurementNoiseCov, cv::Scalar::all(0.0001));
        filter.errorCovPost = cv::Mat::zeros(6, 6, CV_64F);
        for (int axis = 0; axis < 3; ++axis) filter.errorCovPost.at<double>(axis, axis) = 0.01;
        for (int axis = 3; axis < 6; ++axis) filter.errorCovPost.at<double>(axis, axis) = 1.0;
        filter.statePost = cv::Mat::zeros(6, 1, CV_64F);
        for (int axis = 0; axis < 3; ++axis)
            filter.statePost.at<double>(axis) = observation.centroid_world[axis];
    }
    cv::Vec3d predictTo(std::uint64_t timestamp) {
        const double dt = (timestamp - timestamp_ns) / 1000000000.0;
        const cv::Vec3d previous = matrixPosition(filter.statePost);
        filter.transitionMatrix = cv::Mat::eye(6, 6, CV_64F);
        for (int axis = 0; axis < 3; ++axis)
            filter.transitionMatrix.at<double>(axis, axis + 3) = dt;
        cv::setIdentity(filter.processNoiseCov, cv::Scalar::all(0.0));
        const double acceleration_variance = 0.01;
        for (int axis = 0; axis < 3; ++axis) {
            filter.processNoiseCov.at<double>(axis, axis) = dt * dt * dt * dt * acceleration_variance / 4.0;
            filter.processNoiseCov.at<double>(axis, axis + 3) = dt * dt * dt * acceleration_variance / 2.0;
            filter.processNoiseCov.at<double>(axis + 3, axis) = dt * dt * dt * acceleration_variance / 2.0;
            filter.processNoiseCov.at<double>(axis + 3, axis + 3) = dt * dt * acceleration_variance;
        }
        timestamp_ns = timestamp;
        const cv::Vec3d predicted = matrixPosition(filter.predict());
        predicted_displacement = norm(predicted - previous);
        return predicted;
    }
    void correct(const OnlineObservation& observation, const cv::Vec3d& predicted,
                 const OnlineDynamicConfig& config) {
        const double residual = norm(observation.centroid_world - predicted);
        const double threshold = std::max(
            config.base_motion_threshold_m,
            config.robust_sigma_multiplier * norm(observation.mad_world));
        cv::Mat measurement(3, 1, CV_64F);
        for (int axis = 0; axis < 3; ++axis)
            measurement.at<double>(axis) = observation.centroid_world[axis];
        filter.correct(measurement);
        last_observation_ns = timestamp_ns;
        box_width = observation.box_width_pixels;
        box_height = observation.box_height_pixels;
        misses = 0;
        if (std::max(residual, predicted_displacement) > threshold) {
            ++confirming_observations;
            dynamic_probability = std::min(
                1.0, dynamic_probability + config.dynamic_evidence_increment);
        } else {
            confirming_observations = 0;
            dynamic_probability = std::max(
                0.0, dynamic_probability - config.static_evidence_decrement);
        }
        if (!strong_dynamic) {
            strong_dynamic = dynamic_probability >= config.dynamic_enter_threshold &&
                confirming_observations >= config.min_confirming_observations;
        } else if (dynamic_probability < config.dynamic_exit_threshold) {
            strong_dynamic = false;
        }
    }
    cv::Vec3d projectedPosition(std::uint64_t timestamp) const {
        const double dt = (timestamp - timestamp_ns) / 1000000000.0;
        return cv::Vec3d(
            filter.statePost.at<double>(0) + filter.statePost.at<double>(3) * dt,
            filter.statePost.at<double>(1) + filter.statePost.at<double>(4) * dt,
            filter.statePost.at<double>(2) + filter.statePost.at<double>(5) * dt);
    }
    int id;
    std::string label;
    cv::KalmanFilter filter;
    std::uint64_t timestamp_ns;
    std::uint64_t last_observation_ns;
    int confirming_observations;
    double dynamic_probability;
    bool strong_dynamic;
    int misses;
    double box_width;
    double box_height;
    double predicted_displacement;
};

void validateConfig(const OnlineDynamicConfig& config) {
    if (config.min_confirming_observations <= 0 ||
        !std::isfinite(config.base_motion_threshold_m) || config.base_motion_threshold_m <= 0.0 ||
        !std::isfinite(config.robust_sigma_multiplier) || config.robust_sigma_multiplier <= 0.0 ||
        !std::isfinite(config.dynamic_enter_threshold) ||
        !std::isfinite(config.dynamic_exit_threshold) ||
        !(config.dynamic_exit_threshold < config.dynamic_enter_threshold) ||
        config.dynamic_enter_threshold > 1.0 || config.dynamic_exit_threshold < 0.0 ||
        !std::isfinite(config.dynamic_evidence_increment) || config.dynamic_evidence_increment <= 0.0 ||
        !std::isfinite(config.static_evidence_decrement) || config.static_evidence_decrement <= 0.0 ||
        !std::isfinite(config.association_gate_m) || config.association_gate_m <= 0.0 ||
        config.max_prediction_age_ns == 0)
        throw std::invalid_argument("online dynamic configuration is invalid");
    if (config.min_valid_depth_pixels <= 0 || !std::isfinite(config.depth_scale) ||
        config.depth_scale <= 0.0)
        throw std::invalid_argument("online depth configuration is invalid");
}

}  // namespace

class OnlineDynamicState::Impl {
public:
    explicit Impl(const OnlineDynamicConfig& value)
        : config(value), has_timestamp(false), last_timestamp_ns(0), next_track_id(0) {
        validateConfig(config);
    }

    OnlineDynamicConfig config;
    bool has_timestamp;
    std::uint64_t last_timestamp_ns;
    int next_track_id;
    std::vector<Track> tracks;
};

OnlineDynamicConfig::OnlineDynamicConfig()
    : min_confirming_observations(3), base_motion_threshold_m(0.10),
      robust_sigma_multiplier(3.0), dynamic_enter_threshold(0.70),
      dynamic_exit_threshold(0.40), dynamic_evidence_increment(0.35),
      static_evidence_decrement(0.20), association_gate_m(1.0),
      max_prediction_age_ns(250000000ULL), min_valid_depth_pixels(100),
      depth_scale(5000.0) {}

OnlineCamera::OnlineCamera()
    : fx(0.0), fy(0.0), cx(0.0), cy(0.0), image_width(0), image_height(0) {}

OnlineObservation::OnlineObservation()
    : centroid_world(0.0, 0.0, 0.0), mad_world(0.0, 0.0, 0.0),
      box_width_pixels(0.0), box_height_pixels(0.0) {}

OnlinePrediction::OnlinePrediction()
    : strong_track_count(0), unconfirmed_track_count(0) {}

OnlineDynamicState::OnlineDynamicState(const OnlineDynamicConfig& config)
    : impl_(new Impl(config)) {}

OnlineDynamicState::~OnlineDynamicState() {}

void OnlineDynamicState::updateCompletedFrame(
    const std::vector<OnlineObservation>& observations, std::uint64_t timestamp_ns) {
    if (impl_->has_timestamp && timestamp_ns < impl_->last_timestamp_ns)
        throw std::invalid_argument("completed frame timestamps must be causal");
    std::vector<cv::Vec3d> predicted(impl_->tracks.size());
    for (std::size_t index = 0; index < impl_->tracks.size(); ++index)
        predicted[index] = impl_->tracks[index].predictTo(timestamp_ns);
    std::vector<bool> matched(impl_->tracks.size(), false);
    for (std::size_t observation_index = 0; observation_index < observations.size();
         ++observation_index) {
        const OnlineObservation& observation = observations[observation_index];
        if (observation.label.empty() || !finiteVector(observation.centroid_world) ||
            !finiteVector(observation.mad_world) || observation.mad_world[0] < 0.0 ||
            observation.mad_world[1] < 0.0 || observation.mad_world[2] < 0.0 ||
            !std::isfinite(observation.box_width_pixels) || observation.box_width_pixels <= 0.0 ||
            !std::isfinite(observation.box_height_pixels) || observation.box_height_pixels <= 0.0)
            throw std::invalid_argument("online observation is invalid");
        std::size_t best = impl_->tracks.size();
        double best_distance = impl_->config.association_gate_m;
        for (std::size_t track_index = 0; track_index < impl_->tracks.size(); ++track_index) {
            if (matched[track_index] || impl_->tracks[track_index].label != observation.label)
                continue;
            const double distance = norm(observation.centroid_world - predicted[track_index]);
            if (distance <= best_distance) {
                best = track_index;
                best_distance = distance;
            }
        }
        if (best == impl_->tracks.size()) {
            impl_->tracks.push_back(Track(impl_->next_track_id++, observation, timestamp_ns));
            predicted.push_back(observation.centroid_world);
            matched.push_back(true);
        } else {
            matched[best] = true;
            impl_->tracks[best].correct(observation, predicted[best], impl_->config);
        }
    }
    for (std::size_t index = 0; index < matched.size() && index < impl_->tracks.size(); ++index) {
        if (!matched[index]) {
            ++impl_->tracks[index].misses;
            impl_->tracks[index].confirming_observations = 0;
            impl_->tracks[index].dynamic_probability = std::max(
                0.0, impl_->tracks[index].dynamic_probability - impl_->config.static_evidence_decrement);
            if (impl_->tracks[index].dynamic_probability < impl_->config.dynamic_exit_threshold)
                impl_->tracks[index].strong_dynamic = false;
        }
    }
    impl_->has_timestamp = true;
    impl_->last_timestamp_ns = timestamp_ns;
}

void OnlineDynamicState::updateCompletedFrame(
    const SemanticPacket& packet, const cv::Mat& depth,
    const cv::Mat& T_world_camera, const OnlineCamera& camera) {
    if (packet.image_width != camera.image_width ||
        packet.image_height != camera.image_height ||
        depth.cols != camera.image_width || depth.rows != camera.image_height)
        throw std::invalid_argument("online packet, depth, and camera dimensions mismatch");
    if (depth.type() != CV_32FC1 && depth.type() != CV_16UC1)
        throw std::invalid_argument("online depth must be CV_32FC1 metres or CV_16UC1 scaled");
    if (!std::isfinite(camera.fx) || camera.fx <= 0.0 ||
        !std::isfinite(camera.fy) || camera.fy <= 0.0)
        throw std::invalid_argument("online camera intrinsics are invalid");
    cv::Mat pose;
    T_world_camera.convertTo(pose, CV_64F);
    if (pose.rows != 4 || pose.cols != 4 || !cv::checkRange(pose))
        throw std::invalid_argument("completed pose must be finite 4x4");
    std::vector<OnlineObservation> observations;
    for (std::size_t instance_index = 0; instance_index < packet.instances.size();
         ++instance_index) {
        const OnlineInstance& instance = packet.instances[instance_index];
        std::vector<cv::Vec3d> points;
        std::size_t flat_index = 0;
        bool foreground = false;
        for (std::size_t run_index = 0; run_index < instance.mask_counts.size(); ++run_index) {
            const std::size_t run_end = flat_index + instance.mask_counts[run_index];
            if (foreground) {
                for (; flat_index < run_end; ++flat_index) {
                    const int row = static_cast<int>(flat_index % packet.image_height);
                    const int column = static_cast<int>(flat_index / packet.image_height);
                    const double z = depth.type() == CV_32FC1
                        ? static_cast<double>(depth.at<float>(row, column))
                        : static_cast<double>(depth.at<std::uint16_t>(row, column)) /
                            impl_->config.depth_scale;
                    if (!std::isfinite(z) || z <= 0.0) continue;
                    const double x = (column - camera.cx) * z / camera.fx;
                    const double y = (row - camera.cy) * z / camera.fy;
                    cv::Mat camera_point = (cv::Mat_<double>(4, 1) << x, y, z, 1.0);
                    cv::Mat world_point = pose * camera_point;
                    const cv::Vec3d point(
                        world_point.at<double>(0), world_point.at<double>(1),
                        world_point.at<double>(2));
                    if (finiteVector(point)) points.push_back(point);
                }
            } else {
                flat_index = run_end;
            }
            foreground = !foreground;
        }
        if (static_cast<int>(points.size()) < impl_->config.min_valid_depth_pixels)
            continue;
        OnlineObservation observation;
        observation.label = instance.label;
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<double> values;
            values.reserve(points.size());
            for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
                values.push_back(points[point_index][axis]);
            observation.centroid_world[axis] = median(values);
            for (std::size_t value_index = 0; value_index < values.size(); ++value_index)
                values[value_index] = std::abs(values[value_index] - observation.centroid_world[axis]);
            observation.mad_world[axis] = median(values);
        }
        observation.box_width_pixels = instance.box_xyxy[2] - instance.box_xyxy[0];
        observation.box_height_pixels = instance.box_xyxy[3] - instance.box_xyxy[1];
        observations.push_back(observation);
    }
    updateCompletedFrame(observations, packet.source_timestamp_ns);
}

OnlinePrediction OnlineDynamicState::predict(
    std::uint64_t target_timestamp_ns, const cv::Mat& last_T_world_camera,
    const OnlineCamera& camera) const {
    if (!std::isfinite(camera.fx) || camera.fx <= 0.0 ||
        !std::isfinite(camera.fy) || camera.fy <= 0.0 ||
        camera.image_width <= 0 || camera.image_height <= 0)
        throw std::invalid_argument("online camera is invalid");
    if (!impl_->has_timestamp || target_timestamp_ns < impl_->last_timestamp_ns)
        throw std::invalid_argument("online prediction timestamp is invalid");
    cv::Mat pose;
    last_T_world_camera.convertTo(pose, CV_64F);
    if (pose.rows != 4 || pose.cols != 4 || !cv::checkRange(pose))
        throw std::invalid_argument("last completed pose must be finite 4x4");
    OnlinePrediction prediction;
    prediction.score_map.source_timestamp = target_timestamp_ns / 1000000000.0;
    prediction.score_map.frame_key = target_timestamp_ns;
    prediction.score_map.scores_f32 = cv::Mat::zeros(
        camera.image_height, camera.image_width, CV_32FC1);
    if (target_timestamp_ns - impl_->last_timestamp_ns > impl_->config.max_prediction_age_ns) {
        prediction.reason = "PREDICTION_STALE";
        return prediction;
    }
    cv::Mat T_camera_world = pose.inv();
    for (std::size_t index = 0; index < impl_->tracks.size(); ++index) {
        const Track& track = impl_->tracks[index];
        if (target_timestamp_ns - track.last_observation_ns > impl_->config.max_prediction_age_ns)
            continue;
        const cv::Vec3d world = track.projectedPosition(target_timestamp_ns);
        cv::Mat point = (cv::Mat_<double>(4, 1) << world[0], world[1], world[2], 1.0);
        cv::Mat camera_point = T_camera_world * point;
        const double z = camera_point.at<double>(2);
        if (!std::isfinite(z) || z <= 0.0) continue;
        const double u = camera.fx * camera_point.at<double>(0) / z + camera.cx;
        const double v = camera.fy * camera_point.at<double>(1) / z + camera.cy;
        if (!std::isfinite(u) || !std::isfinite(v)) continue;
        const int x0 = std::max(0, static_cast<int>(std::floor(u - track.box_width / 2.0)));
        const int x1 = std::min(camera.image_width, static_cast<int>(std::ceil(u + track.box_width / 2.0)));
        const int y0 = std::max(0, static_cast<int>(std::floor(v - track.box_height / 2.0)));
        const int y1 = std::min(camera.image_height, static_cast<int>(std::ceil(v + track.box_height / 2.0)));
        if (x0 >= x1 || y0 >= y1) continue;
        const float score = track.strong_dynamic
            ? static_cast<float>(track.dynamic_probability) : 0.25f;
        cv::Mat region = prediction.score_map.scores_f32(cv::Rect(x0, y0, x1 - x0, y1 - y0));
        cv::max(region, score, region);
        if (track.strong_dynamic) ++prediction.strong_track_count;
        else ++prediction.unconfirmed_track_count;
    }
    prediction.reason = "ONLINE_PREDICTION_VALID";
    return prediction;
}

}  // namespace semantic
}  // namespace ORB_SLAM2
