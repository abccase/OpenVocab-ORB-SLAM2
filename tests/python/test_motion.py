from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from semantic_py.openvocab_slam.config import DynamicConfirmationConfig
from semantic_py.openvocab_slam.motion import DynamicTrack


TRACK_FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures/synthetic_tracks/motion_sequences.json").read_text(
        encoding="utf-8"
    )
)


def test_frozen_dynamic_config_has_literal_reproducibility_parameters() -> None:
    # Catches a hidden runtime default changing cache identity or causal scores.
    assert DynamicConfirmationConfig.frozen().to_primitive() == {
        "schema": "ovorb.dynamic-cache.v1",
        "min_valid_depth_pixels": 100,
        "min_confirming_observations": 3,
        "base_motion_threshold_m": 0.10,
        "robust_sigma_multiplier": 3.0,
        "dynamic_enter_threshold": 0.70,
        "dynamic_exit_threshold": 0.40,
        "unknown_dynamic_probability": 0.25,
        "uncertain_retention_fraction": 0.50,
        "max_track_misses": 5,
        "centroid_3d_weight": 0.55,
        "mask_iou_weight": 0.30,
        "label_weight": 0.15,
        "association_gate_m": 1.0,
        "depth_scale": 5000.0,
        "kalman_initial_position_variance": 0.01,
        "kalman_initial_velocity_variance": 1.0,
        "kalman_process_acceleration_variance": 0.01,
        "kalman_measurement_variance": 0.0001,
        "dynamic_evidence_increment": 0.35,
        "static_evidence_decrement": 0.20,
        "diagnostic_sample_limit": 512,
        "diagnostic_fractions": [0.25, 0.50, 0.75],
    }


def test_static_person_is_not_strongly_filtered() -> None:
    # Catches static camera-compensated observations accumulating dynamic evidence.
    cfg = DynamicConfirmationConfig.frozen()
    centroids = [np.array(value) for value in TRACK_FIXTURE["static_world_centroids_m"]]
    item = DynamicTrack.new(7, "person", centroids[0], timestamp=0.0)
    states = [
        item.update(centroid, timestamp=float(frame), mad=np.zeros(3), config=cfg)
        for frame, centroid in enumerate(centroids[1:], 1)
    ]

    assert all(state.dynamic_probability < 0.70 and not state.strong_dynamic for state in states)


def test_track_initial_covariance_uses_the_bound_config() -> None:
    # Catches a hidden call to frozen defaults when a manifest-bound config is supplied.
    cfg = replace(
        DynamicConfirmationConfig.frozen(),
        kalman_initial_position_variance=0.25,
        kalman_initial_velocity_variance=2.0,
    )
    item = DynamicTrack.new(
        8,
        "person",
        np.array([0.0, 0.0, 2.0]),
        timestamp=0.0,
        config=cfg,
    )

    np.testing.assert_allclose(np.diag(item.covariance), [0.25, 0.25, 0.25, 2.0, 2.0, 2.0])


def test_motion_evidence_compares_measurement_with_kalman_prediction() -> None:
    # Catches using displacement from the last measurement instead of innovation.
    cfg = DynamicConfirmationConfig.frozen()
    item = DynamicTrack.new(18, "person", np.array([0.0, 0.0, 2.0]), timestamp=0.0)
    item.state[3] = 1.0

    state = item.update(
        np.array([1.0, 0.0, 2.0]),
        timestamp=1.0,
        mad=np.zeros(3),
        config=cfg,
    )

    assert state.confirming_observations == 0
    assert state.dynamic_probability == 0.0


def test_new_moving_track_waits_for_three_observations() -> None:
    # Catches strong filtering on a first or second displacement observation.
    cfg = DynamicConfirmationConfig.frozen()
    centroids = [np.array(value) for value in TRACK_FIXTURE["moving_world_centroids_m"]]
    item = DynamicTrack.new(9, "person", centroids[0], timestamp=0.0)
    states = [
        item.update(centroid, timestamp=float(frame), mad=np.zeros(3), config=cfg)
        for frame, centroid in enumerate(centroids[1:], 1)
    ]

    assert states[0].strong_dynamic is False
    assert states[1].strong_dynamic is False
    assert states[2].strong_dynamic is True


def test_hysteresis_holds_then_exits_only_below_literal_exit_threshold() -> None:
    # Catches using one threshold for both entry and exit.
    cfg = DynamicConfirmationConfig.frozen()
    centroids = [np.array(value) for value in TRACK_FIXTURE["moving_world_centroids_m"]]
    item = DynamicTrack.new(10, "person", centroids[0], timestamp=0.0)
    for frame, centroid in enumerate(centroids[1:], 1):
        state = item.update(
            centroid,
            timestamp=float(frame),
            mad=np.zeros(3),
            config=cfg,
        )
    assert state.strong_dynamic is True
    predicted = item.state[:3] + item.state[3:]
    state = item.update(predicted, timestamp=4.0, mad=np.zeros(3), config=cfg)
    assert state.dynamic_probability == 0.80
    assert state.strong_dynamic is True
    for frame in range(5, 8):
        predicted = item.state[:3] + item.state[3:]
        state = item.update(
            predicted,
            timestamp=float(frame),
            mad=np.zeros(3),
            config=cfg,
        )
    assert state.dynamic_probability == 0.20
    assert state.strong_dynamic is False
