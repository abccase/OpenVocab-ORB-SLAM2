from __future__ import annotations

import math

import numpy as np
import pytest

from semantic_py.openvocab_slam.experiments import compute_trajectory_metrics


def _pose(timestamp: float, xyz: tuple[float, float, float], yaw: float = 0.0) -> tuple[float, ...]:
    return (
        timestamp,
        *xyz,
        0.0,
        0.0,
        math.sin(yaw / 2.0),
        math.cos(yaw / 2.0),
    )


def test_identical_trajectory_has_zero_ate_and_rpe() -> None:
    trajectory = [_pose(0.0, (0.0, 0.0, 0.0)), _pose(1.0, (1.0, 0.0, 0.0)), _pose(2.0, (2.0, 1.0, 0.0))]
    result = compute_trajectory_metrics(trajectory, trajectory)
    assert result["associated_pose_count"] == 3
    assert result["ate_translation_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert result["rpe_translation_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert result["rpe_rotation_rmse_deg"] == pytest.approx(0.0, abs=1e-10)


def test_se3_alignment_removes_global_translation_and_rotation() -> None:
    groundtruth = [_pose(0.0, (0.0, 0.0, 0.0)), _pose(1.0, (1.0, 0.0, 0.0)), _pose(2.0, (1.0, 1.0, 0.0))]
    estimate = [_pose(0.0, (4.0, 2.0, 0.0), math.pi / 2), _pose(1.0, (4.0, 3.0, 0.0), math.pi / 2), _pose(2.0, (3.0, 3.0, 0.0), math.pi / 2)]
    result = compute_trajectory_metrics(groundtruth, estimate)
    assert result["alignment"] == "SE3"
    assert result["scale"] == 1.0
    assert result["ate_translation_rmse_m"] == pytest.approx(0.0, abs=1e-12)


def test_se3_does_not_hide_scale_error() -> None:
    groundtruth = [_pose(0.0, (0.0, 0.0, 0.0)), _pose(1.0, (1.0, 0.0, 0.0)), _pose(2.0, (2.0, 1.0, 0.0))]
    estimate = [_pose(0.0, (0.0, 0.0, 0.0)), _pose(1.0, (2.0, 0.0, 0.0)), _pose(2.0, (4.0, 2.0, 0.0))]
    result = compute_trajectory_metrics(groundtruth, estimate)
    assert result["scale"] == 1.0
    assert result["ate_translation_rmse_m"] > 0.5


def test_one_second_rpe_reports_known_translation_drift() -> None:
    groundtruth = [_pose(0.0, (0.0, 0.0, 0.0)), _pose(1.0, (1.0, 0.0, 0.0)), _pose(2.0, (2.0, 0.0, 0.0))]
    estimate = [_pose(0.0, (0.0, 0.0, 0.0)), _pose(1.0, (1.1, 0.0, 0.0)), _pose(2.0, (2.2, 0.0, 0.0))]
    result = compute_trajectory_metrics(groundtruth, estimate)
    assert result["rpe_delta_seconds"] == 1.0
    assert result["rpe_pair_count"] == 2
    assert result["rpe_translation_rmse_m"] == pytest.approx(0.1)


def test_timestamp_association_over_20ms_is_rejected() -> None:
    groundtruth = [_pose(0.0, (0.0, 0.0, 0.0)), _pose(1.0, (1.0, 0.0, 0.0)), _pose(2.0, (2.0, 0.0, 0.0))]
    estimate = [_pose(0.021, (0.0, 0.0, 0.0)), _pose(1.021, (1.0, 0.0, 0.0)), _pose(2.021, (2.0, 0.0, 0.0))]
    with pytest.raises(ValueError, match="associated poses"):
        compute_trajectory_metrics(groundtruth, estimate)


def test_published_tum_quaternion_rounding_is_normalized() -> None:
    groundtruth = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999),
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999),
        (2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9999),
    ]
    estimate = [_pose(0.0, (0.0, 0.0, 0.0)), _pose(1.0, (1.0, 0.0, 0.0)), _pose(2.0, (2.0, 0.0, 0.0))]
    result = compute_trajectory_metrics(groundtruth, estimate)
    assert result["ate_translation_rmse_m"] == pytest.approx(0.0, abs=1e-12)
