from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from tools.verify_baseline_equivalence import (
    associate_timestamps,
    compute_ate_rmse,
    measure_run,
    verify_equivalence,
)


class BaselineEquivalenceTests(unittest.TestCase):
    def test_run_measurement_rejects_artifact_changed_after_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            trajectory = "\n".join(
                f"{i}.0 {i} {i % 2} 0 0 0 0 1" for i in range(3)
            ) + "\n"
            trajectory_path = run / "CameraTrajectory.txt"
            trajectory_path.write_text(trajectory, encoding="utf-8")
            groundtruth = root / "groundtruth.txt"
            groundtruth.write_text(trajectory, encoding="utf-8")
            telemetry = run / "frame_telemetry.csv"
            telemetry.write_text(
                "frame_index,timestamp,tracking_state,pose_valid,tracking_time_seconds,"
                "raw_keypoints,used_keypoints,removed_dynamic,retained_uncertain,"
                "removed_uncertain,semantic_accessed,semantic_state,cache_load_seconds,"
                "policy_seconds\n" +
                "\n".join(f"{i},{i}.0,2,1,0.01,100,100,0,0,0,0,BASELINE,0,0"
                          for i in range(3)) + "\n",
                encoding="utf-8",
            )
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {
                "state": "COMPLETED", "valid": True, "mode": "baseline",
                "seed": 23011, "expected_frames": 3,
                "trajectory": {"sha256": digest(trajectory_path)},
                "telemetry": {"sha256": digest(telemetry)},
            }
            (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(measure_run(
                run, groundtruth, require_no_semantic_access=True
            )["associated_pose_count"], 3)
            trajectory_path.write_text(trajectory.replace("2.0 2", "2.0 2.001"),
                                       encoding="utf-8")
            with self.assertRaises(ValueError):
                measure_run(run, groundtruth, require_no_semantic_access=True)

    def test_timestamp_association_is_nearest_and_one_to_one(self) -> None:
        pairs = associate_timestamps([0.0, 0.01, 0.03], [0.006, 0.031], 0.02)
        self.assertEqual(pairs, [(1, 0), (2, 1)])

    def test_se3_umeyama_removes_rigid_transform_without_scale(self) -> None:
        source = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        ])
        rotation = np.array([[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]])
        target = (rotation @ source.T).T + np.array([3.0, -2.0, 0.5])
        candidate = [(float(i), *point, 0.0, 0.0, 0.0, 1.0)
                     for i, point in enumerate(source)]
        groundtruth = [(float(i), *point, 0.0, 0.0, 0.0, 1.0)
                       for i, point in enumerate(target)]
        rmse, associated = compute_ate_rmse(candidate, groundtruth, max_difference=0.02)
        self.assertEqual(associated, 4)
        self.assertLess(rmse, 1e-12)

    def test_accepts_distribution_inside_frozen_envelope(self) -> None:
        oracle = [
            {"seed": 23011 + i, "valid_pose_fraction": 0.900 + i * 0.001,
             "ate_rmse_m": 0.020 + i * 0.001}
            for i in range(5)
        ]
        candidate = [
            {"seed": 23011 + i, "valid_pose_fraction": 0.902 + i * 0.001,
             "ate_rmse_m": 0.021 + i * 0.001}
            for i in range(5)
        ]
        result = verify_equivalence(oracle, candidate)
        self.assertTrue(result["valid"])
        self.assertLessEqual(result["valid_pose_fraction_difference"], 0.005)

    def test_rejects_valid_pose_or_ate_distribution_outside_envelope(self) -> None:
        oracle = [
            {"seed": 23011 + i, "valid_pose_fraction": 0.9,
             "ate_rmse_m": 0.02}
            for i in range(5)
        ]
        bad_pose = [dict(row, valid_pose_fraction=0.894) for row in oracle]
        self.assertFalse(verify_equivalence(oracle, bad_pose)["valid"])
        bad_ate = [dict(row, ate_rmse_m=0.021) for row in oracle]
        result = verify_equivalence(oracle, bad_ate)
        self.assertFalse(result["valid"])
        self.assertEqual(result["ate_tolerance_m"], 0.0001)

    def test_requires_exactly_five_unique_paired_seeds(self) -> None:
        rows = [{"seed": 23011, "valid_pose_fraction": 0.9, "ate_rmse_m": 0.02}] * 5
        with self.assertRaises(ValueError):
            verify_equivalence(rows, rows)


if __name__ == "__main__":
    unittest.main(verbosity=2)
