from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from tools.verify_baseline_equivalence import (
    _completed_attempt,
    _validate_registered_pair,
    _write_json_atomic,
    associate_timestamps,
    compute_ate_rmse,
    measure_run,
    verify_equivalence,
)


class BaselineEquivalenceTests(unittest.TestCase):
    def test_completed_attempt_selects_only_the_expected_producer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = "a" * 40
            stale = "b" * 40
            for number, producer in ((1, stale), (2, expected)):
                attempt = root / f"attempt-{number:03d}"
                attempt.mkdir()
                (attempt / "run_manifest.json").write_text(json.dumps({
                    "state": "COMPLETED", "valid": True,
                    "producer_commit": producer,
                }), encoding="utf-8")
            self.assertEqual(
                _completed_attempt(root, expected_producer_commit=expected).name,
                "attempt-002",
            )

            duplicate = root / "attempt-003"
            duplicate.mkdir()
            (duplicate / "run_manifest.json").write_text(json.dumps({
                "state": "COMPLETED", "valid": True,
                "producer_commit": expected,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected producer"):
                _completed_attempt(root, expected_producer_commit=expected)

    def test_registered_pair_rejects_every_condition_identity_mismatch(self) -> None:
        expected_oracle_producer = "58014b7c1f2b73427b67b4e80a8cf334127f48ea"
        expected_candidate_producer = "2" * 40
        shared = {
            "compatibility_commit": "a" * 40,
            "vocabulary": {"sha256": "b" * 64},
            "settings": {"sha256": "c" * 64},
            "association_sha256": "d" * 64,
            "dataset_manifest_sha256": "e" * 64,
            "expected_frames": 3,
        }
        oracle = {**shared, "study": "oracle", "mode": "baseline",
                  "sequence_id": "tiny", "seed": 23011,
                  "producer_commit": expected_oracle_producer}
        candidate = {**shared, "study": "equivalence", "mode": "baseline",
                     "sequence_id": "tiny", "seed": 23011,
                     "producer_commit": "2" * 40, "cache_identity": None,
                     "cache_root": None, "pacing": "dataset_timestamp_paced_relative",
                     "verified_inputs": {"source_tree_sha256": "f" * 64},
                     "registration_identity": {"experiment_manifest_sha256": "9" * 64}}
        _validate_registered_pair(
            oracle, candidate, sequence_id="tiny", seed=23011,
            dataset_manifest_sha256="e" * 64, source_tree_sha256="f" * 64,
            experiment_manifest_sha256="9" * 64,
            expected_oracle_producer_commit=expected_oracle_producer,
            expected_candidate_producer_commit=expected_candidate_producer)
        mutations = [
            ("sequence", "sequence_id", "other"),
            ("seed", "seed", 23012),
            ("study", "study", "smoke"),
            ("mode", "mode", "semantic-feedback"),
            ("compatibility", "compatibility_commit", "0" * 40),
            ("vocabulary", "vocabulary", {"sha256": "0" * 64}),
            ("settings", "settings", {"sha256": "0" * 64}),
            ("association", "association_sha256", "0" * 64),
            ("dataset", "dataset_manifest_sha256", "0" * 64),
            ("cache", "cache_identity", {"manifest_sha256": "0" * 64}),
            ("producer", "producer_commit", ""),
            ("pacing", "pacing", "absolute-deadline"),
        ]
        for label, field, value in mutations:
            with self.subTest(label=label):
                broken = json.loads(json.dumps(candidate))
                broken[field] = value
                with self.assertRaises(ValueError):
                    _validate_registered_pair(
                        oracle, broken, sequence_id="tiny", seed=23011,
                        dataset_manifest_sha256="e" * 64,
                        source_tree_sha256="f" * 64,
                        experiment_manifest_sha256="9" * 64,
                        expected_oracle_producer_commit=expected_oracle_producer,
                        expected_candidate_producer_commit=expected_candidate_producer)
        for label, nested, value in (
            ("source-tree", "source_tree_sha256", "0" * 64),
        ):
            with self.subTest(label=label):
                broken = json.loads(json.dumps(candidate))
                broken["verified_inputs"][nested] = value
                with self.assertRaises(ValueError):
                    _validate_registered_pair(
                        oracle, broken, sequence_id="tiny", seed=23011,
                        dataset_manifest_sha256="e" * 64,
                        source_tree_sha256="f" * 64,
                        experiment_manifest_sha256="9" * 64,
                        expected_oracle_producer_commit=expected_oracle_producer,
                        expected_candidate_producer_commit=expected_candidate_producer)
        broken = json.loads(json.dumps(candidate))
        broken["registration_identity"]["experiment_manifest_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            _validate_registered_pair(
                oracle, broken, sequence_id="tiny", seed=23011,
                dataset_manifest_sha256="e" * 64, source_tree_sha256="f" * 64,
                experiment_manifest_sha256="9" * 64,
                expected_oracle_producer_commit=expected_oracle_producer,
                expected_candidate_producer_commit=expected_candidate_producer)

        for label, base, producer in (
            ("oracle", oracle, "3" * 40),
            ("candidate", candidate, "4" * 40),
        ):
            with self.subTest(valid_but_untrusted_producer=label):
                broken = json.loads(json.dumps(base))
                broken["producer_commit"] = producer
                checked_oracle = broken if label == "oracle" else oracle
                checked_candidate = broken if label == "candidate" else candidate
                with self.assertRaisesRegex(ValueError, f"{label} producer"):
                    _validate_registered_pair(
                        checked_oracle, checked_candidate, sequence_id="tiny", seed=23011,
                        dataset_manifest_sha256="e" * 64,
                        source_tree_sha256="f" * 64,
                        experiment_manifest_sha256="9" * 64,
                        expected_oracle_producer_commit=expected_oracle_producer,
                        expected_candidate_producer_commit=expected_candidate_producer)

    def test_atomic_report_writer_publishes_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            _write_json_atomic(path, {"tool_sha256": "a" * 64,
                                      "numpy_version": np.__version__,
                                      "parameters": {"ate_tolerance_floor_m": 1e-4}})
            self.assertEqual(json.loads(path.read_text())["numpy_version"], np.__version__)
            self.assertFalse((path.parent / f".{path.name}.partial").exists())
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
