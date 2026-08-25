from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from semantic_py.openvocab_slam.p05_protocol import (
    ORACLE_COMMIT,
    SEQUENCE_IDS,
    STUDY_ID,
    load_protocol,
    sha256_file,
)
from tools.run_p05_baseline_noninferiority import expected_conditions, register_batch
from tools.verify_baseline_noninferiority import (
    _candidate_pose_fraction,
    _oracle_pose_fraction,
    build_report,
    main,
)


ROOT = Path(__file__).resolve().parents[2]
TELEMETRY_HEADER = [
    "frame_index", "timestamp", "tracking_state", "pose_valid",
    "tracking_time_seconds", "raw_keypoints", "used_keypoints",
    "removed_dynamic", "retained_uncertain", "removed_uncertain",
    "semantic_accessed", "semantic_state", "cache_load_seconds",
    "policy_seconds", "pacing_lateness_seconds",
]


class P05VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repository = Path(self.temporary.name) / "repository"
        self.repository.mkdir()
        (self.repository / "config").mkdir()
        shutil.copyfile(
            ROOT / "config/P05_BASELINE_NONINFERIORITY_V3.json",
            self.repository / "config/P05_BASELINE_NONINFERIORITY_V3.json",
        )
        shutil.copyfile(
            ROOT / "config/EXPERIMENT_MANIFEST.yaml",
            self.repository / "config/EXPERIMENT_MANIFEST.yaml",
        )
        self.protocol_path = self.repository / "config/P05_BASELINE_NONINFERIORITY_V3.json"
        self.experiment_path = self.repository / "config/EXPERIMENT_MANIFEST.yaml"
        self.protocol = load_protocol(self.protocol_path, self.experiment_path)
        experiment = json.loads(self.experiment_path.read_text(encoding="utf-8"))

        vocabulary = self.repository / "Vocabulary/ORBvoc.txt"
        vocabulary.parent.mkdir()
        vocabulary.write_text("vocabulary\n", encoding="utf-8")
        settings_root = self.repository / "Examples/RGB-D"
        settings_root.mkdir(parents=True)
        for name in ("TUM1.yaml", "TUM3.yaml"):
            (settings_root / name).write_text(name + "\n", encoding="utf-8")
        manifests_root = self.repository / "data/tum/manifests"
        manifests_root.mkdir(parents=True)
        for index, dataset in enumerate(experiment["datasets"]):
            sequence_root = (
                self.repository / "data/tum/raw"
                / dataset["archive"].removesuffix(".tgz")
            )
            sequence_root.mkdir(parents=True)
            association = sequence_root / "associate.txt"
            association.write_text(
                "".join(
                    f"{value}.0 rgb/{value}.png {value}.0 depth/{value}.png\n"
                    for value in range(1, 11)
                ),
                encoding="utf-8",
            )
            (sequence_root / "groundtruth.txt").write_text(
                self.trajectory_text(1.0), encoding="utf-8"
            )
            (manifests_root / f"{dataset['id']}.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "sequence_id": dataset["id"],
                    "association_sha256": sha256_file(association),
                    "extracted_tree_sha256": format(index + 1, "064x"),
                    "settings": dataset["settings"],
                    "validation_status": "VALID",
                }), encoding="utf-8",
            )

        self.candidate_executable = settings_root / "rgbd_tum_ov"
        self.candidate_executable.write_bytes(b"candidate executable\n")
        self.candidate_executable.chmod(0o755)
        self.oracle_executable = (
            self.repository / "artifacts/oracle/source/Examples/RGB-D/rgbd_tum"
        )
        self.oracle_executable.parent.mkdir(parents=True)
        self.oracle_executable.write_bytes(b"oracle executable\n")
        self.oracle_executable.chmod(0o755)
        self.oracle_build_manifest = (
            self.repository / "artifacts/oracle/oracle_build_manifest.json"
        )
        self.oracle_build_manifest.parent.mkdir(parents=True, exist_ok=True)
        self.oracle_build_manifest.write_text(json.dumps({
            "schema_version": 1,
            "state": "COMPLETED",
            "source_commit": ORACLE_COMMIT,
            "worktree_clean": True,
            "executable": {
                "path": str(self.oracle_executable.resolve()),
                "sha256": sha256_file(self.oracle_executable),
                "size_bytes": self.oracle_executable.stat().st_size,
            },
        }), encoding="utf-8")

        subprocess.run(["git", "init", "-q"], cwd=self.repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "verifier@example.invalid"],
            cwd=self.repository, check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Verifier Test"],
            cwd=self.repository, check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.repository, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "fixture"],
            cwd=self.repository, check=True,
        )
        self.run_root = self.repository / "runs/p05-baseline-noninferiority-v3"
        self.registration_path = self.run_root / "batch_registration.json"
        self.registry_path = self.repository / "runs/registry.jsonl"
        self.registration = register_batch(
            self.protocol, self.protocol_path, self.repository,
            self.oracle_build_manifest, self.candidate_executable,
            self.registration_path, self.registry_path,
        )
        self.oracle_root = self.run_root / "oracle"
        self.candidate_root = self.run_root / "candidate"
        self.audit_root = self.run_root / "audits"
        self.output_path = self.repository / "reports/p05-v2.json"
        self.paths = {
            "protocol_path": self.protocol_path,
            "experiment_path": self.experiment_path,
            "registration_path": self.registration_path,
            "oracle_root": self.oracle_root,
            "candidate_root": self.candidate_root,
            "audit_root": self.audit_root,
            "data_root": self.repository / "data/tum/raw",
            "data_manifest_root": self.repository / "data/tum/manifests",
            "repository_root": self.repository,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def trajectory_text(scale: float) -> str:
        points = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        return "".join(
            f"{index}.0 {scale * x:.12f} {scale * y:.12f} 0 0 0 0 1\n"
            for index, (x, y) in enumerate(points, 1)
        )

    @staticmethod
    def artifact(path: Path) -> dict[str, object]:
        return {
            "path": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    def formal_identity(self, condition: dict[str, object]) -> dict[str, object]:
        implementation = str(condition["implementation"])
        value = {
            "study_id": STUDY_ID,
            "block_id": condition["block_id"],
            "implementation": implementation,
            "protocol_manifest_sha256": self.registration["protocol_manifest_sha256"],
        }
        if implementation == "oracle":
            value["build_manifest_sha256"] = self.registration["oracle"][
                "build_manifest_sha256"
            ]
        else:
            value["candidate_registration_commit"] = self.registration["candidate"][
                "producer_commit"
            ]
        return value

    def make_attempt(
        self,
        condition: dict[str, object],
        *,
        pose_delta: float,
        ate_ratio: float,
        attempt_number: int = 1,
    ) -> Path:
        implementation = str(condition["implementation"])
        root = self.oracle_root if implementation == "oracle" else self.candidate_root
        run_dir = (
            root / str(condition["sequence_id"])
            / f"seed-{condition['repetition_id']}" / f"attempt-{attempt_number:03d}"
        )
        run_dir.mkdir(parents=True)
        scale = 1.1 if implementation == "oracle" else 1.0 + 0.1 * ate_ratio
        (run_dir / "CameraTrajectory.txt").write_text(
            self.trajectory_text(scale), encoding="utf-8"
        )
        (run_dir / "KeyFrameTrajectory.txt").write_text(
            self.trajectory_text(scale), encoding="utf-8"
        )
        (run_dir / "stdout.log").write_text("", encoding="utf-8")
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        artifacts = {
            "trajectory": self.artifact(run_dir / "CameraTrajectory.txt"),
            "keyframe_trajectory": self.artifact(run_dir / "KeyFrameTrajectory.txt"),
            "stdout": self.artifact(run_dir / "stdout.log"),
            "stderr": self.artifact(run_dir / "stderr.log"),
        }
        valid_count = 8 if implementation == "oracle" else round((0.8 + pose_delta) * 10)
        if implementation == "oracle":
            telemetry = run_dir / "frame_telemetry.jsonl"
            telemetry.write_text("".join(
                json.dumps({
                    "frame_index": index,
                    "timestamp": float(index + 1),
                    "tracking_state": 2,
                    "pose_valid": index < valid_count,
                    "tracking_time_seconds": 0.01,
                }) + "\n" for index in range(10)
            ), encoding="utf-8")
        else:
            telemetry = run_dir / "frame_telemetry.csv"
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=TELEMETRY_HEADER, lineterminator="\n")
            writer.writeheader()
            for index in range(10):
                writer.writerow({
                    "frame_index": index,
                    "timestamp": f"{index + 1}.0",
                    "tracking_state": 2,
                    "pose_valid": int(index < valid_count),
                    "tracking_time_seconds": 0.01,
                    "raw_keypoints": 100,
                    "used_keypoints": 100,
                    "removed_dynamic": 0,
                    "retained_uncertain": 0,
                    "removed_uncertain": 0,
                    "semantic_accessed": 0,
                    "semantic_state": "BASELINE",
                    "cache_load_seconds": 0,
                    "policy_seconds": 0,
                    "pacing_lateness_seconds": 0,
                })
            telemetry.write_text(buffer.getvalue(), encoding="utf-8")
            final_state = run_dir / "final_state.json"
            final_state.write_text(json.dumps({
                "state": "COMPLETED", "mode": "baseline", "frame_count": 10,
            }), encoding="utf-8")
            timings = run_dir / "timings.json"
            timings.write_text(json.dumps({
                "frame_count": 10,
                "mean_tracking_seconds": 0.01,
                "median_tracking_seconds": 0.01,
                "mean_pacing_lateness_seconds": 0.0,
                "max_pacing_lateness_seconds": 0.0,
                "wall_seconds": 0.1,
            }), encoding="utf-8")
            artifacts["final_state"] = self.artifact(final_state)
            artifacts["timings"] = self.artifact(timings)
        artifacts["telemetry"] = self.artifact(telemetry)

        identity = self.registration[implementation]
        dataset = self.registration["datasets"][str(condition["sequence_id"])]
        manifest = {
            "schema_version": 1 if implementation == "oracle" else 2,
            "state": "COMPLETED",
            "valid": True,
            "study": STUDY_ID,
            "mode": "baseline",
            "sequence_id": condition["sequence_id"],
            "seed": condition["repetition_id"],
            "expected_frames": 10,
            "producer_commit": identity["producer_commit"],
            "compatibility_commit": self.registration["compatibility_commit"],
            "executable": {
                "path": identity["executable"],
                "sha256": identity["executable_sha256"],
            },
            "vocabulary": self.registration["vocabulary"],
            "settings": {
                "path": dataset["settings"],
                "sha256": dataset["settings_sha256"],
            },
            "association_sha256": dataset["association_sha256"],
            "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
            "formal_identity": self.formal_identity(condition),
            "cache_root": None,
            "cache_identity": None,
            **artifacts,
        }
        if implementation == "candidate":
            manifest["pacing"] = "dataset_timestamp_paced_relative"
            manifest["verified_inputs"] = {
                "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
                "source_tree_sha256": dataset["source_tree_sha256"],
            }
            manifest["registration_identity"] = {
                "formal_identity": self.formal_identity(condition),
                "experiment_manifest_sha256": self.registration[
                    "experiment_manifest"
                ]["sha256"],
                "source_tree_sha256": dataset["source_tree_sha256"],
                "cache_root": None,
                "cache_identity": None,
                "mode": "baseline",
                "study": STUDY_ID,
            }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return run_dir

    def make_audits(self) -> None:
        for sequence_id in SEQUENCE_IDS:
            root = self.audit_root / sequence_id
            root.mkdir(parents=True)
            trace = root / "trace.123"
            trace.write_text(
                'openat(AT_FDCWD, "/usr/lib/libc.so.6", O_RDONLY) = 3\n',
                encoding="utf-8",
            )
            (root / "audit_report.json").write_text(json.dumps({
                "schema_version": 1,
                "valid": True,
                "study_id": STUDY_ID,
                "sequence_id": sequence_id,
                "protocol_manifest_sha256": self.registration[
                    "protocol_manifest_sha256"
                ],
                "candidate_producer_commit": self.registration["candidate"][
                    "producer_commit"
                ],
                "candidate_executable_sha256": self.registration["candidate"][
                    "executable_sha256"
                ],
                "cwd": str(root.resolve()),
                "forbidden_roots": [
                    str((self.repository / "cache/semantic").resolve()),
                    str((self.repository / "cache/dynamic").resolve()),
                ],
                "forbidden_files": sorted([
                    str((self.repository / "config/PROMPTS.yaml").resolve()),
                    str((self.repository / "config/SEMANTIC_MODELS.json").resolve()),
                    str((self.repository / "config/DYNAMIC_CACHE_IDENTITY.json").resolve()),
                ]),
                "forbidden_accesses": [],
                "trace_files": [{
                    "path": str(trace.resolve()),
                    "sha256": sha256_file(trace),
                    "size_bytes": trace.stat().st_size,
                }],
            }), encoding="utf-8")

    def make_complete_study(self, *, pose_delta: float = 0.0, ate_ratio: float = 1.0) -> None:
        self.make_audits()
        for condition in expected_conditions(self.protocol):
            self.make_attempt(
                condition, pose_delta=pose_delta, ate_ratio=ate_ratio
            )

    def candidate_attempt(self, sequence_id: str = "fr1_desk", repetition: int = 23011) -> Path:
        return self.candidate_root / sequence_id / f"seed-{repetition}" / "attempt-001"

    def cli_arguments(self) -> list[str]:
        return [
            "--protocol", str(self.protocol_path),
            "--experiment-manifest", str(self.experiment_path),
            "--registration", str(self.registration_path),
            "--oracle-root", str(self.oracle_root),
            "--candidate-root", str(self.candidate_root),
            "--audit-root", str(self.audit_root),
            "--data-root", str(self.repository / "data/tum/raw"),
            "--data-manifests", str(self.repository / "data/tum/manifests"),
            "--repository", str(self.repository),
            "--output", str(self.output_path),
        ]

    def test_oracle_timestamp_accepts_frozen_legacy_rounding_bound(self) -> None:
        telemetry = self.repository / "oracle_rounding.jsonl"
        telemetry.write_text(json.dumps({
            "frame_index": 0,
            "timestamp": 1.000005,
            "tracking_state": 2,
            "pose_valid": True,
            "tracking_time_seconds": 0.01,
        }) + "\n", encoding="utf-8")
        self.assertEqual(
            _oracle_pose_fraction(telemetry, [1.0000101], 1e-5),
            1.0,
        )

    def test_oracle_timestamp_rejects_value_beyond_frozen_bound(self) -> None:
        telemetry = self.repository / "oracle_out_of_bound.jsonl"
        telemetry.write_text(json.dumps({
            "frame_index": 0,
            "timestamp": 1.000011,
            "tracking_state": 2,
            "pose_valid": True,
            "tracking_time_seconds": 0.01,
        }) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "timestamp differs"):
            _oracle_pose_fraction(telemetry, [1.0], 1e-5)

    def test_candidate_timestamp_remains_exact(self) -> None:
        telemetry = self.repository / "candidate_inexact.csv"
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer, fieldnames=TELEMETRY_HEADER, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow({
            "frame_index": "0",
            "timestamp": "1.0000001",
            "tracking_state": "2",
            "pose_valid": "1",
            "tracking_time_seconds": "0.01",
            "raw_keypoints": "100",
            "used_keypoints": "100",
            "removed_dynamic": "0",
            "retained_uncertain": "0",
            "removed_uncertain": "0",
            "semantic_accessed": "0",
            "semantic_state": "BASELINE",
            "cache_load_seconds": "0.0",
            "policy_seconds": "0.0",
            "pacing_lateness_seconds": "0.0",
        })
        telemetry.write_text(buffer.getvalue(), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "timestamp differs"):
            _candidate_pose_fraction(telemetry, [1.0])

    def test_complete_synthetic_study_passes_and_records_tool_identity(self) -> None:
        self.make_complete_study()
        report = build_report(**self.paths)
        self.assertTrue(report["valid"])
        self.assertEqual(report["deterministic_gates"]["expected_run_count"], 180)
        self.assertEqual(report["statistics"]["resamples"], 100000)
        self.assertEqual(len(report["sequences"]), 6)
        self.assertRegex(report["verifier_sha256"], r"^[0-9a-f]{64}$")

    def test_candidate_semantic_access_blocks_statistics(self) -> None:
        self.make_complete_study()
        attempt = self.candidate_attempt()
        telemetry = attempt / "frame_telemetry.csv"
        rows = list(csv.DictReader(telemetry.read_text().splitlines()))
        rows[0]["semantic_accessed"] = "1"
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=TELEMETRY_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        telemetry.write_text(buffer.getvalue(), encoding="utf-8")
        manifest_path = attempt / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["telemetry"]["sha256"] = sha256_file(telemetry)
        manifest["telemetry"]["size_bytes"] = telemetry.stat().st_size
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "baseline accessed semantic state"):
            build_report(**self.paths)

    def test_missing_duplicate_and_corrupt_attempts_fail_closed(self) -> None:
        self.make_complete_study()
        missing = self.oracle_root / "fr1_room/seed-23012/attempt-001"
        shutil.rmtree(missing)
        with self.assertRaisesRegex(ValueError, "exactly one valid attempt"):
            build_report(**self.paths)
        condition = next(
            value for value in expected_conditions(self.protocol)
            if value["implementation"] == "oracle"
            and value["sequence_id"] == "fr1_room"
            and value["repetition_id"] == 23012
        )
        self.make_attempt(condition, pose_delta=0.0, ate_ratio=1.0)
        source = self.candidate_attempt("fr1_desk", 23012)
        shutil.copytree(source, source.parent / "attempt-002")
        with self.assertRaisesRegex(ValueError, "exactly one valid attempt"):
            build_report(**self.paths)
        shutil.rmtree(source.parent / "attempt-002")
        telemetry = self.candidate_attempt() / "frame_telemetry.csv"
        telemetry.write_text(telemetry.read_text() + "corrupt\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            build_report(**self.paths)

    def test_failed_access_audit_blocks_run_discovery(self) -> None:
        self.make_complete_study()
        path = self.audit_root / "fr1_desk/audit_report.json"
        report = json.loads(path.read_text())
        report["valid"] = False
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "access audit"):
            build_report(**self.paths)

    def test_verifier_reparses_trace_instead_of_trusting_valid_report(self) -> None:
        self.make_complete_study()
        report_path = self.audit_root / "fr1_desk/audit_report.json"
        report = json.loads(report_path.read_text())
        trace = Path(report["trace_files"][0]["path"])
        forbidden = self.repository / "cache/semantic/v1/index.jsonl"
        trace.write_text(
            f'openat(AT_FDCWD, "{forbidden}", O_RDONLY) = 3\n',
            encoding="utf-8",
        )
        report["trace_files"][0]["sha256"] = sha256_file(trace)
        report["trace_files"][0]["size_bytes"] = trace.stat().st_size
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "forbidden file access"):
            build_report(**self.paths)

    def test_noninferiority_failure_produces_complete_failed_report(self) -> None:
        self.make_complete_study(pose_delta=-0.2, ate_ratio=1.5)
        report = build_report(**self.paths)
        self.assertFalse(report["valid"])
        self.assertFalse(report["sequences"]["fr1_desk"]["pose_pass"])
        self.assertFalse(report["sequences"]["fr1_desk"]["ate_pass"])

    def test_cli_does_not_publish_partial_report_on_hard_gate_failure(self) -> None:
        self.make_complete_study()
        shutil.rmtree(self.oracle_root / "fr1_room/seed-23012/attempt-001")
        self.assertEqual(main(self.cli_arguments()), 1)
        self.assertFalse(self.output_path.exists())
        self.assertFalse(
            self.output_path.with_name(f".{self.output_path.name}.partial").exists()
        )


if __name__ == "__main__":
    unittest.main()
