from __future__ import annotations

import hashlib
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
    validate_batch_registration,
)
from tools.run_orb_tum import RunResult
from tools.run_p05_baseline_noninferiority import (
    execute_matrix,
    expected_conditions,
    register_batch,
    validate_audit_reports,
    validate_resume,
)


ROOT = Path(__file__).resolve().parents[2]


class P05FormalRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repository = Path(self.temporary.name) / "repository"
        self.repository.mkdir()
        (self.repository / "config").mkdir()
        shutil.copyfile(
            ROOT / "config/P05_BASELINE_NONINFERIORITY_V2.json",
            self.repository / "config/P05_BASELINE_NONINFERIORITY_V2.json",
        )
        shutil.copyfile(
            ROOT / "config/EXPERIMENT_MANIFEST.yaml",
            self.repository / "config/EXPERIMENT_MANIFEST.yaml",
        )
        self.protocol_path = self.repository / "config/P05_BASELINE_NONINFERIORITY_V2.json"
        self.experiment_path = self.repository / "config/EXPERIMENT_MANIFEST.yaml"
        self.protocol = load_protocol(self.protocol_path, self.experiment_path)
        experiment = json.loads(self.experiment_path.read_text(encoding="utf-8"))

        vocabulary = self.repository / "Vocabulary/ORBvoc.txt"
        vocabulary.parent.mkdir()
        vocabulary.write_text("vocabulary\n", encoding="utf-8")
        settings_root = self.repository / "Examples/RGB-D"
        settings_root.mkdir(parents=True)
        for settings_name in ("TUM1.yaml", "TUM3.yaml"):
            (settings_root / settings_name).write_text(settings_name + "\n", encoding="utf-8")

        data_manifests = self.repository / "data/tum/manifests"
        data_manifests.mkdir(parents=True)
        for index, dataset in enumerate(experiment["datasets"]):
            sequence_root = (
                self.repository / "data/tum/raw" / dataset["archive"].removesuffix(".tgz")
            )
            sequence_root.mkdir(parents=True)
            association = sequence_root / "associate.txt"
            association.write_text("1.0 rgb/1.png 1.0 depth/1.png\n", encoding="utf-8")
            (sequence_root / "groundtruth.txt").write_text(
                "1.0 0 0 0 0 0 0 1\n", encoding="utf-8"
            )
            (data_manifests / f"{dataset['id']}.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "sequence_id": dataset["id"],
                    "association_sha256": sha256_file(association),
                    "extracted_tree_sha256": format(index + 1, "064x"),
                    "settings": dataset["settings"],
                    "validation_status": "VALID",
                }),
                encoding="utf-8",
            )

        self.candidate_executable = settings_root / "rgbd_tum_ov"
        self.candidate_executable.write_bytes(b"candidate executable\n")
        self.candidate_executable.chmod(0o755)
        self.oracle_executable = self.repository / "artifacts/oracle/source/Examples/RGB-D/rgbd_tum"
        self.oracle_executable.parent.mkdir(parents=True)
        self.oracle_executable.write_bytes(b"oracle executable\n")
        self.oracle_executable.chmod(0o755)
        self.oracle_build_manifest = self.repository / "artifacts/oracle/oracle_build_manifest.json"
        self.oracle_build_manifest.parent.mkdir(parents=True, exist_ok=True)
        self.oracle_build_manifest.write_text(
            json.dumps({
                "schema_version": 1,
                "state": "COMPLETED",
                "source_commit": ORACLE_COMMIT,
                "worktree_clean": True,
                "executable": {
                    "path": str(self.oracle_executable.resolve()),
                    "sha256": sha256_file(self.oracle_executable),
                    "size_bytes": self.oracle_executable.stat().st_size,
                },
            }),
            encoding="utf-8",
        )

        subprocess.run(["git", "init", "-q"], cwd=self.repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "p05-test@example.invalid"],
            cwd=self.repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "P05 Test"],
            cwd=self.repository,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.repository, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "fixture"],
            cwd=self.repository,
            check=True,
        )
        self.candidate_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repository, text=True
        ).strip()
        self.run_root = self.repository / "runs/p05-baseline-noninferiority-v2"
        self.registration_path = self.run_root / "batch_registration.json"
        self.registry_path = self.repository / "runs/registry.jsonl"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def register(self) -> dict[str, object]:
        return register_batch(
            self.protocol,
            self.protocol_path,
            self.repository,
            self.oracle_build_manifest,
            self.candidate_executable,
            self.registration_path,
            self.registry_path,
        )

    def test_expected_conditions_match_first_literal_and_full_matrix(self) -> None:
        conditions = expected_conditions(self.protocol)
        self.assertEqual(len(conditions), 180)
        self.assertEqual(conditions[:4], [
            {"block_id": "fr1_desk-rep-23011", "sequence_id": "fr1_desk",
             "repetition_id": 23011, "implementation": "oracle"},
            {"block_id": "fr1_desk-rep-23011", "sequence_id": "fr1_desk",
             "repetition_id": 23011, "implementation": "candidate"},
            {"block_id": "fr1_desk-rep-23012", "sequence_id": "fr1_desk",
             "repetition_id": 23012, "implementation": "oracle"},
            {"block_id": "fr1_desk-rep-23012", "sequence_id": "fr1_desk",
             "repetition_id": 23012, "implementation": "candidate"},
        ])
        self.assertEqual(
            {condition["implementation"] for condition in conditions},
            {"oracle", "candidate"},
        )

    def test_registration_binds_all_inputs_and_is_idempotent(self) -> None:
        registration = self.register()
        validate_batch_registration(
            registration, self.protocol, sha256_file(self.protocol_path)
        )
        self.assertEqual(registration["candidate"]["producer_commit"], self.candidate_commit)
        self.assertEqual(registration["matrix"]["condition_count"], 180)
        self.assertEqual(registration["matrix"]["paired_block_count"], 90)
        self.assertEqual(tuple(registration["datasets"]), SEQUENCE_IDS)
        self.assertEqual(
            registration["datasets"]["fr1_desk"]["expected_frames"], 1
        )
        self.assertEqual(self.register(), registration)
        self.assertEqual(len(self.registry_path.read_text().splitlines()), 1)

    def test_registration_rejects_dirty_tree_changed_binary_and_preexisting_attempt(self) -> None:
        tracked = self.repository / "Examples/RGB-D/TUM1.yaml"
        tracked.write_text("dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dirty"):
            self.register()
        subprocess.run(
            ["git", "restore", "Examples/RGB-D/TUM1.yaml"],
            cwd=self.repository,
            check=True,
        )
        registration = self.register()
        self.candidate_executable.write_bytes(b"changed candidate\n")
        subprocess.run(
            ["git", "add", "Examples/RGB-D/rgbd_tum_ov"],
            cwd=self.repository,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-q", "-m", "changed candidate"],
            cwd=self.repository,
            check=True,
        )
        with self.assertRaisesRegex(ValueError, "registration|executable"):
            self.register()
        self.assertEqual(
            json.loads(self.registration_path.read_text()), registration
        )

        other_root = self.repository / "runs/preexisting"
        attempt = other_root / "candidate/fr1_desk/seed-23011/attempt-001"
        attempt.mkdir(parents=True)
        other_registration = other_root / "batch_registration.json"
        with self.assertRaisesRegex(ValueError, "attempt"):
            register_batch(
                self.protocol,
                self.protocol_path,
                self.repository,
                self.oracle_build_manifest,
                self.candidate_executable,
                other_registration,
                self.registry_path,
            )

    def _attempt_manifest(
        self,
        registration: dict[str, object],
        condition: dict[str, object],
        attempt_number: int,
        *,
        producer: str | None = None,
    ) -> Path:
        implementation = str(condition["implementation"])
        producer_identity = registration[implementation]["producer_commit"]
        executable_sha = registration[implementation]["executable_sha256"]
        formal_identity = {
            "study_id": STUDY_ID,
            "block_id": condition["block_id"],
            "implementation": implementation,
            "protocol_manifest_sha256": registration["protocol_manifest_sha256"],
        }
        if implementation == "oracle":
            formal_identity["build_manifest_sha256"] = registration["oracle"][
                "build_manifest_sha256"
            ]
        else:
            formal_identity["candidate_registration_commit"] = registration[
                "candidate"
            ]["producer_commit"]
        run_dir = (
            self.run_root / implementation / str(condition["sequence_id"])
            / f"seed-{condition['repetition_id']}" / f"attempt-{attempt_number:03d}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "CameraTrajectory.txt").write_text(
            "1.0 0 0 0 0 0 0 1\n", encoding="utf-8"
        )
        (run_dir / "KeyFrameTrajectory.txt").write_text(
            "1.0 0 0 0 0 0 0 1\n", encoding="utf-8"
        )
        (run_dir / "stdout.log").write_text("", encoding="utf-8")
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        if implementation == "oracle":
            telemetry_name = "frame_telemetry.jsonl"
            (run_dir / telemetry_name).write_text(json.dumps({
                "frame_index": 0,
                "timestamp": 1.0,
                "tracking_state": 2,
                "pose_valid": True,
                "tracking_time_seconds": 0.01,
            }) + "\n", encoding="utf-8")
        else:
            telemetry_name = "frame_telemetry.csv"
            (run_dir / telemetry_name).write_text(
                "frame_index,timestamp,tracking_state,pose_valid,tracking_time_seconds,"
                "raw_keypoints,used_keypoints,removed_dynamic,retained_uncertain,"
                "removed_uncertain,semantic_accessed,semantic_state,cache_load_seconds,"
                "policy_seconds,pacing_lateness_seconds\n"
                "0,1.0,2,1,0.01,100,100,0,0,0,0,BASELINE,0,0,0\n",
                encoding="utf-8",
            )
            (run_dir / "final_state.json").write_text(json.dumps({
                "state": "COMPLETED", "mode": "baseline", "frame_count": 1,
            }), encoding="utf-8")
            (run_dir / "timings.json").write_text(json.dumps({
                "frame_count": 1,
                "mean_tracking_seconds": 0.01,
                "median_tracking_seconds": 0.01,
                "mean_pacing_lateness_seconds": 0.0,
                "max_pacing_lateness_seconds": 0.0,
                "wall_seconds": 0.01,
            }), encoding="utf-8")
        artifacts = {
            "trajectory": "CameraTrajectory.txt",
            "keyframe_trajectory": "KeyFrameTrajectory.txt",
            "telemetry": telemetry_name,
            "stdout": "stdout.log",
            "stderr": "stderr.log",
        }
        if implementation == "candidate":
            artifacts.update({
                "final_state": "final_state.json",
                "timings": "timings.json",
            })
        artifact_identities = {
            key: {"path": name, "sha256": sha256_file(run_dir / name)}
            for key, name in artifacts.items()
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "state": "COMPLETED",
                "valid": True,
                "study": STUDY_ID,
                "mode": "baseline",
                "sequence_id": condition["sequence_id"],
                "seed": condition["repetition_id"],
                "expected_frames": 1,
                "producer_commit": producer or producer_identity,
                "executable": {"sha256": executable_sha},
                "formal_identity": formal_identity,
                "cache_root": None,
                "cache_identity": None,
                **artifact_identities,
            }),
            encoding="utf-8",
        )
        return run_dir

    def test_resume_ignores_stale_producer_and_rejects_duplicate_expected_attempts(self) -> None:
        registration = self.register()
        condition = expected_conditions(self.protocol)[0]
        self._attempt_manifest(registration, condition, 1, producer="0" * 40)
        expected = self._attempt_manifest(registration, condition, 2)
        completed = validate_resume(self.run_root, registration, self.protocol)
        key = f"{condition['implementation']}:{condition['block_id']}"
        self.assertEqual(completed[key], expected)
        self._attempt_manifest(registration, condition, 3)
        with self.assertRaisesRegex(ValueError, "duplicate valid attempts"):
            validate_resume(self.run_root, registration, self.protocol)

    def test_resume_does_not_skip_artifact_corrupted_after_manifest(self) -> None:
        registration = self.register()
        condition = expected_conditions(self.protocol)[1]
        run_dir = self._attempt_manifest(registration, condition, 1)
        (run_dir / "frame_telemetry.csv").write_text("corrupt\n", encoding="utf-8")
        completed = validate_resume(self.run_root, registration, self.protocol)
        key = f"{condition['implementation']}:{condition['block_id']}"
        self.assertNotIn(key, completed)

    def test_audit_reports_require_all_six_registered_candidate_identities(self) -> None:
        registration = self.register()
        audit_root = self.run_root / "audits"
        for sequence_id in SEQUENCE_IDS:
            path = audit_root / sequence_id / "audit_report.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "schema_version": 1,
                "valid": True,
                "study_id": STUDY_ID,
                "sequence_id": sequence_id,
                "protocol_manifest_sha256": registration["protocol_manifest_sha256"],
                "candidate_producer_commit": self.candidate_commit,
                "candidate_executable_sha256": registration["candidate"]["executable_sha256"],
                "forbidden_accesses": [],
            }), encoding="utf-8")
        reports = validate_audit_reports(audit_root, registration)
        self.assertEqual(tuple(reports), SEQUENCE_IDS)
        (audit_root / SEQUENCE_IDS[-1] / "audit_report.json").unlink()
        with self.assertRaisesRegex(ValueError, "audit report"):
            validate_audit_reports(audit_root, registration)

    def test_execute_matrix_preserves_order_and_passes_no_candidate_cache(self) -> None:
        registration = self.register()
        calls: list[tuple[str, str, int, dict[str, object]]] = []

        def oracle_runner(condition, **kwargs):
            calls.append(("oracle", condition.sequence_id, condition.seed, kwargs))
            return RunResult(self.run_root / "fake-oracle", True, 1, None)

        def candidate_runner(condition, **kwargs):
            calls.append(("candidate", condition.sequence_id, condition.seed, kwargs))
            return RunResult(self.run_root / "fake-candidate", True, 1, None)

        result = execute_matrix(
            self.protocol,
            registration,
            self.run_root,
            self.registry_path,
            oracle_runner=oracle_runner,
            candidate_runner=candidate_runner,
        )
        self.assertEqual(result, {"expected": 180, "completed": 180, "valid": True})
        self.assertEqual(
            [(kind, sequence, repetition) for kind, sequence, repetition, _ in calls[:2]],
            [("oracle", "fr1_desk", 23011), ("candidate", "fr1_desk", 23011)],
        )
        candidate_kwargs = next(kwargs for kind, _, _, kwargs in calls if kind == "candidate")
        self.assertEqual(candidate_kwargs["mode"], "baseline")
        self.assertIsNone(candidate_kwargs["cache_root"])
        self.assertIsNone(candidate_kwargs["cache_identity"])
        self.assertEqual(
            candidate_kwargs["formal_identity"]["candidate_registration_commit"],
            self.candidate_commit,
        )


if __name__ == "__main__":
    unittest.main()
