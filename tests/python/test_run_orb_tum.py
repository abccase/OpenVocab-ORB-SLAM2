from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.run_orb_tum import (
    RunCondition,
    parse_trajectory,
    run_baseline_condition,
    run_ov_condition,
)


class OracleRunnerTests(unittest.TestCase):
    def make_fixture(self, root: Path, *, exit_code: int = 0) -> tuple[RunCondition, Path, Path]:
        sequence = root / "sequence"
        sequence.mkdir()
        (sequence / "associate.txt").write_text("1.0 rgb/1.png 1.0 depth/1.png\n", encoding="utf-8")
        settings = root / "TUM1.yaml"
        settings.write_text("settings\n", encoding="utf-8")
        vocabulary = root / "ORBvoc.txt"
        vocabulary.write_text("vocabulary\n", encoding="utf-8")
        executable = root / "fake_orb.py"
        executable.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env python3
                import json, os
                from pathlib import Path
                marker = Path('launch-count.txt')
                marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else '1')
                Path('CameraTrajectory.txt').write_text('1.0 0 0 0 0 0 0 1\\n')
                Path('KeyFrameTrajectory.txt').write_text('1.0 0 0 0 0 0 0 1\\n')
                telemetry = Path(os.environ['ORB_SLAM2_FRAME_TELEMETRY'])
                telemetry.write_text(json.dumps({{'frame_index': 0, 'timestamp': 1.0,
                    'tracking_state': 2, 'pose_valid': True, 'tracking_time_seconds': 0.01}}) + '\\n')
                raise SystemExit({exit_code})
                """
            ),
            encoding="utf-8",
        )
        executable.chmod(0o755)
        condition = RunCondition("tiny", 23011, sequence, settings)
        return condition, executable, vocabulary

    def test_completed_run_records_and_validates_all_oracle_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition, executable, vocabulary = self.make_fixture(root)

            result = run_baseline_condition(
                condition,
                executable=executable,
                vocabulary=vocabulary,
                output_root=root / "runs",
                compatibility_commit="baseline-commit",
                producer_commit="producer-commit",
            )

            self.assertTrue(result.valid)
            self.assertEqual(result.frame_count, 1)
            manifest = json.loads((result.run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["state"], "COMPLETED")
            self.assertEqual(manifest["seed"], 23011)
            self.assertEqual(manifest["compatibility_commit"], "baseline-commit")
            self.assertEqual(manifest["producer_commit"], "producer-commit")
            self.assertEqual(manifest["trajectory"]["pose_count"], 1)
            self.assertEqual(len(manifest["trajectory"]["sha256"]), 64)

    def test_resume_skips_only_a_still_valid_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition, executable, vocabulary = self.make_fixture(root)
            first = run_baseline_condition(
                condition, executable=executable, vocabulary=vocabulary,
                output_root=root / "runs", compatibility_commit="baseline-commit",
                producer_commit="producer-commit",
            )
            second = run_baseline_condition(
                condition, executable=executable, vocabulary=vocabulary,
                output_root=root / "runs", compatibility_commit="baseline-commit",
                producer_commit="producer-commit",
            )

            self.assertEqual(first.run_dir, second.run_dir)
            self.assertEqual((first.run_dir / "launch-count.txt").read_text(), "1")

    def test_nonzero_process_is_preserved_as_invalid_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition, executable, vocabulary = self.make_fixture(root, exit_code=3)

            result = run_baseline_condition(
                condition, executable=executable, vocabulary=vocabulary,
                output_root=root / "runs", compatibility_commit="baseline-commit",
                producer_commit="producer-commit",
            )

            self.assertFalse(result.valid)
            manifest = json.loads((result.run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["state"], "FAILED")
            self.assertEqual(manifest["exit_code"], 3)
            self.assertIn("process exited 3", manifest["invalid_reason"])

    def test_smoke_run_records_separate_study_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition, executable, vocabulary = self.make_fixture(root)

            result = run_baseline_condition(
                condition, executable=executable, vocabulary=vocabulary,
                output_root=root / "runs", compatibility_commit="baseline-commit",
                producer_commit="producer-commit",
                study="smoke",
            )

            manifest = json.loads((result.run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["study"], "smoke")
            self.assertTrue(manifest["run_id"].startswith("smoke-"))

    def test_trajectory_parser_rejects_nonfinite_or_nonmonotonic_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.txt"
            path.write_text("2.0 0 0 0 0 0 0 1\n1.0 0 0 0 0 0 0 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_trajectory(path)
            path.write_text("1.0 nan 0 0 0 0 0 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_trajectory(path)

    def make_ov_fixture(self, root: Path) -> tuple[RunCondition, Path, Path]:
        condition, _, vocabulary = self.make_fixture(root)
        executable = root / "fake_ov.py"
        executable.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import csv, json, sys
                from pathlib import Path
                marker = Path('launch-count.txt')
                marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else '1')
                Path('invocation.json').write_text(json.dumps(sys.argv[1:]))
                mode = sys.argv[5]
                Path('CameraTrajectory.txt').write_text('1.0 0 0 0 0 0 0 1\\n')
                Path('KeyFrameTrajectory.txt').write_text('1.0 0 0 0 0 0 0 1\\n')
                header = ('frame_index,timestamp,tracking_state,pose_valid,tracking_time_seconds,'
                          'raw_keypoints,used_keypoints,removed_dynamic,retained_uncertain,'
                          'removed_uncertain,semantic_accessed,semantic_state,cache_load_seconds,'
                          'policy_seconds')
                semantic = mode == 'semantic-feedback'
                state = 'CACHE_VALID' if semantic else 'BASELINE'
                used = 90 if semantic else 100
                removed = 10 if semantic else 0
                Path('frame_telemetry.csv').write_text(
                    header + '\\n' + f'0,1.0,2,1,0.01,100,{used},{removed},0,0,{int(semantic)},{state},0.001,0.002\\n')
                Path('timings.json').write_text(json.dumps({'frame_count': 1, 'mean_tracking_seconds': 0.01}))
                Path('final_state.json').write_text(json.dumps(
                    {'state': 'COMPLETED', 'mode': mode, 'frame_count': 1}))
                """
            ),
            encoding="utf-8",
        )
        executable.chmod(0o755)
        return condition, executable, vocabulary

    def test_equivalence_runner_never_passes_or_records_semantic_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition, executable, vocabulary = self.make_ov_fixture(root)
            result = run_ov_condition(
                condition, mode="baseline", executable=executable,
                vocabulary=vocabulary, output_root=root / "runs",
                compatibility_commit="baseline-commit", producer_commit="producer-commit",
                study="equivalence",
            )
            self.assertTrue(result.valid)
            invocation = json.loads((result.run_dir / "invocation.json").read_text())
            self.assertEqual(len(invocation), 7)
            self.assertEqual(invocation[4], "baseline")
            manifest = json.loads((result.run_dir / "run_manifest.json").read_text())
            self.assertIsNone(manifest["cache_identity"])
            self.assertEqual(manifest["telemetry"]["format"], "csv")

    def test_semantic_runner_passes_independent_trusted_cache_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition, executable, vocabulary = self.make_ov_fixture(root)
            cache_root = root / "cache"
            cache_root.mkdir()
            identity = {
                "manifest_sha256": "1" * 64,
                "completion_sha256": "2" * 64,
                "index_sha256": "3" * 64,
            }
            result = run_ov_condition(
                condition, mode="semantic-feedback", executable=executable,
                vocabulary=vocabulary, output_root=root / "runs",
                compatibility_commit="baseline-commit", producer_commit="producer-commit",
                study="smoke", cache_root=cache_root, cache_identity=identity,
            )
            self.assertTrue(result.valid)
            invocation = json.loads((result.run_dir / "invocation.json").read_text())
            self.assertEqual(len(invocation), 11)
            self.assertEqual(invocation[4], "semantic-feedback")
            self.assertEqual(invocation[-3:], ["1" * 64, "2" * 64, "3" * 64])
            manifest = json.loads((result.run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["cache_identity"], identity)

    def test_ov_runner_resumes_only_hash_valid_completed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition, executable, vocabulary = self.make_ov_fixture(root)
            arguments = dict(
                mode="baseline", executable=executable, vocabulary=vocabulary,
                output_root=root / "runs", compatibility_commit="baseline-commit",
                producer_commit="producer-commit", study="equivalence",
            )
            first = run_ov_condition(condition, **arguments)
            second = run_ov_condition(condition, **arguments)
            self.assertEqual(first.run_dir, second.run_dir)
            self.assertEqual((first.run_dir / "launch-count.txt").read_text(), "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
