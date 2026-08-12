from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.run_orb_tum import RunCondition, parse_trajectory, run_baseline_condition


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
            )

            self.assertTrue(result.valid)
            self.assertEqual(result.frame_count, 1)
            manifest = json.loads((result.run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["state"], "COMPLETED")
            self.assertEqual(manifest["seed"], 23011)
            self.assertEqual(manifest["compatibility_commit"], "baseline-commit")
            self.assertEqual(manifest["trajectory"]["pose_count"], 1)
            self.assertEqual(len(manifest["trajectory"]["sha256"]), 64)

    def test_resume_skips_only_a_still_valid_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            condition, executable, vocabulary = self.make_fixture(root)
            first = run_baseline_condition(
                condition, executable=executable, vocabulary=vocabulary,
                output_root=root / "runs", compatibility_commit="baseline-commit",
            )
            second = run_baseline_condition(
                condition, executable=executable, vocabulary=vocabulary,
                output_root=root / "runs", compatibility_commit="baseline-commit",
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
            )

            self.assertFalse(result.valid)
            manifest = json.loads((result.run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["state"], "FAILED")
            self.assertEqual(manifest["exit_code"], 3)
            self.assertIn("process exited 3", manifest["invalid_reason"])

    def test_trajectory_parser_rejects_nonfinite_or_nonmonotonic_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.txt"
            path.write_text("2.0 0 0 0 0 0 0 1\n1.0 0 0 0 0 0 0 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_trajectory(path)
            path.write_text("1.0 nan 0 0 0 0 0 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_trajectory(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
