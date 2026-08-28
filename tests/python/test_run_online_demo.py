from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from tools.run_online_demo import (
    preserve_executable_path,
    summarize_online_run,
    write_truncated_association,
)


HEADER = [
    "frame_index", "timestamp", "tracking_state", "pose_valid",
    "tracking_time_seconds", "raw_keypoints", "used_keypoints",
    "removed_dynamic", "retained_uncertain", "removed_uncertain",
    "semantic_accessed", "semantic_state", "cache_load_seconds",
    "policy_seconds", "pacing_lateness_seconds", "ipc_call_seconds",
    "ipc_reason", "request_attempted", "request_sent", "packet_age_ms",
    "inference_ms", "strong_track_count", "unconfirmed_track_count",
]


def row(index: int, state: str, *, attempted: int, age: float = -1.0,
        inference: float = -1.0, tracking: float = 0.02,
        ipc: float = 0.001) -> dict[str, object]:
    return {
        "frame_index": index,
        "timestamp": 1.0 + index / 30.0,
        "tracking_state": 2,
        "pose_valid": 1,
        "tracking_time_seconds": tracking,
        "raw_keypoints": 100,
        "used_keypoints": 100,
        "removed_dynamic": 0,
        "retained_uncertain": 0,
        "removed_uncertain": 0,
        "semantic_accessed": 1,
        "semantic_state": state,
        "cache_load_seconds": 0,
        "policy_seconds": 0,
        "pacing_lateness_seconds": 0,
        "ipc_call_seconds": ipc,
        "ipc_reason": "VALID_PACKET" if state == "ONLINE_VALID" else "NO_PACKET",
        "request_attempted": attempted,
        "request_sent": attempted,
        "packet_age_ms": age,
        "inference_ms": inference,
        "strong_track_count": 0,
        "unconfirmed_track_count": 0,
    }


class OnlineDemoSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write_telemetry(self, rows: list[dict[str, object]]) -> Path:
        path = self.root / "frame_telemetry.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_events(self, events: list[dict[str, object]]) -> Path:
        path = self.root / "semantic_service_events.jsonl"
        path.write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
        return path

    def write_summary(self, **updates: object) -> Path:
        value: dict[str, object] = {
            "frame_count": 4,
            "final_state": "COMPLETED",
            "request_attempts": 3,
            "requests_sent": 3,
            "unique_valid_packets": 1,
            "maximum_ipc_call_ms": 2.0,
            "wall_seconds": 2.0,
            "request_rate_cap_hz": 5.0,
            "max_mask_age_ms": 250.0,
        }
        value.update(updates)
        path = self.root / "online_summary.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_reports_measured_rates_and_separates_request_and_result_drops(self) -> None:
        telemetry = self.write_telemetry([
            row(0, "DEGRADED_TO_BASELINE", attempted=1, tracking=0.01),
            row(1, "ONLINE_VALID", attempted=1, age=100.0, inference=40.0,
                tracking=0.02),
            row(2, "ONLINE_VALID", attempted=0, age=133.0, inference=40.0,
                tracking=0.03, ipc=0.002),
            row(3, "DEGRADED_TO_BASELINE", attempted=1, tracking=0.04),
        ])
        events = self.write_events([
            {"state": "INFERENCE_COMPLETED", "frame_id": 1, "inference_ms": 40.0},
            {"state": "PUBLISHED", "frame_id": 1, "inference_ms": 40.0},
            {"state": "INFERENCE_COMPLETED", "frame_id": 2, "inference_ms": 50.0},
            {"state": "PUBLISHED", "frame_id": 2, "inference_ms": 50.0},
        ])

        report = summarize_online_run(
            telemetry, events, self.write_summary(), watchdog_ms=10.0,
            request_rate_cap_hz=5.0, max_mask_age_ms=250.0,
            require_kill_transition=True,
            peak_vram_bytes=1234,
        )

        self.assertEqual(report["state"], "VALID")
        self.assertEqual(report["request_attempt_rate_hz"], 1.5)
        self.assertEqual(report["accepted_semantic_rate_hz"], 0.5)
        self.assertAlmostEqual(report["request_drop_fraction"], 1.0 / 3.0)
        self.assertEqual(report["result_drop_fraction"], 0.0)
        self.assertEqual(report["result_unusable_fraction"], 0.5)
        self.assertEqual(report["mean_packet_age_ms"], 116.5)
        self.assertEqual(report["mean_service_inference_ms"], 45.0)
        self.assertEqual(report["mean_tracking_ms"], 25.0)
        self.assertEqual(report["peak_vram_bytes"], 1234)
        self.assertEqual(report["request_rate_cap_hz"], 5.0)
        self.assertTrue(report["kill_transition_verified"])

    def test_rejects_runner_summary_with_different_frozen_ipc_config(self) -> None:
        telemetry = self.write_telemetry([
            row(0, "ONLINE_VALID", attempted=1, age=10.0, inference=2.0),
        ])
        events = self.write_events([
            {"state": "INFERENCE_COMPLETED", "frame_id": 0, "inference_ms": 2.0},
            {"state": "PUBLISHED", "frame_id": 0, "inference_ms": 2.0},
        ])
        base = {
            "frame_count": 1, "request_attempts": 1, "requests_sent": 1,
            "unique_valid_packets": 1, "maximum_ipc_call_ms": 1.0,
            "wall_seconds": 1.0,
        }

        for changed in (
            {"request_rate_cap_hz": 4.0},
            {"max_mask_age_ms": 251.0},
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                ValueError, "frozen IPC config"
            ):
                summarize_online_run(
                    telemetry, events, self.write_summary(**base, **changed),
                    watchdog_ms=10.0, max_mask_age_ms=250.0,
                    request_rate_cap_hz=5.0, require_kill_transition=False,
                    peak_vram_bytes=None,
                )

    def test_kill_acceptance_rejects_missing_online_to_degraded_transition(self) -> None:
        telemetry = self.write_telemetry([
            row(0, "DEGRADED_TO_BASELINE", attempted=1),
            row(1, "DEGRADED_TO_BASELINE", attempted=1),
        ])
        events = self.write_events([
            {"state": "INFERENCE_COMPLETED", "frame_id": 0, "inference_ms": 2.0},
            {"state": "PUBLISHED", "frame_id": 0, "inference_ms": 2.0}
        ])

        with self.assertRaisesRegex(ValueError, "ONLINE_VALID before degradation"):
            summarize_online_run(
                telemetry, events,
                self.write_summary(frame_count=2, request_attempts=2,
                                   requests_sent=2, unique_valid_packets=1,
                                   maximum_ipc_call_ms=1.0),
                watchdog_ms=10.0, require_kill_transition=True,
                request_rate_cap_hz=5.0, max_mask_age_ms=250.0,
                peak_vram_bytes=None,
            )

    def test_watchdog_is_frozen_and_enforced(self) -> None:
        telemetry = self.write_telemetry([
            row(0, "ONLINE_VALID", attempted=1, age=10.0, inference=2.0,
                ipc=0.011),
        ])
        events = self.write_events([
            {"state": "INFERENCE_COMPLETED", "frame_id": 0, "inference_ms": 2.0},
            {"state": "PUBLISHED", "frame_id": 0, "inference_ms": 2.0},
        ])

        with self.assertRaisesRegex(ValueError, "IPC watchdog"):
            summarize_online_run(
                telemetry, events,
                self.write_summary(frame_count=1, request_attempts=1,
                                   requests_sent=1, unique_valid_packets=1,
                                   maximum_ipc_call_ms=11.0, wall_seconds=1.0),
                watchdog_ms=10.0, require_kill_transition=False,
                request_rate_cap_hz=5.0, max_mask_age_ms=250.0,
                peak_vram_bytes=None,
            )

    def test_real_backend_can_report_an_honest_degraded_only_measurement(self) -> None:
        telemetry = self.write_telemetry([
            row(0, "DEGRADED_TO_BASELINE", attempted=1),
            row(1, "DEGRADED_TO_BASELINE", attempted=0),
        ])
        events = self.write_events([
            {"state": "INFERENCE_COMPLETED", "frame_id": 0,
             "inference_ms": 1900.0},
            {"state": "PUBLISHED", "frame_id": 0, "inference_ms": 1900.0}
        ])

        report = summarize_online_run(
            telemetry, events,
            self.write_summary(
                frame_count=2, request_attempts=1, requests_sent=1,
                unique_valid_packets=0, maximum_ipc_call_ms=1.0,
            ),
            watchdog_ms=10.0, require_kill_transition=False,
            request_rate_cap_hz=5.0, max_mask_age_ms=250.0,
            allow_degraded_only=True,
            peak_vram_bytes=4_000_000_000,
        )

        self.assertEqual(report["state"], "VALID_WITH_HARDWARE_LIMITATION")
        self.assertEqual(report["accepted_semantic_rate_hz"], 0.0)
        self.assertEqual(report["result_drop_fraction"], 0.0)
        self.assertEqual(report["result_unusable_fraction"], 1.0)
        self.assertIsNone(report["mean_packet_age_ms"])
        self.assertEqual(
            report["limitation"], "NO_PACKET_MET_THE_250_MS_CAUSAL_AGE_LIMIT"
        )
        self.assertEqual(report["max_mask_age_ms"], 250.0)

    def test_truncated_association_keeps_complete_noncomment_rows(self) -> None:
        source = self.root / "associate.txt"
        source.write_text(
            "# header\n1.0 rgb/1.png 1.0 depth/1.png\n\n"
            "2.0 rgb/2.png 2.0 depth/2.png\n"
            "3.0 rgb/3.png 3.0 depth/3.png\n",
            encoding="utf-8",
        )
        output = self.root / "short.txt"

        write_truncated_association(source, output, 2)

        self.assertEqual(
            output.read_text(encoding="utf-8").splitlines(),
            ["1.0 rgb/1.png 1.0 depth/1.png", "2.0 rgb/2.png 2.0 depth/2.png"],
        )

    def test_venv_interpreter_path_is_not_dereferenced(self) -> None:
        system_python = self.root / "python-system"
        system_python.write_text("", encoding="utf-8")
        venv_python = self.root / "venv-python"
        venv_python.symlink_to(system_python)

        self.assertEqual(preserve_executable_path(venv_python), str(venv_python))


if __name__ == "__main__":
    unittest.main(verbosity=2)
