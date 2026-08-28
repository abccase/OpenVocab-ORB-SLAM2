#!/usr/bin/env python3
"""Run, validate, and register the P06 online semantic-feedback demo."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TELEMETRY_FIELDS = [
    "frame_index", "timestamp", "tracking_state", "pose_valid",
    "tracking_time_seconds", "raw_keypoints", "used_keypoints",
    "removed_dynamic", "retained_uncertain", "removed_uncertain",
    "semantic_accessed", "semantic_state", "cache_load_seconds",
    "policy_seconds", "pacing_lateness_seconds", "ipc_call_seconds",
    "ipc_reason", "request_attempted", "request_sent", "packet_age_ms",
    "inference_ms", "strong_track_count", "unconfirmed_track_count",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserve_executable_path(path: Path) -> str:
    """Make an executable path absolute without dereferencing a venv symlink."""
    return str(path.expanduser().absolute())


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.parent / f".{path.name}.partial"
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: {path}")
    return value


def _normalize_prompt(raw: str) -> str:
    terms: list[str] = []
    for part in raw.replace("\n", " ").split("."):
        term = " ".join(part.strip().lower().split())
        if term and term not in terms:
            terms.append(term)
    if not terms:
        raise ValueError("formal prompt has no terms")
    return " . ".join(terms) + " ."


def write_truncated_association(source: Path, output: Path, frame_count: int) -> None:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    rows = [
        line.strip() for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows) < frame_count:
        raise ValueError("association has fewer rows than requested")
    for line in rows[:frame_count]:
        if len(line.split()) != 4:
            raise ValueError("association row must have four fields")
    output.write_text("\n".join(rows[:frame_count]) + "\n", encoding="utf-8")


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.is_file():
        return events
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid service event at line {line_number}") from error
        if not isinstance(event, dict) or not isinstance(event.get("state"), str):
            raise ValueError(f"invalid service event at line {line_number}")
        events.append(event)
    return events


def _finite_float(value: object, label: str, *, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {label}") from error
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"invalid {label}")
    return parsed


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def summarize_online_run(
    telemetry_path: Path,
    events_path: Path,
    summary_path: Path,
    *,
    watchdog_ms: float,
    request_rate_cap_hz: float,
    max_mask_age_ms: float,
    require_kill_transition: bool,
    peak_vram_bytes: int | None,
    allow_degraded_only: bool = False,
) -> dict[str, object]:
    """Validate outputs and return measured P06 online metrics."""
    summary = _read_json_object(summary_path, "online summary")
    if summary.get("final_state") != "COMPLETED":
        raise ValueError("online runner did not complete")
    wall_seconds = _finite_float(summary.get("wall_seconds"), "wall time", minimum=0.0)
    if wall_seconds <= 0.0:
        raise ValueError("wall time must be positive")
    watchdog_ms = _finite_float(watchdog_ms, "watchdog", minimum=0.0)
    if watchdog_ms <= 0.0:
        raise ValueError("watchdog must be positive")
    request_rate_cap_hz = _finite_float(
        request_rate_cap_hz, "request rate cap", minimum=0.0
    )
    if request_rate_cap_hz <= 0.0 or request_rate_cap_hz > 5.0:
        raise ValueError("request rate cap must be in (0, 5] Hz")
    max_mask_age_ms = _finite_float(
        max_mask_age_ms, "maximum mask age", minimum=0.0
    )
    if max_mask_age_ms <= 0.0:
        raise ValueError("maximum mask age must be positive")
    declared_rate_cap = _finite_float(
        summary.get("request_rate_cap_hz"), "summary request rate cap", minimum=0.0
    )
    declared_max_age = _finite_float(
        summary.get("max_mask_age_ms"), "summary maximum mask age", minimum=0.0
    )
    if (not math.isclose(declared_rate_cap, request_rate_cap_hz, rel_tol=1e-12) or
            not math.isclose(declared_max_age, max_mask_age_ms, rel_tol=1e-12)):
        raise ValueError("runner summary does not match frozen IPC config")

    with telemetry_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != TELEMETRY_FIELDS:
            raise ValueError("online telemetry schema mismatch")
        rows = list(reader)
    expected_frames = int(summary.get("frame_count", -1))
    if expected_frames <= 0 or len(rows) != expected_frames:
        raise ValueError("online telemetry frame count mismatch")

    states: list[str] = []
    tracking_ms: list[float] = []
    ipc_ms: list[float] = []
    packet_ages: list[float] = []
    attempted = 0
    sent = 0
    for index, row in enumerate(rows):
        if int(row["frame_index"]) != index:
            raise ValueError("online telemetry frame indices are not contiguous")
        state = row["semantic_state"]
        if state not in {"ONLINE_VALID", "DEGRADED_TO_BASELINE"}:
            raise ValueError("online telemetry has an invalid semantic state")
        if row["semantic_accessed"] != "1":
            raise ValueError("online frame did not access semantic provider")
        states.append(state)
        tracking_ms.append(
            1000.0 * _finite_float(row["tracking_time_seconds"], "tracking time", minimum=0.0)
        )
        ipc_ms.append(
            1000.0 * _finite_float(row["ipc_call_seconds"], "IPC time", minimum=0.0)
        )
        if row["request_attempted"] not in {"0", "1"} or row["request_sent"] not in {"0", "1"}:
            raise ValueError("invalid request flags")
        attempted += int(row["request_attempted"])
        sent += int(row["request_sent"])
        if state == "ONLINE_VALID":
            age = _finite_float(row["packet_age_ms"], "packet age", minimum=0.0)
            if age > max_mask_age_ms:
                raise ValueError(
                    f"ONLINE_VALID packet exceeded {max_mask_age_ms:g} ms"
                )
            packet_ages.append(age)

    summary_attempts = int(summary.get("request_attempts", -1))
    summary_sent = int(summary.get("requests_sent", -1))
    unique_packets = int(summary.get("unique_valid_packets", -1))
    if (summary_attempts != attempted or summary_sent != sent or
            unique_packets < 0 or sent > attempted or
            (unique_packets == 0 and not allow_degraded_only)):
        raise ValueError("online request counters are inconsistent")
    maximum_ipc_ms = max(ipc_ms)
    declared_maximum = _finite_float(
        summary.get("maximum_ipc_call_ms"), "maximum IPC time", minimum=0.0
    )
    if abs(maximum_ipc_ms - declared_maximum) > max(0.05, declared_maximum * 0.05):
        raise ValueError("online summary IPC maximum does not match telemetry")
    if maximum_ipc_ms > watchdog_ms:
        raise ValueError(
            f"IPC watchdog exceeded: {maximum_ipc_ms:.6f} ms > {watchdog_ms:.6f} ms"
        )

    events = _load_events(events_path)
    completed = [event for event in events if event["state"] == "INFERENCE_COMPLETED"]
    inference_failed = [event for event in events if event["state"] == "INFERENCE_FAILED"]
    published = [event for event in events if event["state"] == "PUBLISHED"]
    dropped = [event for event in events if event["state"] == "RESULT_DROPPED"]
    if not published:
        raise ValueError("semantic service published no packets")
    consumed = len(completed) + len(inference_failed)
    send_outcomes = len(published) + len(dropped)
    if (unique_packets > len(published) or consumed > attempted or
            send_outcomes > len(completed)):
        raise ValueError("service/runner packet counters are inconsistent")
    inference_ms = [
        _finite_float(event.get("inference_ms"), "service inference time", minimum=0.0)
        for event in completed
    ]

    online_indices = [index for index, state in enumerate(states) if state == "ONLINE_VALID"]
    transition_verified = bool(
        online_indices and any(
            state == "DEGRADED_TO_BASELINE"
            for state in states[online_indices[-1] + 1:]
        )
    )
    if require_kill_transition and not transition_verified:
        raise ValueError("kill test requires ONLINE_VALID before degradation")

    degraded_frames = states.count("DEGRADED_TO_BASELINE")
    degraded_only = unique_packets == 0
    report: dict[str, object] = {
        "schema_version": 1,
        "state": "VALID_WITH_HARDWARE_LIMITATION" if degraded_only else "VALID",
        "frame_count": len(rows),
        "wall_seconds": wall_seconds,
        "request_attempts": attempted,
        "requests_sent": sent,
        "service_consumed_requests": consumed,
        "service_completed_inferences": len(completed),
        "service_published_packets": len(published),
        "service_dropped_results": len(dropped),
        "unique_valid_packets": unique_packets,
        "request_attempt_rate_hz": attempted / wall_seconds,
        "accepted_semantic_rate_hz": unique_packets / wall_seconds,
        "request_drop_fraction": max(0.0, 1.0 - consumed / attempted),
        "result_drop_fraction": len(dropped) / send_outcomes if send_outcomes else 0.0,
        "result_unusable_fraction": max(0.0, 1.0 - unique_packets / len(published)),
        "degraded_frame_fraction": degraded_frames / len(rows),
        "mean_packet_age_ms": statistics.fmean(packet_ages) if packet_ages else None,
        "p95_packet_age_ms": _percentile(packet_ages, 0.95) if packet_ages else None,
        "mean_service_inference_ms": statistics.fmean(inference_ms),
        "p95_service_inference_ms": _percentile(inference_ms, 0.95),
        "mean_tracking_ms": statistics.fmean(tracking_ms),
        "p95_tracking_ms": _percentile(tracking_ms, 0.95),
        "maximum_ipc_call_ms": maximum_ipc_ms,
        "watchdog_threshold_ms": watchdog_ms,
        "request_rate_cap_hz": request_rate_cap_hz,
        "max_mask_age_ms": max_mask_age_ms,
        "kill_transition_verified": transition_verified if require_kill_transition else None,
        "peak_vram_bytes": peak_vram_bytes,
        "limitation": (
            f"NO_PACKET_MET_THE_{max_mask_age_ms:g}_MS_CAUSAL_AGE_LIMIT"
            if degraded_only else None
        ),
    }
    return report


def _next_attempt(root: Path) -> tuple[Path, int]:
    existing = sorted(root.glob("attempt-*")) if root.is_dir() else []
    number = len(existing) + 1
    return root / f"attempt-{number:03d}", number


def _event_count(path: Path, state: str) -> int:
    try:
        return sum(event.get("state") == state for event in _load_events(path))
    except (OSError, ValueError):
        return 0


def _sample_vram_bytes(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi", "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False, capture_output=True, text=True, timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    values: list[int] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                values.append(int(fields[1]) * 1024 * 1024)
            except ValueError:
                pass
    return max(values) if values else None


def _stop_child(process: subprocess.Popen[bytes], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_seconds)


def run_online_demo(args: argparse.Namespace) -> tuple[Path, bool, str | None]:
    config = _read_json_object(args.config.resolve(), "P06 config")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported P06 config")
    if args.sequence != config.get("sequence_id"):
        raise ValueError("P06 config sequence mismatch")
    if args.prompt_set != config.get("prompt_set_id"):
        raise ValueError("P06 config prompt-set mismatch")
    watchdog_ms = _finite_float(config.get("ipc_watchdog_ms"), "IPC watchdog", minimum=0.0)
    request_rate_cap_hz = _finite_float(
        config.get("request_rate_cap_hz"), "request rate cap", minimum=0.0
    )
    max_mask_age_ms = _finite_float(
        config.get("max_mask_age_ms"), "maximum mask age", minimum=0.0
    )
    if request_rate_cap_hz <= 0.0 or request_rate_cap_hz > 5.0:
        raise ValueError("request rate cap must be in (0, 5] Hz")
    if max_mask_age_ms <= 0.0:
        raise ValueError("maximum mask age must be positive")

    sequence_root = args.sequence_root.resolve()
    association = sequence_root / "associate.txt"
    sequence_manifest_path = (
        PROJECT_ROOT / "data/tum/manifests" / f"{args.sequence}.json"
    )
    sequence_manifest = _read_json_object(sequence_manifest_path, "dataset manifest")
    if (sequence_manifest.get("sequence_id") != args.sequence or
            sequence_manifest.get("validation_status") != "VALID" or
            sequence_manifest.get("association_sha256") != _sha256_file(association)):
        raise ValueError("dataset identity mismatch")

    prompts = _read_json_object(PROJECT_ROOT / "config/PROMPTS.yaml", "prompt config")
    if prompts.get("prompt_set_id") != args.prompt_set:
        raise ValueError("prompt identity mismatch")
    prompt = _normalize_prompt(str(prompts.get("frozen_formal_prompt", "")))
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    model_manifest_path = PROJECT_ROOT / "config/SEMANTIC_MODELS.json"
    model_manifest_sha256 = _sha256_file(model_manifest_path)

    kind = "kill-test" if args.kill_after_packets else "demo"
    condition_root = args.output_root.resolve() / args.sequence / f"seed-{args.seed}" / kind
    run_dir, attempt = _next_attempt(condition_root)
    run_dir.mkdir(parents=True, exist_ok=False)
    selected_association = association
    if args.max_frames is not None:
        selected_association = run_dir / "associate-short.txt"
        write_truncated_association(association, selected_association, args.max_frames)
    expected_frames = sum(
        bool(line.strip() and not line.lstrip().startswith("#"))
        for line in selected_association.read_text(encoding="utf-8").splitlines()
    )

    run_id = (
        f"p06-online-{kind}-{args.sequence}-seed-{args.seed}-attempt-{attempt:03d}"
    )
    event_log = run_dir / "semantic_service_events.jsonl"
    runner_command = [
        str(args.executable.resolve()), str(args.vocabulary.resolve()),
        str(args.settings.resolve()), str(sequence_root), str(selected_association.resolve()),
        args.sequence, str(args.seed), run_id, prompt_sha256,
        model_manifest_sha256, args.request_endpoint, args.result_endpoint,
        format(request_rate_cap_hz, ".17g"), format(max_mask_age_ms, ".17g"),
    ]
    service_command = [
        preserve_executable_path(args.service_python), str(args.service_script.resolve()),
        "--run-id", run_id, "--prompt-set", args.prompt_set,
        "--request-endpoint", args.request_endpoint,
        "--result-endpoint", args.result_endpoint,
        "--device", args.device, "--backend", args.service_backend,
        "--event-log", str(event_log.resolve()),
    ]
    base: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "study": "p06-online-demo-v1",
        "mode": "online-semantic-feedback",
        "kind": kind,
        "sequence_id": args.sequence,
        "seed": args.seed,
        "state": "REGISTERED",
        "valid": False,
        "start_time_utc": _utc_now(),
        "expected_frames": expected_frames,
        "prompt_set_id": args.prompt_set,
        "prompt_sha256": prompt_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "dataset_manifest_sha256": _sha256_file(sequence_manifest_path),
        "association_sha256": _sha256_file(selected_association),
        "ipc_watchdog_ms": watchdog_ms,
        "request_rate_cap_hz": request_rate_cap_hz,
        "max_mask_age_ms": max_mask_age_ms,
        "kill_after_packets": args.kill_after_packets,
        "service_backend": args.service_backend,
        "runner_command": runner_command,
        "service_command": service_command,
        "cwd": str(run_dir),
    }
    manifest_path = run_dir / "run_manifest.json"
    _write_json_atomic(manifest_path, base)
    _append_jsonl(args.registry.resolve(), {
        **base,
        "expected_outputs": [
            str(run_dir / name) for name in (
                "CameraTrajectory.txt", "KeyFrameTrajectory.txt", "frame_telemetry.csv",
                "online_summary.json", "online_demo_report.json", "run_manifest.json",
            )
        ],
    })
    running = {**base, "state": "RUNNING"}
    _write_json_atomic(manifest_path, running)
    _append_jsonl(args.registry.resolve(), running)

    service: subprocess.Popen[bytes] | None = None
    runner: subprocess.Popen[bytes] | None = None
    peak_vram: int | None = None
    killed_service = False
    invalid_reason: str | None = None
    started = time.monotonic()
    next_vram_sample = started
    with (run_dir / "service_stdout.log").open("wb") as service_stdout, \
            (run_dir / "service_stderr.log").open("wb") as service_stderr, \
            (run_dir / "runner_stdout.log").open("wb") as runner_stdout, \
            (run_dir / "runner_stderr.log").open("wb") as runner_stderr:
        try:
            service = subprocess.Popen(
                service_command, cwd=run_dir, stdout=service_stdout, stderr=service_stderr
            )
            ready_deadline = time.monotonic() + args.service_ready_timeout
            while _event_count(event_log, "SERVICE_READY") == 0:
                if service.poll() is not None:
                    invalid_reason = f"semantic service exited {service.returncode} before ready"
                    break
                sample = _sample_vram_bytes(service.pid)
                if sample is not None:
                    peak_vram = sample if peak_vram is None else max(peak_vram, sample)
                if time.monotonic() >= ready_deadline:
                    invalid_reason = "semantic service readiness timeout"
                    break
                time.sleep(0.1)
            if invalid_reason is None:
                time.sleep(0.25)
                runner = subprocess.Popen(
                    runner_command, cwd=run_dir, stdout=runner_stdout, stderr=runner_stderr
                )
                deadline = time.monotonic() + args.run_timeout
                while runner.poll() is None:
                    now = time.monotonic()
                    service_exit_code = service.poll()
                    if service_exit_code is None:
                        if now >= next_vram_sample:
                            sample = _sample_vram_bytes(service.pid)
                            if sample is not None:
                                peak_vram = (
                                    sample if peak_vram is None else max(peak_vram, sample)
                                )
                            next_vram_sample = now + 0.5
                    elif not killed_service and invalid_reason is None:
                        invalid_reason = f"semantic service exited unexpectedly ({service.returncode})"
                    if (args.kill_after_packets and not killed_service and
                            _event_count(event_log, "PUBLISHED") >= args.kill_after_packets):
                        _stop_child(service)
                        killed_service = True
                    if time.monotonic() >= deadline:
                        invalid_reason = "online runner timeout"
                        _stop_child(runner)
                        break
                    time.sleep(0.05)
                if runner.returncode != 0 and invalid_reason is None:
                    invalid_reason = f"online runner exited {runner.returncode}"
                if args.kill_after_packets and not killed_service and invalid_reason is None:
                    invalid_reason = "kill trigger was not reached"
        finally:
            if runner is not None:
                _stop_child(runner)
            if service is not None:
                _stop_child(service)

    wall_seconds = time.monotonic() - started
    report_path = run_dir / "online_demo_report.json"
    if (invalid_reason is None and args.service_backend == "real" and
            args.device == "cuda" and peak_vram is None):
        invalid_reason = "CUDA peak VRAM measurement unavailable"
    if invalid_reason is None:
        try:
            report = summarize_online_run(
                run_dir / "frame_telemetry.csv", event_log,
                run_dir / "online_summary.json", watchdog_ms=watchdog_ms,
                request_rate_cap_hz=request_rate_cap_hz,
                max_mask_age_ms=max_mask_age_ms,
                require_kill_transition=bool(args.kill_after_packets),
                allow_degraded_only=(
                    args.service_backend == "real" and not args.kill_after_packets
                ),
                peak_vram_bytes=peak_vram,
            )
            report.update({
                "run_id": run_id,
                "sequence_id": args.sequence,
                "seed": args.seed,
                "service_terminated_after_packets": (
                    args.kill_after_packets if killed_service else None
                ),
            })
            _write_json_atomic(report_path, report)
        except (OSError, ValueError, KeyError) as error:
            invalid_reason = str(error)

    valid = invalid_reason is None
    final: dict[str, object] = {
        **base,
        "state": "COMPLETED" if valid else "FAILED",
        "valid": valid,
        "end_time_utc": _utc_now(),
        "orchestrator_wall_seconds": wall_seconds,
        "service_exit_code": service.returncode if service is not None else None,
        "runner_exit_code": runner.returncode if runner is not None else None,
        "service_killed_by_test": killed_service,
        "invalid_reason": invalid_reason,
        "peak_vram_bytes": peak_vram,
    }
    if valid:
        final["artifacts"] = {
            name: _artifact(run_dir / filename) for name, filename in {
                "trajectory": "CameraTrajectory.txt",
                "keyframe_trajectory": "KeyFrameTrajectory.txt",
                "telemetry": "frame_telemetry.csv",
                "runner_summary": "online_summary.json",
                "online_report": "online_demo_report.json",
                "service_events": "semantic_service_events.jsonl",
                "runner_stdout": "runner_stdout.log",
                "runner_stderr": "runner_stderr.log",
                "service_stdout": "service_stdout.log",
                "service_stderr": "service_stderr.log",
            }.items()
        }
    _write_json_atomic(manifest_path, final)
    _append_jsonl(args.registry.resolve(), final)
    return run_dir, valid, invalid_reason


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="fr3_walking_xyz")
    parser.add_argument("--prompt-set", default="tum_office_v1")
    parser.add_argument("--seed", type=int, default=23011)
    parser.add_argument(
        "--sequence-root", type=Path,
        default=PROJECT_ROOT / "data/tum/raw/rgbd_dataset_freiburg3_walking_xyz",
    )
    parser.add_argument("--settings", type=Path, default=PROJECT_ROOT / "Examples/RGB-D/TUM3.yaml")
    parser.add_argument("--vocabulary", type=Path, default=PROJECT_ROOT / "Vocabulary/ORBvoc.txt")
    parser.add_argument(
        "--executable", type=Path,
        default=PROJECT_ROOT / "Examples/RGB-D/rgbd_tum_online",
    )
    parser.add_argument(
        "--service-python", type=Path,
        default=PROJECT_ROOT / "venv/semantic-gpu/bin/python",
    )
    parser.add_argument(
        "--service-script", type=Path,
        default=PROJECT_ROOT / "tools/run_semantic_service.py",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config/P06_ONLINE_DEMO.json"
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs/p06-online")
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "runs/registry.jsonl")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--service-backend", choices=("real", "protocol-test"), default="real"
    )
    parser.add_argument("--request-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--result-endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--kill-after-packets", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--service-ready-timeout", type=float, default=180.0)
    parser.add_argument("--run-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.kill_after_packets is not None and args.kill_after_packets <= 0:
        parser.error("--kill-after-packets must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir, valid, reason = run_online_demo(args)
    print(f"{'VALID' if valid else 'INVALID'} dir={run_dir} reason={reason}")
    return 0 if valid else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ONLINE_DEMO_FAILED: {error}", file=sys.stderr)
        raise SystemExit(2)
