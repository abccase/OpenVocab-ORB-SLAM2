#!/usr/bin/env python3
"""Analyze only a strictly complete P08 formal matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from semantic_py.openvocab_slam.experiments import (  # noqa: E402
    SEQUENCE_IDS,
    STUDY_ID,
    paired_statistics,
    read_run_matrix,
    sha256_file,
)
from tools.run_study import (  # noqa: E402
    _read_json,
    collect_valid_attempts,
    validate_registration_current,
)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"CSV rows have inconsistent fields: {path}")
    temporary = path.parent / f".{path.name}.partial"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _telemetry_metrics(path: Path, mode: str) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    frame_count = len(rows)
    pose_valid = [int(row["pose_valid"]) for row in rows]
    states = [int(row["tracking_state"]) for row in rows]
    lost = [state == 3 for state in states]
    lost_intervals = 0
    for index, value in enumerate(lost):
        if value and (index == 0 or not lost[index - 1]):
            lost_intervals += 1
    relocalizations = sum(
        states[index - 1] == 3 and states[index] == 2
        for index in range(1, frame_count)
    )
    raw = sum(int(row["raw_keypoints"]) for row in rows)
    used = sum(int(row["used_keypoints"]) for row in rows)
    removed = sum(int(row["removed_dynamic"]) for row in rows)
    retained = sum(int(row["retained_uncertain"]) for row in rows)
    tracking_ms = [float(row["tracking_time_seconds"]) * 1000.0 for row in rows]
    cache_ms = [float(row["cache_load_seconds"]) * 1000.0 for row in rows]
    policy_ms = [float(row["policy_seconds"]) * 1000.0 for row in rows]
    ipc_ms = [float(row["ipc_call_seconds"]) * 1000.0 for row in rows]
    accessed = sum(int(row["semantic_accessed"]) for row in rows)
    return {
        "valid_pose_fraction": sum(pose_valid) / frame_count,
        "lost_frame_fraction": sum(lost) / frame_count,
        "lost_interval_count": lost_intervals,
        "relocalization_count": relocalizations,
        "tracking_time_mean_ms": statistics.fmean(tracking_ms),
        "tracking_time_median_ms": statistics.median(tracking_ms),
        "raw_keypoints_total": raw,
        "used_keypoints_total": used,
        "removed_dynamic_total": removed,
        "retained_uncertain_total": retained,
        "feature_removal_fraction": removed / raw if raw else 0.0,
        "semantic_cache_coverage": accessed / frame_count,
        "cache_load_mean_ms": statistics.fmean(cache_ms),
        "policy_mean_ms": statistics.fmean(policy_ms),
        "ipc_call_mean_ms": statistics.fmean(ipc_ms),
        "ipc_not_applicable": all(row["ipc_reason"] == "NOT_APPLICABLE" for row in rows),
        "degraded_frame_fraction": 0.0,
        "mode_contract": "no_semantic_access" if mode == "baseline" else "immutable_cache_only",
    }


def _cache_diagnostics(cache_root: Path) -> dict[str, object]:
    manifest = _read_json(cache_root / "cache_manifest.json", "dynamic cache manifest")
    frame_count = int(manifest["expected_frame_count"])
    index_count = sum(1 for line in (cache_root / "cache_index.jsonl").read_text().splitlines() if line.strip())
    labels_by_track: dict[int, Counter[str]] = defaultdict(Counter)
    instance_count = 0
    strong_count = 0
    with (cache_root / "dynamic_tracks.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            instance_count += 1
            strong_count += int(bool(row.get("strong_dynamic")))
            track_id = row.get("track_id")
            if isinstance(track_id, int):
                labels_by_track[track_id][str(row.get("label"))] += 1
    stable = sum(max(counter.values()) for counter in labels_by_track.values())
    tracked = sum(sum(counter.values()) for counter in labels_by_track.values())
    return {
        "cache_frame_coverage": index_count / frame_count,
        "instances_per_frame": instance_count / frame_count,
        "strong_dynamic_instance_fraction": strong_count / instance_count if instance_count else 0.0,
        "track_count": len(labels_by_track),
        "label_stability": stable / tracked if tracked else 1.0,
    }


def _map_evidence() -> dict[str, object]:
    result: dict[str, object] = {
        "scope": "P07 representative-map validation; localization formal runs do not rebuild TSDF",
        "representative_maps": {},
    }
    for sequence in ("fr1_desk", "fr3_walking_xyz"):
        path = REPOSITORY_ROOT / "artifacts" / "maps" / f"{sequence}-integrity.json"
        value = _read_json(path, "map integrity")
        if value.get("valid") is not True:
            raise ValueError(f"representative map integrity is invalid: {sequence}")
        result["representative_maps"][sequence] = {
            "integrity_sha256": sha256_file(path),
            "cloud_points": value["cloud_points"],
            "mesh_triangles": value["mesh_triangles"],
            "static_objects": value["static_objects"],
            "dynamic_tracks": value["dynamic_tracks"],
        }
    return result


def _classification(statistics_result: Mapping[str, object]) -> str:
    overall = statistics_result["overall"]
    assert isinstance(overall, Mapping)
    low, high = overall["bootstrap_ci95_m"]
    sequence_results = statistics_result["sequences"]
    assert isinstance(sequence_results, Mapping)
    if high < 0.0:
        return "improvement"
    if low > 0.0:
        return "negative"
    if low <= 0.0 <= high and all(
        result["bootstrap_ci95_m"][0] <= 0.0 <= result["bootstrap_ci95_m"][1]
        for result in sequence_results.values()
    ):
        return "neutral"
    return "mixed"


def _diagnostics(raw_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_sequence: dict[str, object] = {}
    for sequence in SEQUENCE_IDS:
        sequence_rows = [row for row in raw_rows if row["sequence_id"] == sequence]
        semantic = [row for row in sequence_rows if row["mode"] == "semantic-feedback"]
        baseline = [row for row in sequence_rows if row["mode"] == "baseline"]
        removal = statistics.fmean(float(row["feature_removal_fraction"]) for row in semantic)
        pose_delta = statistics.fmean(float(row["valid_pose_fraction"]) for row in semantic) - statistics.fmean(float(row["valid_pose_fraction"]) for row in baseline)
        timing_delta = statistics.fmean(float(row["tracking_time_mean_ms"]) for row in semantic) - statistics.fmean(float(row["tracking_time_mean_ms"]) for row in baseline)
        by_sequence[sequence] = {
            "feature_starvation": removal > 0.5 or pose_delta < -0.10,
            "feature_removal_fraction_mean": removal,
            "valid_pose_fraction_delta": pose_delta,
            "mask_error_signal": sequence in SEQUENCE_IDS[:2] and removal > 0.10,
            "motion_confirmation_lag_signal": float(semantic[0]["strong_dynamic_instance_fraction"]) == 0.0 and removal == 0.0,
            "stale_semantics": False,
            "loop_closure_interaction_signal": sequence == "fr1_room" and pose_delta < 0.0,
            "tracking_time_delta_ms": timing_delta,
            "timing_regression_signal": timing_delta > 5.0,
        }
    return {
        "method": "outcome-blind frozen telemetry/cache diagnostics",
        "thresholds": {
            "feature_starvation_removal_fraction": 0.5,
            "feature_starvation_pose_delta": -0.10,
            "static_control_mask_error_removal_fraction": 0.10,
            "timing_regression_ms": 5.0,
        },
        "sequences": by_sequence,
    }


def analyze(run_root: Path, output: Path) -> dict[str, object]:
    registration = _read_json(run_root / "study_registration.json", "study registration")
    validate_registration_current(registration)
    order = read_run_matrix(run_root / "run_matrix.csv")
    completed, invalid_attempts = collect_valid_attempts(run_root, registration, order)
    if len(completed) != 60:
        raise ValueError(f"strict analysis requires exactly 60 valid runs, found {len(completed)}")
    cache_by_sequence = {
        sequence: _cache_diagnostics(Path(str(registration["datasets"][sequence]["cache_root"])))
        for sequence in SEQUENCE_IDS
    }
    raw_rows: list[dict[str, object]] = []
    for condition in order:
        attempt, metric_payload = completed[int(condition["order_index"])]
        telemetry = _telemetry_metrics(attempt / "frame_telemetry.csv", str(condition["mode"]))
        metrics = metric_payload["metrics"]
        cache = cache_by_sequence[str(condition["sequence_id"])]
        raw_rows.append({
            "order_index": condition["order_index"],
            "sequence_id": condition["sequence_id"],
            "seed": condition["seed"],
            "mode": condition["mode"],
            "run_dir": str(attempt.resolve()),
            "trajectory_sha256": metric_payload["trajectory_sha256"],
            "groundtruth_sha256": metric_payload["groundtruth_sha256"],
            "associated_pose_count": metrics["associated_pose_count"],
            "ate_translation_rmse_m": metrics["ate_translation_rmse_m"],
            "rpe_pair_count": metrics["rpe_pair_count"],
            "rpe_translation_rmse_m": metrics["rpe_translation_rmse_m"],
            "rpe_rotation_rmse_deg": metrics["rpe_rotation_rmse_deg"],
            **telemetry,
            **cache,
        })
    paired = paired_statistics(raw_rows)
    paired_rows = [dict(row) for row in paired["pairs"]]
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.csv"
    paired_path = output / "paired_results.csv"
    order_copy = output / "run_matrix.csv"
    _write_csv(metrics_path, raw_rows)
    _write_csv(paired_path, paired_rows)
    order_copy.write_bytes((run_root / "run_matrix.csv").read_bytes())
    diagnostics = _diagnostics(raw_rows)
    diagnostics_path = output / "diagnostics.json"
    _write_json_atomic(diagnostics_path, diagnostics)
    map_evidence = _map_evidence()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_root = output / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    figure_path = figure_root / "paired_ate_delta.png"
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for index, sequence in enumerate(SEQUENCE_IDS):
        values = [row["ate_delta_m"] for row in paired_rows if row["sequence_id"] == sequence]
        axis.scatter([index] * len(values), values, alpha=0.8)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(range(len(SEQUENCE_IDS)), SEQUENCE_IDS, rotation=25, ha="right")
    axis.set_ylabel("semantic-feedback minus baseline ATE RMSE (m)")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    summary = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "valid_run_count": 60,
        "invalid_attempt_count": len(invalid_attempts),
        "run_order_sha256": registration["run_order_sha256"],
        "producer_commit": registration["producer_commit"],
        "metric_protocol": {
            "alignment": "SE3", "scale_alignment": False,
            "association_max_seconds": 0.02, "rpe_delta_seconds": 1.0,
        },
        "paired_statistics": paired,
        "outcome_classification": _classification(paired),
        "diagnostics": diagnostics,
        "map_metrics": map_evidence,
        "artifacts": {
            "run_matrix": {"path": str(order_copy), "sha256": sha256_file(order_copy)},
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "paired_results": {"path": str(paired_path), "sha256": sha256_file(paired_path)},
            "diagnostics": {"path": str(diagnostics_path), "sha256": sha256_file(diagnostics_path)},
            "figure": {"path": str(figure_path), "sha256": sha256_file(figure_path)},
        },
    }
    _write_json_atomic(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=REPOSITORY_ROOT / "runs" / STUDY_ID)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.runs, args.output)
        print(json.dumps({
            "study_id": result["study_id"],
            "valid_run_count": result["valid_run_count"],
            "outcome_classification": result["outcome_classification"],
            "artifacts": result["artifacts"],
        }, indent=2, sort_keys=True))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"P08_ANALYZE INVALID: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
