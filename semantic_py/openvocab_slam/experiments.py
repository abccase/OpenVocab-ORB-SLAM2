"""Frozen P08 study identities, trajectory metrics, and paired statistics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STUDY_ID = "ovorb2_tum_v1"
ORDER_SEED = 23010
BOOTSTRAP_SEED = 23010
BOOTSTRAP_RESAMPLES = 100000
SEQUENCE_IDS = (
    "fr1_desk",
    "fr1_room",
    "fr3_sitting_xyz",
    "fr3_sitting_halfsphere",
    "fr3_walking_xyz",
    "fr3_walking_halfsphere",
)
MODES = ("baseline", "semantic-feedback")
SEEDS = (23011, 23012, 23013, 23014, 23015)
ASSOCIATION_MAX_SECONDS = 0.02
RPE_DELTA_SECONDS = 1.0


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def load_experiment_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid experiment manifest: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("experiment manifest schema mismatch")
    if value.get("study_id") != STUDY_ID:
        raise ValueError("experiment manifest study mismatch")
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or [row.get("id") for row in datasets] != list(SEQUENCE_IDS):
        raise ValueError("experiment manifest datasets differ from frozen protocol")
    if value.get("modes") != list(MODES):
        raise ValueError("experiment manifest modes differ from frozen protocol")
    if value.get("repetitions") != 5 or value.get("random_seeds") != list(SEEDS):
        raise ValueError("experiment manifest repetitions differ from frozen protocol")
    if value.get("playback") != "dataset_timestamp_paced":
        raise ValueError("experiment manifest playback differs from frozen protocol")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or (
        metrics.get("trajectory_alignment") != "SE3"
        or metrics.get("ate") != "translation_rmse_m"
        or metrics.get("rpe_delta") != RPE_DELTA_SECONDS
        or metrics.get("rpe_delta_unit") != "seconds"
    ):
        raise ValueError("experiment manifest metrics differ from frozen protocol")
    statistics_config = value.get("statistics")
    if not isinstance(statistics_config, dict) or (
        statistics_config.get("unit") != "paired_run_by_sequence_and_seed"
        or statistics_config.get("primary_contrast")
        != "semantic-feedback_minus_baseline_ATE_RMSE"
        or statistics_config.get("confidence_interval")
        != "paired_bootstrap_95_percent"
        or statistics_config.get("report_all_sequences") is not True
        or statistics_config.get("positive_result_required") is not False
    ):
        raise ValueError("experiment manifest statistics differ from frozen protocol")
    return value


def build_run_matrix(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    if manifest.get("study_id") != STUDY_ID:
        raise ValueError("cannot build matrix for another study")
    rng = random.Random(ORDER_SEED)
    extra_modes = [MODES[0]] * 3 + [MODES[1]] * 3
    rng.shuffle(extra_modes)
    blocks_with_order: list[tuple[str, int, str]] = []
    for sequence, extra_mode in zip(SEQUENCE_IDS, extra_modes, strict=True):
        other_mode = MODES[1] if extra_mode == MODES[0] else MODES[0]
        first_modes = [extra_mode] * 3 + [other_mode] * 2
        rng.shuffle(first_modes)
        blocks_with_order.extend(
            (sequence, seed, first)
            for seed, first in zip(SEEDS, first_modes, strict=True)
        )
    rng.shuffle(blocks_with_order)
    rows: list[dict[str, object]] = []
    for sequence, seed, first in blocks_with_order:
        second = MODES[1] if first == MODES[0] else MODES[0]
        block_id = f"{sequence}-seed-{seed}"
        for mode in (first, second):
            rows.append(
                {
                    "order_index": len(rows),
                    "block_id": block_id,
                    "sequence_id": sequence,
                    "mode": mode,
                    "seed": seed,
                }
            )
    return rows


def _matrix_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=["order_index", "block_id", "sequence_id", "mode", "seed"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def freeze_run_matrix(rows: Sequence[Mapping[str, object]], path: Path) -> str:
    expected = {
        (sequence, mode, seed)
        for sequence in SEQUENCE_IDS
        for mode in MODES
        for seed in SEEDS
    }
    actual = {(row.get("sequence_id"), row.get("mode"), row.get("seed")) for row in rows}
    if len(rows) != 60 or actual != expected:
        raise ValueError("run matrix differs from frozen 60-condition protocol")
    content = _matrix_bytes(rows)
    path = Path(path)
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError("existing frozen run order differs from requested matrix")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.partial"
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    return sha256_file(path)


def read_run_matrix(path: Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    parsed = [
        {
            "order_index": int(row["order_index"]),
            "block_id": row["block_id"],
            "sequence_id": row["sequence_id"],
            "mode": row["mode"],
            "seed": int(row["seed"]),
        }
        for row in rows
    ]
    if [row["order_index"] for row in parsed] != list(range(60)):
        raise ValueError("frozen run order indices are not contiguous")
    freeze_run_matrix(parsed, path)
    return parsed


def _invalid(reason: str) -> ValidationResult:
    return ValidationResult(False, reason)


def validate_run_manifest(
    run: Mapping[str, object],
    condition: Mapping[str, object],
    registration: Mapping[str, object],
) -> ValidationResult:
    sequence = condition.get("sequence_id")
    mode = condition.get("mode")
    seed = condition.get("seed")
    if run.get("study") != STUDY_ID or registration.get("study_id") != STUDY_ID:
        return _invalid("study identity mismatch")
    if run.get("sequence_id") != sequence or run.get("mode") != mode:
        return _invalid("sequence or mode mismatch")
    if isinstance(seed, bool) or not isinstance(seed, int) or run.get("seed") != seed:
        return _invalid("seed identity mismatch")
    if run.get("state") != "COMPLETED" or run.get("valid") is not True or run.get("exit_code") != 0:
        return _invalid("process completion mismatch")
    if run.get("degraded") is not False:
        return _invalid("formal run is degraded or degradation status is missing")
    if run.get("producer_commit") != registration.get("producer_commit"):
        return _invalid("producer commit mismatch")
    if run.get("compatibility_commit") != registration.get("compatibility_commit"):
        return _invalid("compatibility commit mismatch")
    expected_formal = {
        "study_id": STUDY_ID,
        "block_id": condition.get("block_id"),
        "mode": mode,
        "protocol_manifest_sha256": registration.get("experiment_manifest_sha256"),
        "run_order_sha256": registration.get("run_order_sha256"),
        "producer_commit": registration.get("producer_commit"),
    }
    if run.get("formal_identity") != expected_formal:
        return _invalid("formal run identity mismatch")
    datasets = registration.get("datasets")
    if not isinstance(datasets, Mapping) or not isinstance(datasets.get(sequence), Mapping):
        return _invalid("registered dataset identity is missing")
    dataset = datasets[sequence]
    verified = run.get("verified_inputs")
    if not isinstance(verified, Mapping) or verified.get("source_tree_sha256") != dataset.get("source_tree_sha256"):
        return _invalid("source tree identity mismatch")
    cache = run.get("cache_identity")
    expected_cache = dataset.get("cache_identity")
    if mode == "baseline":
        if cache is not None:
            return _invalid("baseline run received semantic cache assets")
    elif cache != expected_cache or not isinstance(cache, Mapping):
        return _invalid("semantic cache identity mismatch")
    elif (
        verified.get("dynamic_manifest_sha256") != expected_cache.get("manifest_sha256")
        or verified.get("dynamic_completion_sha256") != expected_cache.get("completion_sha256")
        or verified.get("dynamic_index_sha256") != expected_cache.get("index_sha256")
        or verified.get("prompt_sha256") != dataset.get("prompt_sha256")
        or verified.get("dynamic_config_sha256") != dataset.get("configuration_sha256")
    ):
        return _invalid("semantic cache verified inputs mismatch")
    expected_frames = run.get("expected_frames")
    telemetry = run.get("telemetry")
    if (
        isinstance(expected_frames, bool)
        or not isinstance(expected_frames, int)
        or expected_frames <= 0
        or run.get("frame_count") != expected_frames
        or not isinstance(telemetry, Mapping)
        or telemetry.get("row_count") != expected_frames
        or not _is_hex(telemetry.get("sha256"), 64)
    ):
        return _invalid("telemetry coverage is incomplete")
    trajectory = run.get("trajectory")
    if not isinstance(trajectory, Mapping) or not _is_hex(trajectory.get("sha256"), 64):
        return _invalid("trajectory parse or identity is missing")
    metric = run.get("metric")
    if not isinstance(metric, Mapping) or metric.get("alignment") != "SE3":
        return _invalid("metric alignment must be SE3")
    if metric.get("association_max_seconds") != ASSOCIATION_MAX_SECONDS:
        return _invalid("metric timestamp association differs from protocol")
    if metric.get("rpe_delta_seconds") != RPE_DELTA_SECONDS:
        return _invalid("metric RPE delta differs from protocol")
    if not _is_hex(metric.get("output_sha256"), 64):
        return _invalid("metric output identity is missing")
    return ValidationResult(True, "valid")


def _validated_trajectory(rows: Sequence[Sequence[float]], label: str) -> list[tuple[float, ...]]:
    result: list[tuple[float, ...]] = []
    previous = -math.inf
    for raw in rows:
        if len(raw) != 8:
            raise ValueError(f"{label} trajectory row must contain eight values")
        row = tuple(float(value) for value in raw)
        if not all(math.isfinite(value) for value in row) or row[0] <= previous:
            raise ValueError(f"{label} trajectory is non-finite or non-monotonic")
        norm = math.sqrt(sum(value * value for value in row[4:8]))
        if abs(norm - 1.0) > 1e-5:
            raise ValueError(f"{label} trajectory quaternion is not normalized")
        previous = row[0]
        result.append(row)
    if len(result) < 3:
        raise ValueError("ATE requires at least three associated poses")
    return result


def _associate(
    groundtruth: Sequence[tuple[float, ...]],
    estimate: Sequence[tuple[float, ...]],
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(groundtruth):
        for estimate_index, est in enumerate(estimate):
            difference = abs(gt[0] - est[0])
            if difference <= ASSOCIATION_MAX_SECONDS:
                candidates.append((difference, gt_index, estimate_index))
    used_gt: set[int] = set()
    used_estimate: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gt_index, estimate_index in sorted(candidates):
        if gt_index not in used_gt and estimate_index not in used_estimate:
            used_gt.add(gt_index)
            used_estimate.add(estimate_index)
            matches.append((gt_index, estimate_index))
    matches.sort()
    if len(matches) < 3:
        raise ValueError("ATE requires at least three associated poses within 0.02 seconds")
    return [(groundtruth[gt], estimate[est]) for gt, est in matches]


def _quaternion_rotation(row: Sequence[float]) -> np.ndarray:
    x, y, z, w = row[4:8]
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_matrix(row: Sequence[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _quaternion_rotation(row)
    pose[:3, 3] = row[1:4]
    return pose


def _se3_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def compute_trajectory_metrics(
    groundtruth_rows: Sequence[Sequence[float]],
    estimate_rows: Sequence[Sequence[float]],
) -> dict[str, object]:
    groundtruth = _validated_trajectory(groundtruth_rows, "groundtruth")
    estimate = _validated_trajectory(estimate_rows, "estimate")
    associated = _associate(groundtruth, estimate)
    target_xyz = np.asarray([row[0][1:4] for row in associated], dtype=np.float64)
    source_xyz = np.asarray([row[1][1:4] for row in associated], dtype=np.float64)
    rotation, translation = _se3_alignment(source_xyz, target_xyz)
    aligned_xyz = (rotation @ source_xyz.T).T + translation
    ate_errors = np.linalg.norm(target_xyz - aligned_xyz, axis=1)

    aligned_estimates: list[np.ndarray] = []
    gt_poses: list[np.ndarray] = []
    timestamps: list[float] = []
    alignment_pose = np.eye(4, dtype=np.float64)
    alignment_pose[:3, :3] = rotation
    alignment_pose[:3, 3] = translation
    for groundtruth_row, estimate_row in associated:
        timestamps.append(groundtruth_row[0])
        gt_poses.append(_pose_matrix(groundtruth_row))
        aligned_estimates.append(alignment_pose @ _pose_matrix(estimate_row))

    rpe_translation: list[float] = []
    rpe_rotation: list[float] = []
    for index, timestamp in enumerate(timestamps):
        target_time = timestamp + RPE_DELTA_SECONDS
        later = [
            (abs(candidate - target_time), candidate_index)
            for candidate_index, candidate in enumerate(timestamps)
            if candidate_index > index
        ]
        if not later:
            continue
        difference, later_index = min(later)
        if difference > ASSOCIATION_MAX_SECONDS:
            continue
        gt_delta = np.linalg.inv(gt_poses[index]) @ gt_poses[later_index]
        est_delta = np.linalg.inv(aligned_estimates[index]) @ aligned_estimates[later_index]
        error = np.linalg.inv(gt_delta) @ est_delta
        rpe_translation.append(float(np.linalg.norm(error[:3, 3])))
        cosine = min(1.0, max(-1.0, (float(np.trace(error[:3, :3])) - 1.0) / 2.0))
        rpe_rotation.append(math.degrees(math.acos(cosine)))
    if not rpe_translation:
        raise ValueError("no pose pairs satisfy the frozen one-second RPE delta")
    return {
        "alignment": "SE3",
        "scale": 1.0,
        "association_max_seconds": ASSOCIATION_MAX_SECONDS,
        "associated_pose_count": len(associated),
        "ate_translation_rmse_m": float(math.sqrt(float(np.mean(ate_errors**2)))),
        "rpe_delta_seconds": RPE_DELTA_SECONDS,
        "rpe_pair_count": len(rpe_translation),
        "rpe_translation_rmse_m": float(math.sqrt(float(np.mean(np.square(rpe_translation))))),
        "rpe_rotation_rmse_deg": float(math.sqrt(float(np.mean(np.square(rpe_rotation))))),
    }


def _bootstrap_ci(values: np.ndarray) -> list[float]:
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = generator.integers(
        0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)), dtype=np.int16
    )
    estimates = np.median(values[indices], axis=1)
    return [
        float(np.quantile(estimates, 0.025, method="linear")),
        float(np.quantile(estimates, 0.975, method="linear")),
    ]


def paired_statistics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    indexed: dict[tuple[str, int, str], float] = {}
    for row in rows:
        key = (str(row.get("sequence_id")), int(row.get("seed", -1)), str(row.get("mode")))
        if key in indexed:
            raise ValueError(f"duplicate raw metric row: {key}")
        try:
            ate = float(row["ate_translation_rmse_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("raw metric row lacks finite ATE") from exc
        if not math.isfinite(ate) or ate < 0.0:
            raise ValueError("raw metric row lacks finite ATE")
        indexed[key] = ate
    if len(rows) != 60:
        raise ValueError("raw metric table must contain exactly 60 runs")
    expected = {
        (sequence, seed, mode)
        for sequence in SEQUENCE_IDS
        for seed in SEEDS
        for mode in MODES
    }
    if set(indexed) != expected:
        raise ValueError("raw metric sequences, modes, or seeds differ from frozen protocol")
    pairs: list[dict[str, object]] = []
    sequence_results: dict[str, object] = {}
    for sequence in SEQUENCE_IDS:
        differences = []
        for seed in SEEDS:
            baseline = indexed[(sequence, seed, "baseline")]
            semantic = indexed[(sequence, seed, "semantic-feedback")]
            difference = semantic - baseline
            differences.append(difference)
            pairs.append(
                {
                    "sequence_id": sequence,
                    "seed": seed,
                    "baseline_ate_rmse_m": baseline,
                    "semantic_feedback_ate_rmse_m": semantic,
                    "ate_delta_m": difference,
                }
            )
        values = np.asarray(differences, dtype=np.float64)
        sequence_results[sequence] = {
            "pair_count": 5,
            "mean_ate_delta_m": float(statistics.fmean(differences)),
            "median_ate_delta_m": float(statistics.median(differences)),
            "bootstrap_ci95_m": _bootstrap_ci(values),
        }
    all_differences = np.asarray([float(pair["ate_delta_m"]) for pair in pairs])
    return {
        "sequences": sequence_results,
        "pairs": pairs,
        "overall": {
            "pair_count": 30,
            "mean_ate_delta_m": float(np.mean(all_differences)),
            "median_ate_delta_m": float(np.median(all_differences)),
            "bootstrap_ci95_m": _bootstrap_ci(all_differences),
        },
        "bootstrap": {
            "algorithm": "paired_bootstrap_median",
            "generator": "PCG64",
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_interval": 0.95,
            "quantile_method": "linear",
            "numpy_version": np.__version__,
        },
    }
