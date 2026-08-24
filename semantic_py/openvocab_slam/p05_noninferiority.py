"""Pure paired-bootstrap statistics for the frozen P05 V2 gate."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

import numpy as np

from semantic_py.openvocab_slam.p05_protocol import (
    EXPECTED_STATISTICS,
    REPETITION_IDS,
    SEQUENCE_IDS,
)


MeasuredRow = Mapping[str, object]
ValidatedRows = dict[int, tuple[float, float]]


def bootstrap_indices(pair_count: int, resamples: int, seed: int) -> np.ndarray:
    if pair_count != 15 or resamples != 100000 or seed != 23010:
        raise ValueError("bootstrap configuration differs from frozen protocol")
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.integers(
        0,
        pair_count,
        size=(resamples, pair_count),
        dtype=np.int16,
    )


def _validated_by_repetition(rows: Sequence[MeasuredRow], label: str) -> ValidatedRows:
    if len(rows) != 15:
        raise ValueError(f"{label} must contain exactly 15 paired runs")
    values: ValidatedRows = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} row is not a mapping")
        repetition = row.get("repetition_id")
        if isinstance(repetition, bool) or not isinstance(repetition, int):
            raise ValueError(f"{label} repetition ID is not an integer")
        if repetition in values:
            raise ValueError(f"{label} contains duplicate repetition {repetition}")
        try:
            pose = float(row["valid_pose_fraction"])
            ate = float(row["ate_rmse_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} measurement is incomplete") from exc
        if not math.isfinite(pose) or not 0.0 <= pose <= 1.0:
            raise ValueError(f"{label} has invalid valid-pose fraction")
        if not math.isfinite(ate) or ate <= 0.0:
            raise ValueError(f"{label} has invalid ATE")
        values[repetition] = (pose, ate)
    if set(values) != set(REPETITION_IDS):
        raise ValueError(f"{label} repetition IDs differ from the frozen protocol")
    return {repetition: values[repetition] for repetition in REPETITION_IDS}


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _unpaired_summaries(
    oracle: ValidatedRows,
    candidate: ValidatedRows,
) -> dict[str, object]:
    return {
        "oracle": {
            "valid_pose_fraction": _summary([value[0] for value in oracle.values()]),
            "ate_rmse_m": _summary([value[1] for value in oracle.values()]),
        },
        "candidate": {
            "valid_pose_fraction": _summary([value[0] for value in candidate.values()]),
            "ate_rmse_m": _summary([value[1] for value in candidate.values()]),
        },
    }


def evaluate_sequence(
    oracle_rows: Sequence[MeasuredRow],
    candidate_rows: Sequence[MeasuredRow],
    statistics_config: Mapping[str, object],
) -> dict[str, object]:
    if dict(statistics_config) != EXPECTED_STATISTICS:
        raise ValueError("statistics configuration differs from frozen protocol")
    oracle = _validated_by_repetition(oracle_rows, "oracle")
    candidate = _validated_by_repetition(candidate_rows, "candidate")
    if tuple(oracle) != tuple(candidate):
        raise ValueError("oracle and candidate repetition IDs are not paired")

    repetitions = tuple(oracle)
    pose = np.asarray(
        [candidate[key][0] - oracle[key][0] for key in repetitions],
        dtype=np.float64,
    )
    ate_log = np.asarray(
        [math.log(candidate[key][1] / oracle[key][1]) for key in repetitions],
        dtype=np.float64,
    )
    indices = bootstrap_indices(15, 100000, 23010)
    pose_means = pose[indices].mean(axis=1)
    ate_log_means = ate_log[indices].mean(axis=1)
    pose_lower = float(np.quantile(pose_means, 0.05, method="linear"))
    ate_upper = float(
        math.exp(np.quantile(ate_log_means, 0.95, method="linear"))
    )
    pose_pass = pose_lower >= -0.10
    ate_pass = ate_upper <= 1.25

    return {
        "valid": pose_pass and ate_pass,
        "pose_pass": pose_pass,
        "ate_pass": ate_pass,
        "pose_estimate": float(pose.mean()),
        "pose_lower_95": pose_lower,
        "pose_two_sided_95": [
            float(np.quantile(pose_means, 0.025, method="linear")),
            float(np.quantile(pose_means, 0.975, method="linear")),
        ],
        "ate_geometric_ratio_estimate": float(math.exp(ate_log.mean())),
        "ate_ratio_upper_95": ate_upper,
        "ate_ratio_two_sided_95": [
            float(math.exp(np.quantile(ate_log_means, 0.025, method="linear"))),
            float(math.exp(np.quantile(ate_log_means, 0.975, method="linear"))),
        ],
        "paired_values": [
            {
                "repetition_id": key,
                "oracle_valid_pose_fraction": oracle[key][0],
                "candidate_valid_pose_fraction": candidate[key][0],
                "pose_delta": candidate[key][0] - oracle[key][0],
                "oracle_ate_rmse_m": oracle[key][1],
                "candidate_ate_rmse_m": candidate[key][1],
                "ate_log_ratio": math.log(candidate[key][1] / oracle[key][1]),
            }
            for key in repetitions
        ],
        "margins": {
            "pose_delta_lower": -0.10,
            "ate_ratio_upper": 1.25,
        },
        "bootstrap": {
            "generator": "PCG64",
            "seed": 23010,
            "resamples": 100000,
            "quantile_method": "linear",
            "numpy_version": np.__version__,
        },
        "unpaired_summaries": _unpaired_summaries(oracle, candidate),
    }


def evaluate_study(
    rows_by_sequence: Mapping[
        str,
        tuple[Sequence[MeasuredRow], Sequence[MeasuredRow]],
    ],
    sequence_ids: Sequence[str],
    statistics_config: Mapping[str, object],
) -> dict[str, object]:
    if tuple(sequence_ids) != SEQUENCE_IDS:
        raise ValueError("requested sequences differ from the frozen protocol")
    if set(rows_by_sequence) != set(SEQUENCE_IDS):
        raise ValueError("study sequences differ from the frozen protocol")
    sequences = {
        sequence_id: evaluate_sequence(
            rows_by_sequence[sequence_id][0],
            rows_by_sequence[sequence_id][1],
            statistics_config,
        )
        for sequence_id in SEQUENCE_IDS
    }
    return {
        "valid": all(bool(value["valid"]) for value in sequences.values()),
        "sequences": sequences,
    }
