"""Frozen protocol identities for the P05 baseline noninferiority study."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path


STUDY_ID = "ovorb2_p05_baseline_noninferiority_v2"
ORACLE_COMMIT = "58014b7c1f2b73427b67b4e80a8cf334127f48ea"
CANDIDATE_POLICY = "HEAD_AT_REGISTRATION"
SEQUENCE_IDS = (
    "fr1_desk",
    "fr1_room",
    "fr3_sitting_xyz",
    "fr3_sitting_halfsphere",
    "fr3_walking_xyz",
    "fr3_walking_halfsphere",
)
REPETITION_IDS = tuple(range(23011, 23026))
IMPLEMENTATIONS = ("oracle", "candidate")

EXPECTED_STATISTICS = {
    "algorithm": "paired_bootstrap",
    "generator": "PCG64",
    "seed": 23010,
    "resamples": 100000,
    "generator_scope": "reinitialize_per_sequence",
    "shared_index_matrix_for_metrics": True,
    "quantile_method": "linear",
    "confidence_level_one_sided": 0.95,
    "pose_delta_lower_margin": -0.10,
    "ate_geometric_ratio_upper_margin": 1.25,
}

EXPECTED_METRICS = {
    "pose_delta": "candidate_valid_pose_fraction_minus_oracle",
    "ate_log_ratio": "log_candidate_ate_over_oracle_ate",
    "trajectory_alignment": "SE3",
    "scale_alignment": False,
    "timestamp_association_max_seconds": 0.02,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected a JSON object")
    return value


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def expected_blocks() -> list[dict[str, object]]:
    rng = random.Random(23010)
    blocks: list[dict[str, object]] = []
    for sequence_id in SEQUENCE_IDS:
        extra_first = rng.choice(IMPLEMENTATIONS)
        other = "candidate" if extra_first == "oracle" else "oracle"
        first_positions = [extra_first] * 8 + [other] * 7
        rng.shuffle(first_positions)
        for repetition_id, first in zip(REPETITION_IDS, first_positions, strict=True):
            second = "candidate" if first == "oracle" else "oracle"
            blocks.append(
                {
                    "block_id": f"{sequence_id}-rep-{repetition_id}",
                    "sequence_id": sequence_id,
                    "repetition_id": repetition_id,
                    "execution_order": [first, second],
                }
            )
    return blocks


def load_protocol(path: Path, experiment_path: Path) -> dict[str, object]:
    protocol = _read_json_object(path, "P05 protocol manifest")
    experiment = _read_json_object(experiment_path, "experiment manifest")

    if protocol.get("schema_version") != 1:
        raise ValueError("P05 protocol schema version mismatch")
    if protocol.get("study_id") != STUDY_ID:
        raise ValueError("P05 protocol study identity mismatch")
    if protocol.get("experiment_manifest") != "config/EXPERIMENT_MANIFEST.yaml":
        raise ValueError("P05 experiment manifest reference mismatch")
    if protocol.get("oracle") != {"producer_commit": ORACLE_COMMIT}:
        raise ValueError("P05 oracle identity mismatch")
    if protocol.get("candidate") != {"producer_policy": CANDIDATE_POLICY}:
        raise ValueError("P05 candidate policy mismatch")
    if protocol.get("sequence_ids") != list(SEQUENCE_IDS):
        raise ValueError("P05 sequence identity or order mismatch")
    if protocol.get("repetition_ids") != list(REPETITION_IDS):
        raise ValueError("P05 repetition identity mismatch")
    if protocol.get("metrics") != EXPECTED_METRICS:
        raise ValueError("P05 metrics configuration mismatch")
    if protocol.get("statistics") != EXPECTED_STATISTICS:
        raise ValueError("P05 statistics configuration mismatch")
    if protocol.get("blocks") != expected_blocks():
        raise ValueError("P05 blocks or execution order mismatch")

    datasets = experiment.get("datasets")
    if not isinstance(datasets, list) or not all(isinstance(row, dict) for row in datasets):
        raise ValueError("experiment dataset manifest is invalid")
    experiment_ids = [row.get("id") for row in datasets]
    if experiment_ids != list(SEQUENCE_IDS):
        raise ValueError("experiment dataset IDs differ from P05 protocol")
    return protocol


def validate_batch_registration(
    registration: Mapping[str, object],
    protocol: Mapping[str, object],
    protocol_sha256: str,
) -> None:
    if registration.get("schema_version") != 1:
        raise ValueError("batch registration schema version mismatch")
    if registration.get("state") != "REGISTERED":
        raise ValueError("batch registration state mismatch")
    if registration.get("study_id") != STUDY_ID:
        raise ValueError("batch registration study identity mismatch")
    if registration.get("protocol_manifest_sha256") != protocol_sha256:
        raise ValueError("batch registration protocol hash mismatch")
    if not _is_lower_hex(protocol_sha256, 64):
        raise ValueError("trusted protocol hash is invalid")
    if registration.get("candidate_policy") != CANDIDATE_POLICY:
        raise ValueError("batch registration candidate policy mismatch")

    oracle = registration.get("oracle")
    candidate = registration.get("candidate")
    if not isinstance(oracle, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("batch registration implementation identities are missing")
    if oracle.get("producer_commit") != ORACLE_COMMIT:
        raise ValueError("batch registration oracle producer mismatch")
    if not _is_lower_hex(oracle.get("executable_sha256"), 64):
        raise ValueError("batch registration oracle executable hash is invalid")
    if not _is_lower_hex(oracle.get("build_manifest_sha256"), 64):
        raise ValueError("batch registration oracle build-manifest hash is invalid")
    if not _is_lower_hex(candidate.get("producer_commit"), 40):
        raise ValueError("batch registration candidate producer is invalid")
    if not _is_lower_hex(candidate.get("executable_sha256"), 64):
        raise ValueError("batch registration candidate executable hash is invalid")
    if protocol.get("study_id") != STUDY_ID:
        raise ValueError("trusted protocol object has wrong study identity")
    if protocol.get("oracle") != {"producer_commit": ORACLE_COMMIT}:
        raise ValueError("trusted protocol object has wrong oracle identity")
    if protocol.get("candidate") != {"producer_policy": CANDIDATE_POLICY}:
        raise ValueError("trusted protocol object has wrong candidate policy")
