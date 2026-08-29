from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from semantic_py.openvocab_slam.experiments import (
    MODES,
    SEEDS,
    SEQUENCE_IDS,
    build_run_matrix,
    freeze_run_matrix,
    load_experiment_manifest,
    validate_run_manifest,
)
from tools.run_orb_tum import RunCondition, _validated_formal_identity


ROOT = Path(__file__).resolve().parents[2]


def _valid_run() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    condition = {
        "order_index": 0,
        "block_id": "fr1_desk-seed-23011",
        "sequence_id": "fr1_desk",
        "mode": "semantic-feedback",
        "seed": 23011,
    }
    registration = {
        "study_id": "ovorb2_tum_v1",
        "producer_commit": "a" * 40,
        "compatibility_commit": "f2e6f51cdc8d067655d90a78c06261378e07e8f3",
        "experiment_manifest_sha256": "b" * 64,
        "run_order_sha256": "c" * 64,
        "datasets": {
            "fr1_desk": {
                "source_tree_sha256": "d" * 64,
                "cache_identity": {
                    "manifest_sha256": "e" * 64,
                    "completion_sha256": "f" * 64,
                    "index_sha256": "0" * 64,
                },
                "prompt_sha256": "1" * 64,
                "configuration_sha256": "2" * 64,
            }
        },
    }
    run = {
        "schema_version": 2,
        "state": "COMPLETED",
        "valid": True,
        "study": "ovorb2_tum_v1",
        "sequence_id": "fr1_desk",
        "mode": "semantic-feedback",
        "seed": 23011,
        "producer_commit": "a" * 40,
        "compatibility_commit": "f2e6f51cdc8d067655d90a78c06261378e07e8f3",
        "formal_identity": {
            "study_id": "ovorb2_tum_v1",
            "block_id": "fr1_desk-seed-23011",
            "mode": "semantic-feedback",
            "protocol_manifest_sha256": "b" * 64,
            "run_order_sha256": "c" * 64,
            "producer_commit": "a" * 40,
        },
        "verified_inputs": {
            "source_tree_sha256": "d" * 64,
            "dynamic_manifest_sha256": "e" * 64,
            "dynamic_completion_sha256": "f" * 64,
            "dynamic_index_sha256": "0" * 64,
            "prompt_sha256": "1" * 64,
            "dynamic_config_sha256": "2" * 64,
        },
        "cache_identity": {
            "manifest_sha256": "e" * 64,
            "completion_sha256": "f" * 64,
            "index_sha256": "0" * 64,
        },
        "pacing": "dataset_timestamp_paced_relative",
        "degraded": False,
        "exit_code": 0,
        "expected_frames": 2,
        "frame_count": 2,
        "telemetry": {"sha256": "3" * 64, "row_count": 2},
        "trajectory": {"sha256": "4" * 64, "pose_count": 2},
        "metric": {
            "alignment": "SE3",
            "association_max_seconds": 0.02,
            "rpe_delta_seconds": 1.0,
            "output_sha256": "5" * 64,
        },
    }
    return run, condition, registration


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_seed", "seed"),
        ("wrong_cache", "cache"),
        ("degraded_formal_run", "degraded"),
        ("wrong_alignment", "alignment"),
        ("incomplete_telemetry", "telemetry"),
    ],
)
def test_invalid_run_never_enters_metrics(mutation: str, message: str) -> None:
    run, condition, registration = _valid_run()
    if mutation == "missing_seed":
        run.pop("seed")
    elif mutation == "wrong_cache":
        run["cache_identity"] = dict(run["cache_identity"], index_sha256="9" * 64)
    elif mutation == "degraded_formal_run":
        run["degraded"] = True
    elif mutation == "wrong_alignment":
        run["metric"] = dict(run["metric"], alignment="Sim3")
    elif mutation == "incomplete_telemetry":
        run["telemetry"] = dict(run["telemetry"], row_count=1)
    result = validate_run_manifest(run, condition, registration)
    assert result.valid is False
    assert message in result.reason.lower()


def test_baseline_rejects_any_semantic_cache_binding() -> None:
    run, condition, registration = _valid_run()
    condition["mode"] = "baseline"
    run["mode"] = "baseline"
    run["formal_identity"] = dict(run["formal_identity"], mode="baseline")
    result = validate_run_manifest(run, condition, registration)
    assert not result.valid
    assert "baseline" in result.reason


def test_frozen_order_is_balanced_unique_and_immutable(tmp_path: Path) -> None:
    manifest = load_experiment_manifest(ROOT / "config/EXPERIMENT_MANIFEST.yaml")
    rows = build_run_matrix(manifest)
    assert len(rows) == 60
    assert {
        (row["sequence_id"], row["mode"], row["seed"]) for row in rows
    } == {
        (sequence, mode, seed)
        for sequence in SEQUENCE_IDS
        for mode in MODES
        for seed in SEEDS
    }
    blocks = [rows[index : index + 2] for index in range(0, 60, 2)]
    assert all(block[0]["block_id"] == block[1]["block_id"] for block in blocks)
    assert sum(block[0]["mode"] == "baseline" for block in blocks) == 15
    for sequence in SEQUENCE_IDS:
        sequence_blocks = [block for block in blocks if block[0]["sequence_id"] == sequence]
        baseline_first = sum(block[0]["mode"] == "baseline" for block in sequence_blocks)
        assert baseline_first in {2, 3}

    output = tmp_path / "run_matrix.csv"
    first_hash = freeze_run_matrix(rows, output)
    assert first_hash == hashlib.sha256(output.read_bytes()).hexdigest()
    assert freeze_run_matrix(rows, output) == first_hash
    changed = copy.deepcopy(rows)
    changed[0], changed[1] = changed[1], changed[0]
    with pytest.raises(ValueError, match="frozen run order"):
        freeze_run_matrix(changed, output)


def test_manifest_rejects_protocol_mutation(tmp_path: Path) -> None:
    value = json.loads((ROOT / "config/EXPERIMENT_MANIFEST.yaml").read_text())
    value["metrics"]["trajectory_alignment"] = "Sim3"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics"):
        load_experiment_manifest(path)


def test_shared_runner_accepts_only_bound_p08_formal_identity(tmp_path: Path) -> None:
    condition = RunCondition("fr1_desk", 23011, tmp_path, tmp_path / "TUM1.yaml")
    identity = {
        "study_id": "ovorb2_tum_v1",
        "block_id": "fr1_desk-seed-23011",
        "mode": "baseline",
        "protocol_manifest_sha256": "a" * 64,
        "run_order_sha256": "b" * 64,
        "producer_commit": "c" * 40,
    }
    assert _validated_formal_identity(
        identity,
        condition=condition,
        expected_implementation="candidate",
        producer_commit="c" * 40,
        expected_mode="baseline",
    ) == identity
    with pytest.raises(ValueError, match="mode"):
        _validated_formal_identity(
            dict(identity, mode="semantic-feedback"),
            condition=condition,
            expected_implementation="candidate",
            producer_commit="c" * 40,
            expected_mode="baseline",
        )
