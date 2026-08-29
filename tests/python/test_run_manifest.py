from __future__ import annotations

import copy
import hashlib
import json
import textwrap
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
from tools.run_orb_tum import RunCondition, _validated_formal_identity, run_ov_condition
from tools.run_study import (
    expected_registration_identity,
    terminalize_failed_validation,
    validate_attempt_registration_identity,
)


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


def _study_registration(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
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
        "compatibility_commit": "b" * 40,
        "experiment_manifest_sha256": "c" * 64,
        "run_order_sha256": "d" * 64,
        "executable": str(tmp_path / "rgbd_tum_ov"),
        "executable_sha256": "e" * 64,
        "vocabulary": str(tmp_path / "ORBvoc.txt"),
        "vocabulary_sha256": "f" * 64,
        "datasets": {
            "fr1_desk": {
                "sequence_root": str(tmp_path / "sequence"),
                "dataset_manifest_sha256": "0" * 64,
                "source_tree_sha256": "1" * 64,
                "association": str(tmp_path / "sequence" / "associate.txt"),
                "association_sha256": "2" * 64,
                "expected_frames": 3,
                "settings": str(tmp_path / "TUM1.yaml"),
                "settings_sha256": "3" * 64,
                "cache_root": str(tmp_path / "cache"),
                "cache_identity": {
                    "manifest_sha256": "4" * 64,
                    "completion_sha256": "5" * 64,
                    "index_sha256": "6" * 64,
                },
                "semantic_manifest_sha256": "7" * 64,
                "semantic_identity_sha256": "8" * 64,
                "inference_config_sha256": "9" * 64,
                "prompt_sha256": "a" * 64,
                "prompt_config_sha256": "b" * 64,
                "configuration_sha256": "c" * 64,
                "dynamic_schema": "ovorb.dynamic-cache.v1",
                "semantic_schema": "ovorb.semantic-cache.v1",
            }
        },
    }
    return condition, registration


@pytest.mark.parametrize(
    "mutation",
    ["settings", "semantic_manifest", "prompt_config", "coordinated_command"],
)
def test_attempt_identity_is_reconstructed_not_trusted(
    tmp_path: Path, mutation: str
) -> None:
    condition, registration = _study_registration(tmp_path)
    expected = expected_registration_identity(condition, registration)
    manifest = {
        "registration_identity": copy.deepcopy(expected),
        "settings": copy.deepcopy(expected["settings"]),
        "verified_inputs": copy.deepcopy(expected["verified_inputs"]),
        "command": list(expected["command"]),
    }
    if mutation == "settings":
        manifest["settings"]["sha256"] = "d" * 64
        manifest["registration_identity"]["settings"]["sha256"] = "d" * 64
    elif mutation == "semantic_manifest":
        manifest["verified_inputs"]["semantic_manifest_sha256"] = "d" * 64
        manifest["registration_identity"]["verified_inputs"]["semantic_manifest_sha256"] = "d" * 64
    elif mutation == "prompt_config":
        manifest["verified_inputs"]["prompt_config_sha256"] = "d" * 64
        manifest["registration_identity"]["verified_inputs"]["prompt_config_sha256"] = "d" * 64
    else:
        manifest["command"][-1] = "d" * 64
        manifest["registration_identity"]["command"][-1] = "d" * 64
    with pytest.raises(ValueError, match="registered identity"):
        validate_attempt_registration_identity(manifest, condition, registration)


def test_failed_validation_terminalizes_without_overwriting_evidence(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt-001"
    attempt.mkdir()
    manifest_path = attempt / "run_manifest.json"
    manifest_path.write_text(json.dumps({
        "state": "COMPLETED", "valid": True, "invalid_reason": None,
    }), encoding="utf-8")
    metric_path = attempt / "metric_output.json"
    metric_path.write_bytes(b"tampered-metric-evidence\n")
    before = hashlib.sha256(metric_path.read_bytes()).hexdigest()
    original_manifest = manifest_path.read_bytes()

    terminalize_failed_validation(attempt, "metric output hash mismatch")

    terminal = json.loads(manifest_path.read_text())
    assert terminal["state"] == "FAILED"
    assert terminal["valid"] is False
    assert "metric output hash mismatch" in terminal["invalid_reason"]
    assert hashlib.sha256(metric_path.read_bytes()).hexdigest() == before
    assert (attempt / "run_manifest.pre_p08_validation.json").read_bytes() == original_manifest
    failure = json.loads((attempt / "p08_validation_failure.json").read_text())
    assert failure["state"] == "FAILED"


@pytest.mark.parametrize("failure_kind", ["interrupted", "metric_tamper"])
def test_failed_p08_attempt_is_preserved_and_resume_allocates_new_id(
    tmp_path: Path, failure_kind: str
) -> None:
    sequence = tmp_path / "sequence"
    sequence.mkdir()
    (sequence / "associate.txt").write_text(
        "1.0 rgb/1.png 1.0 depth/1.png\n", encoding="utf-8"
    )
    settings = tmp_path / "TUM1.yaml"
    settings.write_text("settings\n", encoding="utf-8")
    vocabulary = tmp_path / "ORBvoc.txt"
    vocabulary.write_text("vocabulary\n", encoding="utf-8")
    executable = tmp_path / "fake_ov.py"
    executable.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import json, sys
        from pathlib import Path
        mode = sys.argv[5]
        Path('CameraTrajectory.txt').write_text('1.0 0 0 0 0 0 0 1\\n')
        Path('KeyFrameTrajectory.txt').write_text('1.0 0 0 0 0 0 0 1\\n')
        header = ('frame_index,timestamp,tracking_state,pose_valid,tracking_time_seconds,'
                  'raw_keypoints,used_keypoints,removed_dynamic,retained_uncertain,'
                  'removed_uncertain,semantic_accessed,semantic_state,cache_load_seconds,'
                  'policy_seconds,pacing_lateness_seconds,ipc_call_seconds,ipc_reason,'
                  'request_attempted,request_sent,packet_age_ms,inference_ms,'
                  'strong_track_count,unconfirmed_track_count')
        Path('frame_telemetry.csv').write_text(
            header + '\\n0,1.0,2,1,0.01,100,100,0,0,0,0,BASELINE,0,0,0,'
            '0,NOT_APPLICABLE,0,0,-1,-1,0,0\\n')
        Path('timings.json').write_text(json.dumps({'frame_count': 1,
            'mean_tracking_seconds': 0.01, 'median_tracking_seconds': 0.01,
            'mean_pacing_lateness_seconds': 0, 'max_pacing_lateness_seconds': 0,
            'wall_seconds': 0.01}))
        Path('final_state.json').write_text(json.dumps(
            {'state': 'COMPLETED', 'mode': mode, 'frame_count': 1}))
    """), encoding="utf-8")
    executable.chmod(0o755)
    condition = RunCondition("tiny", 23011, sequence, settings)
    producer = "a" * 40
    identity = {
        "study_id": "ovorb2_tum_v1",
        "block_id": "tiny-seed-23011",
        "mode": "baseline",
        "protocol_manifest_sha256": "b" * 64,
        "run_order_sha256": "c" * 64,
        "producer_commit": producer,
    }
    arguments = dict(
        mode="baseline", executable=executable, vocabulary=vocabulary,
        output_root=tmp_path / "runs", compatibility_commit="compatibility",
        producer_commit=producer, study="ovorb2_tum_v1", formal_identity=identity,
    )
    first = run_ov_condition(condition, **arguments)
    evidence = first.run_dir / "metric_output.json"
    evidence.write_bytes(b"original-invalid-evidence\n")
    evidence_hash = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if failure_kind == "interrupted":
        manifest_path = first.run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update({"state": "RUNNING", "valid": False})
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    terminalize_failed_validation(first.run_dir, failure_kind)

    replacement = run_ov_condition(condition, **arguments)

    assert replacement.run_dir.name == "attempt-002"
    assert replacement.run_dir != first.run_dir
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == evidence_hash
