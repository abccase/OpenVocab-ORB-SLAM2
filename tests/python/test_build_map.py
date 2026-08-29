from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from semantic_py.openvocab_slam.map import CameraIntrinsics, TWorldCamera


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_registered_run(
    root: Path,
    registry: Path,
    *,
    run_id: str,
    sequence_id: str = "fr1_desk",
    seed: int = 23011,
    mode: str = "semantic-feedback",
) -> Path:
    from tools.build_map import append_jsonl

    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    trajectory = run_dir / "CameraTrajectory.txt"
    trajectory.write_text("1.000000 0 0 0 0 0 0 1\n", encoding="utf-8")
    payloads = {
        "keyframe_trajectory": run_dir / "KeyFrameTrajectory.txt",
        "telemetry": run_dir / "frame_telemetry.csv",
        "timings": run_dir / "timings.json",
        "final_state": run_dir / "final_state.json",
        "stdout": run_dir / "stdout.log",
        "stderr": run_dir / "stderr.log",
    }
    payloads["keyframe_trajectory"].write_text(
        "1.000000 0 0 0 0 0 0 1\n", encoding="utf-8"
    )
    payloads["telemetry"].write_text("frame_id\n0\n", encoding="utf-8")
    payloads["timings"].write_text("{}\n", encoding="utf-8")
    payloads["final_state"].write_text("{}\n", encoding="utf-8")
    payloads["stdout"].write_text("", encoding="utf-8")
    payloads["stderr"].write_text("", encoding="utf-8")

    def artifact(path: Path, *, pose_count: int | None = None):
        value = {
            "path": path.name,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if pose_count is not None:
            value["pose_count"] = pose_count
        return value

    cache_identity = {
        "manifest_sha256": "a" * 64,
        "completion_sha256": "b" * 64,
        "index_sha256": "c" * 64,
    }
    verified_inputs = {
        "dataset_manifest_sha256": "d" * 64,
        "source_tree_sha256": "e" * 64,
        "dynamic_manifest_sha256": "a" * 64,
        "dynamic_completion_sha256": "b" * 64,
        "dynamic_index_sha256": "c" * 64,
        "dynamic_config_sha256": "f" * 64,
        "semantic_manifest_sha256": "1" * 64,
        "semantic_identity_sha256": "2" * 64,
        "inference_config_sha256": "3" * 64,
        "prompt_sha256": "4" * 64,
        "prompt_config_sha256": "5" * 64,
        "protocol": {
            "dynamic": "ovorb.dynamic-cache.v1",
            "semantic": "ovorb.semantic-cache.v1",
        },
    }
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "study": "smoke",
        "mode": mode,
        "sequence_id": sequence_id,
        "seed": seed,
        "cwd": str(run_dir.resolve()),
        "expected_frames": 1,
        "frame_count": 1,
        "exit_code": 0,
        "invalid_reason": None,
        "state": "COMPLETED",
        "valid": True,
        "cache_identity": cache_identity,
        "verified_inputs": verified_inputs,
        "registration_identity": {
            "study": "smoke",
            "mode": mode,
            "sequence_id": sequence_id,
            "seed": seed,
            "expected_frames": 1,
            "cache_identity": cache_identity,
            "verified_inputs": verified_inputs,
        },
        "trajectory": artifact(trajectory, pose_count=1),
        **{
            key: artifact(path, pose_count=1 if key == "keyframe_trajectory" else None)
            for key, path in payloads.items()
        },
    }
    manifest["telemetry"]["format"] = "csv"
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    append_jsonl(registry, manifest)
    return manifest_path


def test_strict_selector_accepts_one_valid_registered_offline_run(tmp_path: Path) -> None:
    from tools.build_map import select_registered_run

    registry = tmp_path / "registry.jsonl"
    expected = write_registered_run(tmp_path, registry, run_id="run-a")

    selected = select_registered_run(registry, "fr1_desk", 23011)

    assert selected.manifest_path == expected.resolve()
    assert selected.trajectory_path == (expected.parent / "CameraTrajectory.txt").resolve()
    assert selected.manifest["run_id"] == "run-a"


@pytest.mark.parametrize("count", [0, 2])
def test_strict_selector_fails_unless_exactly_one_run_matches(
    tmp_path: Path, count: int
) -> None:
    from tools.build_map import select_registered_run

    registry = tmp_path / "registry.jsonl"
    registry.write_text("", encoding="utf-8")
    for index in range(count):
        write_registered_run(tmp_path, registry, run_id=f"run-{index}")

    with pytest.raises(ValueError, match="exactly one"):
        select_registered_run(registry, "fr1_desk", 23011)


def test_strict_selector_uses_latest_state_and_rejects_tampering(tmp_path: Path) -> None:
    from tools.build_map import append_jsonl, select_registered_run

    registry = tmp_path / "registry.jsonl"
    manifest_path = write_registered_run(tmp_path, registry, run_id="run-a")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    append_jsonl(registry, {**manifest, "state": "FAILED", "valid": False})

    with pytest.raises(ValueError, match="exactly one"):
        select_registered_run(registry, "fr1_desk", 23011)

    registry.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (manifest_path.parent / "CameraTrajectory.txt").write_text(
        "1.000000 1 0 0 0 0 0 1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="trajectory hash"):
        select_registered_run(registry, "fr1_desk", 23011)


def test_strict_selector_rejects_incomplete_completed_manifest(tmp_path: Path) -> None:
    from tools.build_map import select_registered_run

    registry = tmp_path / "registry.jsonl"
    manifest_path = write_registered_run(tmp_path, registry, run_id="run-a")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["exit_code"] = 9
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="completion identity"):
        select_registered_run(registry, "fr1_desk", 23011)


def test_trajectory_matching_uses_declared_six_decimal_identity(tmp_path: Path) -> None:
    from tools.build_map import load_tum_trajectory, pose_for_association_timestamp

    trajectory = tmp_path / "CameraTrajectory.txt"
    trajectory.write_text(
        "1305031453.359684 1 2 3 0 0 0 1\n", encoding="utf-8"
    )
    poses = load_tum_trajectory(trajectory)

    pose = pose_for_association_timestamp(poses, "1305031453.359683990")

    assert pose is not None
    assert pose.matrix[:3, 3].tolist() == [1.0, 2.0, 3.0]
    assert pose_for_association_timestamp(poses, "1305031453.359685100") is None


def test_map_configuration_rejects_nonformal_seed(tmp_path: Path) -> None:
    from tools.build_map import MapBuildConfig

    source = Path("config/P07_MAP.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["seed"] = 7
    path = tmp_path / "map.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen identity"):
        MapBuildConfig.from_path(path)


def test_map_configuration_rejects_mutated_formal_parameter(tmp_path: Path) -> None:
    from tools.build_map import MapBuildConfig

    value = json.loads(Path("config/P07_MAP.json").read_text(encoding="utf-8"))
    value["objects"]["dbscan_eps_m"] = 0.2
    path = tmp_path / "map.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen hash"):
        MapBuildConfig.from_path(path)


def test_object_backprojection_excludes_depth_beyond_map_truncation() -> None:
    from tools.build_map import _backproject_mask_points

    mask = np.ones((1, 2), dtype=bool)
    depth = np.array([[1.0, 6.0]], dtype=np.float32)
    points = _backproject_mask_points(
        mask,
        depth,
        CameraIntrinsics(2, 1, 1.0, 1.0, 0.0, 0.0),
        TWorldCamera.identity(),
        limit=10,
        depth_trunc_m=5.0,
    )

    np.testing.assert_allclose(points, [[0.0, 0.0, 1.0]])


def test_production_numeric_track_ids_are_classified_and_canonicalized() -> None:
    from tools.build_map import classify_track_rows, canonical_track_id

    rows = [
        {"frame_id": 0, "local_id": 0, "track_id": 7, "strong_dynamic": False},
        {"frame_id": 1, "local_id": 0, "track_id": 7, "strong_dynamic": True},
        {"frame_id": 1, "local_id": 1, "track_id": 12, "strong_dynamic": False},
        {"frame_id": 2, "local_id": 0, "track_id": None, "strong_dynamic": False},
    ]

    by_instance, dynamic_ids = classify_track_rows(rows)

    assert by_instance[(1, 1)]["track_id"] == 12
    assert dynamic_ids == {7}
    assert canonical_track_id(12) == "track-000012"


def test_selected_run_must_bind_the_exact_cache_files() -> None:
    from tools.build_map import validate_selected_cache_binding

    cache_identity = {
        "manifest_sha256": "a" * 64,
        "completion_sha256": "b" * 64,
        "index_sha256": "c" * 64,
    }
    verified = {
        "dynamic_manifest_sha256": "a" * 64,
        "dynamic_completion_sha256": "b" * 64,
        "dynamic_index_sha256": "c" * 64,
    }
    run = {"cache_identity": cache_identity, "verified_inputs": verified}

    validate_selected_cache_binding(run, cache_identity, verified)

    changed = {**cache_identity, "index_sha256": "d" * 64}
    with pytest.raises(ValueError, match="cache identity"):
        validate_selected_cache_binding(run, changed, verified)


def test_cache_completion_rejects_wrong_schema_even_when_hashes_match(
    tmp_path: Path,
) -> None:
    from tools.build_map import _validate_cache_completion

    index = tmp_path / "cache_index.jsonl"
    index.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "cache_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "wrong.schema",
        "study_id": "ovorb2_tum_v1",
        "sequence_id": "fr1_desk",
        "association_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "expected_frame_count": 1,
    }), encoding="utf-8")
    (tmp_path / "cache_complete.json").write_text(json.dumps({
        "manifest_sha256": sha256(manifest_path),
        "index_sha256": sha256(index),
        "frame_count": 1,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic cache manifest identity"):
        _validate_cache_completion(
            tmp_path,
            sequence_id="fr1_desk",
            association_sha256="a" * 64,
            source_tree_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            expected_frame_count=1,
            study_id="ovorb2_tum_v1",
            dynamic=False,
        )
