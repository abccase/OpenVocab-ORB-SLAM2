import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from semantic_py.openvocab_slam.cache import CacheWriter
from semantic_py.openvocab_slam.dynamic_cache import hash_dataset_tree
from semantic_py.openvocab_slam.schemas import (
    CacheManifest,
    InstanceObservation,
    SemanticFramePacket,
    encode_binary_mask_rle,
)
from tools.generate_dynamic_cache import build_dynamic_job, generate_dynamic_cache, validate_dynamic_cache


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _refresh_complete_hashes(cache_root: Path) -> None:
    complete_path = cache_root / "cache_complete.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    for key, relative in (
        ("index_sha256", "cache_index.jsonl"),
        ("tracks_sha256", "dynamic_tracks.jsonl"),
        ("diagnostics_index_sha256", "diagnostics_index.jsonl"),
    ):
        path = cache_root / relative
        if key in complete and path.exists():
            complete[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    complete_path.write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _packet(
    frame_id: int,
    timestamp: float,
    mask: np.ndarray,
    source_image_sha256: str,
) -> SemanticFramePacket:
    return SemanticFramePacket(
        schema="ovorb.semantic-cache.v1", study_id="study", sequence_id="tiny", frame_id=frame_id,
        timestamp=timestamp, source_image_sha256=source_image_sha256,
        image_width=mask.shape[1], image_height=mask.shape[0], prompt_sha256="2" * 64,
        model_manifest_sha256="3" * 64, inference_config_sha256="4" * 64, inference_time_seconds=0.0,
        instances=(
            InstanceObservation(
                0,
                "person",
                0.9,
                (0.0, 0.0, float(mask.shape[1]), float(mask.shape[0])),
                encode_binary_mask_rle(mask),
            ),
        ),
    )


def _semantic_manifest(
    source_tree_sha256: str,
    association_sha256: str,
    frame_count: int,
) -> CacheManifest:
    return CacheManifest(
        "ovorb.semantic-cache.v1",
        "study",
        "tiny",
        source_tree_sha256,
        association_sha256,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "7" * 40,
        10,
        frame_count,
        None,
    )


def _prepare(
    root: Path,
    *,
    pose_timestamps: tuple[int, ...] | None = None,
    final_mask: np.ndarray | None = None,
    depth_step: int = 1000,
    run_trajectory_sha256: str | None = None,
    depth_timestamps: tuple[float, ...] | None = None,
    corrupt_depth_frame: int | None = None,
    run_manifest_updates: dict[str, object] | None = None,
    depth_values: tuple[int, ...] | None = None,
) -> tuple[Path, object]:
    dataset = root / "dataset"
    (dataset / "depth").mkdir(parents=True)
    (dataset / "rgb").mkdir()
    association = dataset / "associate.txt"
    rows = []
    rgb_sha256: list[str] = []
    frame_count = len(depth_values) if depth_values is not None else 4
    selected_depth_timestamps = depth_timestamps or tuple(
        float(frame_id) for frame_id in range(frame_count)
    )
    if len(selected_depth_timestamps) != frame_count:
        raise ValueError("depth timestamp fixture coverage mismatch")
    for frame_id in range(frame_count):
        depth_value = (
            depth_values[frame_id]
            if depth_values is not None
            else 1000 + frame_id * depth_step
        )
        image = np.full((10, 10), depth_value, dtype=np.uint16)
        depth_path = dataset / f"depth/{frame_id}.png"
        if frame_id == corrupt_depth_frame:
            depth_path.write_bytes(b"not-an-image")
        else:
            assert cv2.imwrite(str(depth_path), image)
        rgb_path = dataset / f"rgb/{frame_id}.png"
        rgb = np.full((10, 10, 3), (frame_id * 40, 30, 200), dtype=np.uint8)
        assert cv2.imwrite(str(rgb_path), rgb)
        rgb_sha256.append(hashlib.sha256(rgb_path.read_bytes()).hexdigest())
        rows.append(
            f"{float(frame_id):.9f} rgb/{frame_id}.png "
            f"{selected_depth_timestamps[frame_id]:.9f} depth/{frame_id}.png"
        )
    association.write_text("\n".join(rows) + "\n", encoding="utf-8")
    semantic_root = root / "semantic"
    writer = CacheWriter(
        semantic_root,
        _semantic_manifest(
            hash_dataset_tree(dataset),
            hashlib.sha256(association.read_bytes()).hexdigest(),
            frame_count,
        ),
    )
    mask = np.ones((10, 10), dtype=bool)
    for frame_id in range(frame_count):
        packet_mask = final_mask if frame_id == 3 and final_mask is not None else mask
        writer.add(_packet(frame_id, float(frame_id), packet_mask, rgb_sha256[frame_id]))
    writer.finalize()
    trajectory = root / "CameraTrajectory.txt"
    selected_pose_timestamps = pose_timestamps or tuple(range(frame_count))
    trajectory.write_text(
        "\n".join(
            f"{float(i):.9f} 0 0 0 0 0 0 1"
            for i in selected_pose_timestamps
        )
        + "\n",
        encoding="utf-8",
    )
    run_manifest = root / "run_manifest.json"
    trajectory_sha256 = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    association_sha256 = hashlib.sha256(association.read_bytes()).hexdigest()
    dataset_manifest = root / "dataset_manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sequence_id": "tiny",
                "association_sha256": association_sha256,
                "extracted_tree_sha256": hash_dataset_tree(dataset),
                "counts": {"associations": frame_count},
                "image_dimensions": {"width": 10, "height": 10},
                "validation_status": "VALID",
            }
        ),
        encoding="utf-8",
    )
    dataset_manifest_sha256 = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
    run_value: dict[str, object] = {
        "schema_version": 1,
        "run_id": "oracle-tiny-seed-23011-attempt-001",
        "study": "oracle",
        "mode": "baseline",
        "compatibility_commit": "bd85add6b40e6fa719883e9eb87b38a3f15e7c6d",
        "producer_commit": "58014b7c1f2b73427b67b4e80a8cf334127f48ea",
        "state": "COMPLETED",
        "valid": True,
        "exit_code": 0,
        "invalid_reason": None,
        "sequence_id": "tiny",
        "seed": 23011,
        "association_sha256": association_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "expected_frames": frame_count,
        "frame_count": frame_count,
        "trajectory": {
            "path": "CameraTrajectory.txt",
            "pose_count": len(selected_pose_timestamps),
            "sha256": run_trajectory_sha256 or trajectory_sha256,
        },
    }
    run_value.update(run_manifest_updates or {})
    run_manifest.write_text(
        json.dumps(run_value),
        encoding="utf-8",
    )
    intrinsics = np.array([[100.0, 0.0, 4.5], [0.0, 100.0, 4.5], [0.0, 0.0, 1.0]])
    job = build_dynamic_job(
        root,
        "tiny",
        dataset,
        semantic_root,
        trajectory,
        run_manifest,
        intrinsics,
        producer_commit="8" * 40,
        dataset_manifest_path=dataset_manifest,
    )
    return root, job


def test_future_observation_cannot_change_prior_score_map_hashes(tmp_path: Path) -> None:
    # Catches cache generation that reads future semantic packets before writing frame t.
    _, first_job = _prepare(tmp_path / "first")
    changed_final_mask = np.zeros((10, 10), dtype=bool)
    changed_final_mask[:, :5] = True
    _, second_job = _prepare(tmp_path / "second", final_mask=changed_final_mask)
    first = generate_dynamic_cache(first_job)
    second = generate_dynamic_cache(second_job)

    assert [row["sha256"] for row in second.frame_index[:3]] == [row["sha256"] for row in first.frame_index[:3]]
    assert second.frame_index[3]["sha256"] != first.frame_index[3]["sha256"]
    assert validate_dynamic_cache(first_job).valid


def test_missing_exact_pose_produces_unknown_coverage(tmp_path: Path) -> None:
    # Catches interpolation/future filling of a bootstrap pose.
    _, job = _prepare(tmp_path, pose_timestamps=(0, 2, 3))
    result = generate_dynamic_cache(job)

    assert result.track_rows[1]["reason_codes"] == ["MISSING_EXACT_BOOTSTRAP_POSE"]
    assert result.track_rows[1]["strong_dynamic"] is False
    assert len(result.frame_index) == 4


def test_future_depth_timestamp_is_unknown_and_depth_is_never_read(tmp_path: Path) -> None:
    # Catches using a depth observation that occurs after the formal RGB frame time.
    _, job = _prepare(
        tmp_path,
        depth_timestamps=(0.0, 1.5, 2.0, 3.0),
        corrupt_depth_frame=1,
    )
    result = generate_dynamic_cache(job)

    row = next(item for item in result.track_rows if item["frame_id"] == 1)
    assert row["reason_codes"] == ["FUTURE_DEPTH_TIMESTAMP"]
    assert row["strong_dynamic"] is False
    assert row["track_id"] is None


def test_unknown_row_records_future_depth_and_missing_pose_in_order(
    tmp_path: Path,
) -> None:
    # Catches collapsing simultaneous causal failures into one lossy reason string.
    _, job = _prepare(
        tmp_path,
        pose_timestamps=(0, 2, 3),
        depth_timestamps=(0.0, 1.5, 2.0, 3.0),
        corrupt_depth_frame=1,
    )

    result = generate_dynamic_cache(job)

    row = next(item for item in result.track_rows if item["frame_id"] == 1)
    assert row["reason_codes"] == [
        "FUTURE_DEPTH_TIMESTAMP",
        "MISSING_EXACT_BOOTSTRAP_POSE",
    ]
    assert row["strong_dynamic"] is False


def test_manifest_binds_inputs_and_generation_fails_closed_on_mutation(tmp_path: Path) -> None:
    # Catches generation continuing after its bound bootstrap trajectory changes.
    _, job = _prepare(tmp_path)
    assert job.manifest.schema == "ovorb.dynamic-cache.v1"
    assert job.manifest.producer_commit == "8" * 40
    assert len(job.manifest.semantic_manifest_sha256) == 64
    assert len(job.manifest.source_tree_sha256) == 64
    assert len(job.manifest.bootstrap_trajectory_sha256) == 64
    assert len(job.manifest.dynamic_config_sha256) == 64
    assert job.manifest.expected_instance_count == 4
    job.trajectory_path.write_text("0.000000000 9 0 0 0 0 0 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bootstrap trajectory hash mismatch"):
        generate_dynamic_cache(job)
    assert not (job.cache_root / "cache_complete.json").exists()


def test_job_rejects_bootstrap_manifest_trajectory_disagreement(tmp_path: Path) -> None:
    # Catches binding a valid file and a valid manifest that disagree with each other.
    with pytest.raises(ValueError, match="bootstrap run trajectory hash mismatch"):
        _prepare(tmp_path, run_trajectory_sha256="0" * 64)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"study": "formal"}, "study"),
        ({"mode": "semantic-feedback"}, "mode"),
        ({"run_id": "wrong"}, "run identity"),
        ({"association_sha256": "0" * 64}, "association"),
        ({"dataset_manifest_sha256": "0" * 64}, "dataset manifest"),
        ({"expected_frames": 5}, "frame count"),
        ({"frame_count": 3}, "frame count"),
    ],
)
def test_job_rejects_noncanonical_bootstrap_identity(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare(tmp_path, run_manifest_updates=updates)


def test_job_rejects_wrong_baseline_compatibility_commit(tmp_path: Path) -> None:
    # Catches accepting a commit-shaped value instead of the frozen baseline tag target.
    with pytest.raises(ValueError, match="compatibility commit"):
        _prepare(tmp_path, run_manifest_updates={"compatibility_commit": "0" * 40})


def test_job_rejects_wrong_formal_baseline_producer_commit(tmp_path: Path) -> None:
    # Catches accepting a commit-shaped value unrelated to canonical oracle artifacts.
    with pytest.raises(ValueError, match="producer commit"):
        _prepare(tmp_path, run_manifest_updates={"producer_commit": "1" * 40})


def test_resume_reuses_valid_atomic_outputs_and_moving_track_confirms(tmp_path: Path) -> None:
    # Catches duplicate resume rows, stale partials, and score maps ignoring strong state.
    _, job = _prepare(tmp_path, depth_values=(1000, 2000, 4000, 7000))
    first = generate_dynamic_cache(job)
    second = generate_dynamic_cache(job)

    assert second.frame_index == first.frame_index
    assert len(second.frame_index) == 4
    assert not list(job.cache_root.rglob("*.partial"))
    final_score_map = np.load(job.cache_root / second.frame_index[-1]["path"], allow_pickle=False)
    assert np.all(final_score_map >= 0.70)
    assert validate_dynamic_cache(job).valid


def test_interrupted_prefix_orphan_and_partial_resume(tmp_path: Path) -> None:
    # Catches resume logic that handles only a completed-cache fast return.
    _, job = _prepare(tmp_path)
    first = generate_dynamic_cache(job)
    original_hashes = [row["sha256"] for row in first.frame_index]
    (job.cache_root / "cache_complete.json").unlink()
    (job.cache_root / "dynamic_tracks.jsonl").unlink()
    (job.cache_root / "diagnostics_index.jsonl").unlink()
    for path in (job.cache_root / "diagnostics").glob("*.png"):
        path.unlink()
    index = _jsonl(job.cache_root / "cache_index.jsonl")
    _write_jsonl(job.cache_root / "cache_index.jsonl", index[:2])
    (job.cache_root / "score_maps/000003.npy").unlink()
    (job.cache_root / "score_maps/.000003.npy.partial").write_bytes(b"interrupted")

    resumed = generate_dynamic_cache(job)

    assert [row["sha256"] for row in resumed.frame_index] == original_hashes
    assert not list(job.cache_root.rglob("*.partial"))
    assert validate_dynamic_cache(job).valid


def test_validator_rejects_wrong_frame_and_semantic_identity(tmp_path: Path) -> None:
    # Catches a self-consistent index that no longer matches the semantic packet.
    _, job = _prepare(tmp_path)
    generate_dynamic_cache(job)
    index_path = job.cache_root / "cache_index.jsonl"
    index = _jsonl(index_path)
    index[1]["timestamp"] = 1.25
    index[1]["semantic_packet_sha256"] = "0" * 64
    _write_jsonl(index_path, index)
    _refresh_complete_hashes(job.cache_root)

    validation = validate_dynamic_cache(job)

    assert validation.valid is False
    assert any("semantic identity" in error for error in validation.errors)


@pytest.mark.parametrize("mutation", ["duplicate", "missing"])
def test_validator_rejects_duplicate_or_missing_instance_rows(
    tmp_path: Path,
    mutation: str,
) -> None:
    # Catches track JSONL coverage checks based only on total row count.
    _, job = _prepare(tmp_path)
    generate_dynamic_cache(job)
    tracks_path = job.cache_root / "dynamic_tracks.jsonl"
    rows = _jsonl(tracks_path)
    rows = rows + [dict(rows[0])] if mutation == "duplicate" else rows[:-1]
    _write_jsonl(tracks_path, rows)
    _refresh_complete_hashes(job.cache_root)

    validation = validate_dynamic_cache(job)

    assert validation.valid is False
    assert any("instance identity" in error for error in validation.errors)


def test_real_strong_history_clamps_actionable_score_until_exit(tmp_path: Path) -> None:
    # Catches exposing raw hysteresis-band probability as an actionable removal score.
    depth_values = (1000, 2000, 4000, 7000) + (7000,) * 7
    _, job = _prepare(tmp_path, depth_values=depth_values)

    result = generate_dynamic_cache(job)

    by_frame = {int(row["frame_id"]): row for row in result.track_rows}
    assert by_frame[7]["dynamic_probability"] == 0.6
    assert by_frame[8]["dynamic_probability"] == 0.4
    assert by_frame[7]["strong_dynamic"] is True
    assert by_frame[8]["strong_dynamic"] is True
    assert by_frame[7]["score_map_probability"] == 0.70
    assert by_frame[8]["score_map_probability"] == 0.70
    assert by_frame[9]["dynamic_probability"] == 0.2
    assert by_frame[9]["strong_dynamic"] is False
    assert by_frame[9]["score_map_probability"] == 0.2
    assert validate_dynamic_cache(job).valid


@pytest.mark.parametrize(
    "mutation",
    [
        "track_id",
        "reason",
        "count",
        "geometry",
        "state_transition",
        "type",
        "range",
        "strong_raw_below_exit",
        "strong_score_below_enter",
        "nonstrong_score_mismatch",
    ],
)
def test_validator_rejects_track_history_and_row_semantic_tamper(
    tmp_path: Path,
    mutation: str,
) -> None:
    # Catches validating rows independently instead of against their causal track history.
    _, job = _prepare(tmp_path, depth_values=(1000, 2000, 4000, 7000))
    generate_dynamic_cache(job)
    tracks_path = job.cache_root / "dynamic_tracks.jsonl"
    rows = _jsonl(tracks_path)
    row = rows[0] if mutation == "nonstrong_score_mismatch" else rows[-1]
    if mutation == "track_id":
        row["track_id"] = 999
    elif mutation == "reason":
        row["reason_codes"] = ["NOT_AN_ALLOWED_REASON"]
    elif mutation == "count":
        row["observation_count"] = 999
    elif mutation == "geometry":
        row["centroid_world"] = [9.0, 9.0, 9.0]
    elif mutation == "state_transition":
        row["strong_dynamic"] = False
    elif mutation == "type":
        row["observation_count"] = True
    elif mutation == "range":
        row["dynamic_probability"] = 1.1
    elif mutation == "strong_raw_below_exit":
        row["dynamic_probability"] = 0.399
    elif mutation == "strong_score_below_enter":
        row["score_map_probability"] = 0.699
    else:
        assert row["strong_dynamic"] is False
        row["score_map_probability"] = 0.9
    _write_jsonl(tracks_path, rows)
    _refresh_complete_hashes(job.cache_root)

    validation = validate_dynamic_cache(job)

    assert validation.valid is False


def test_validator_rejects_track_row_and_score_map_disagreement(tmp_path: Path) -> None:
    # Catches failing to bind each semantic mask's pixels to its actionable row probability.
    _, job = _prepare(tmp_path, depth_values=(1000, 2000, 4000, 7000))
    generate_dynamic_cache(job)
    tracks_path = job.cache_root / "dynamic_tracks.jsonl"
    rows = _jsonl(tracks_path)
    rows[-1]["score_map_probability"] = 0.8
    _write_jsonl(tracks_path, rows)
    _refresh_complete_hashes(job.cache_root)

    validation = validate_dynamic_cache(job)

    assert validation.valid is False


def test_validator_rejects_self_consistent_score_map_replacement(tmp_path: Path) -> None:
    # Catches trusting a coordinated score-map/index/completion rewrite.
    _, job = _prepare(tmp_path)
    generate_dynamic_cache(job)
    index_path = job.cache_root / "cache_index.jsonl"
    index = _jsonl(index_path)
    score_path = job.cache_root / index[0]["path"]
    np.save(score_path, np.zeros((10, 10), dtype=np.float32), allow_pickle=False)
    index[0]["sha256"] = hashlib.sha256(score_path.read_bytes()).hexdigest()
    _write_jsonl(index_path, index)
    _refresh_complete_hashes(job.cache_root)

    validation = validate_dynamic_cache(job)

    assert validation.valid is False


def test_diagnostic_overlays_are_predeclared_bound_and_validated(tmp_path: Path) -> None:
    # Catches outcome-selected diagnostics or unmanifested overlay output.
    _, job = _prepare(tmp_path)
    assert [item["frame_id"] for item in job.manifest.diagnostic_frames] == [1, 2, 3]
    result = generate_dynamic_cache(job)

    assert [row["frame_id"] for row in result.diagnostic_index] == [1, 2, 3]
    assert all((job.cache_root / row["path"]).is_file() for row in result.diagnostic_index)
    assert validate_dynamic_cache(job).valid


@pytest.mark.parametrize("mutation", ["missing", "extra", "tampered"])
def test_validator_rejects_missing_extra_or_tampered_diagnostic_overlay(
    tmp_path: Path,
    mutation: str,
) -> None:
    # Catches checking only the diagnostic index rather than overlay set and bytes.
    _, job = _prepare(tmp_path)
    result = generate_dynamic_cache(job)
    first_overlay = job.cache_root / result.diagnostic_index[0]["path"]
    if mutation == "missing":
        first_overlay.unlink()
    elif mutation == "extra":
        (first_overlay.parent / "extra.png").write_bytes(first_overlay.read_bytes())
    else:
        first_overlay.write_bytes(b"tampered diagnostic overlay")

    validation = validate_dynamic_cache(job)

    assert validation.valid is False
    assert any("diagnostic" in error for error in validation.errors)


def test_validator_rejects_self_consistent_diagnostic_overlay_tamper(
    tmp_path: Path,
) -> None:
    # Catches trusting output-controlled overlay/index/completion hashes as an integrity root.
    _, job = _prepare(tmp_path)
    result = generate_dynamic_cache(job)
    diagnostics_path = job.cache_root / "diagnostics_index.jsonl"
    diagnostic_rows = _jsonl(diagnostics_path)
    target = job.cache_root / result.diagnostic_index[0]["path"]
    ok, encoded = cv2.imencode(
        ".png",
        np.zeros((10, 10, 3), dtype=np.uint8),
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    assert ok
    replacement = encoded.tobytes()
    target.write_bytes(replacement)
    diagnostic_rows[0]["sha256"] = hashlib.sha256(replacement).hexdigest()
    _write_jsonl(diagnostics_path, diagnostic_rows)
    _refresh_complete_hashes(job.cache_root)

    validation = validate_dynamic_cache(job)

    assert validation.valid is False
    assert any("diagnostic" in error for error in validation.errors)


def test_confirmed_static_track_returns_to_low_score(tmp_path: Path) -> None:
    # Catches treating every non-dynamic track as perpetually uncertain.
    _, job = _prepare(tmp_path, depth_step=0)
    result = generate_dynamic_cache(job)

    final_score_map = np.load(job.cache_root / result.frame_index[-1]["path"], allow_pickle=False)
    assert np.all(final_score_map == 0.0)
