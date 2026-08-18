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


def _packet(frame_id: int, timestamp: float, mask: np.ndarray) -> SemanticFramePacket:
    return SemanticFramePacket(
        schema="ovorb.semantic-cache.v1", study_id="study", sequence_id="tiny", frame_id=frame_id,
        timestamp=timestamp, source_image_sha256=hashlib.sha256(f"rgb-{frame_id}".encode()).hexdigest(),
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


def _semantic_manifest(source_tree_sha256: str, association_sha256: str) -> CacheManifest:
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
        4,
        None,
    )


def _prepare(
    root: Path,
    *,
    pose_timestamps: tuple[int, ...] = (0, 1, 2, 3),
    final_mask: np.ndarray | None = None,
    depth_step: int = 1000,
    run_trajectory_sha256: str | None = None,
) -> tuple[Path, object]:
    dataset = root / "dataset"
    (dataset / "depth").mkdir(parents=True)
    (dataset / "rgb").mkdir()
    association = dataset / "associate.txt"
    rows = []
    for frame_id in range(4):
        image = np.full((10, 10), 1000 + frame_id * depth_step, dtype=np.uint16)
        assert cv2.imwrite(str(dataset / f"depth/{frame_id}.png"), image)
        (dataset / f"rgb/{frame_id}.png").write_bytes(f"rgb-{frame_id}".encode())
        rows.append(f"{float(frame_id):.9f} rgb/{frame_id}.png {float(frame_id):.9f} depth/{frame_id}.png")
    association.write_text("\n".join(rows) + "\n", encoding="utf-8")
    semantic_root = root / "semantic"
    writer = CacheWriter(
        semantic_root,
        _semantic_manifest(
            hash_dataset_tree(dataset),
            hashlib.sha256(association.read_bytes()).hexdigest(),
        ),
    )
    mask = np.ones((10, 10), dtype=bool)
    for frame_id in range(4):
        packet_mask = final_mask if frame_id == 3 and final_mask is not None else mask
        writer.add(_packet(frame_id, float(frame_id), packet_mask))
    writer.finalize()
    trajectory = root / "CameraTrajectory.txt"
    trajectory.write_text("\n".join(f"{float(i):.9f} 0 0 0 0 0 0 1" for i in pose_timestamps) + "\n", encoding="utf-8")
    run_manifest = root / "run_manifest.json"
    trajectory_sha256 = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    run_manifest.write_text(
        json.dumps(
            {
                "state": "COMPLETED",
                "valid": True,
                "sequence_id": "tiny",
                "seed": 23011,
                "trajectory": {"sha256": run_trajectory_sha256 or trajectory_sha256},
            }
        ),
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
    assert validate_dynamic_cache(first_job.cache_root, first_job.manifest).valid


def test_missing_exact_pose_produces_unknown_coverage(tmp_path: Path) -> None:
    # Catches interpolation/future filling of a bootstrap pose.
    _, job = _prepare(tmp_path, pose_timestamps=(0, 2, 3))
    result = generate_dynamic_cache(job)

    assert result.track_rows[1]["reason"] == "MISSING_EXACT_BOOTSTRAP_POSE"
    assert result.track_rows[1]["strong_dynamic"] is False
    assert len(result.frame_index) == 4


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


def test_resume_reuses_valid_atomic_outputs_and_moving_track_confirms(tmp_path: Path) -> None:
    # Catches duplicate resume rows, stale partials, and score maps ignoring strong state.
    _, job = _prepare(tmp_path)
    first = generate_dynamic_cache(job)
    second = generate_dynamic_cache(job)

    assert second.frame_index == first.frame_index
    assert len(second.frame_index) == 4
    assert not list(job.cache_root.rglob("*.partial"))
    final_score_map = np.load(job.cache_root / second.frame_index[-1]["path"], allow_pickle=False)
    assert np.all(final_score_map >= 0.70)
    assert validate_dynamic_cache(job.cache_root, job.manifest).valid


def test_confirmed_static_track_returns_to_low_score(tmp_path: Path) -> None:
    # Catches treating every non-dynamic track as perpetually uncertain.
    _, job = _prepare(tmp_path, depth_step=0)
    result = generate_dynamic_cache(job)

    final_score_map = np.load(job.cache_root / result.frame_index[-1]["path"], allow_pickle=False)
    assert np.all(final_score_map == 0.0)
