from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

import semantic_py.openvocab_slam.map as map_module
from semantic_py.openvocab_slam.map import (
    CameraIntrinsics,
    StaticTsdfVolume,
    TCameraWorld,
    TsdfConfig,
    TWorldCamera,
)


def synthetic_frames() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    height, width = 48, 64
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for left in (8, 28, 48):
        color = np.full((height, width, 3), (40, 80, 120), dtype=np.uint8)
        depth_m = np.ones((height, width), dtype=np.float32)
        dynamic_scores = np.zeros((height, width), dtype=np.float32)
        depth_m[20:28, left:left + 8] = 0.70
        dynamic_scores[20:28, left:left + 8] = 0.90
        color[20:28, left:left + 8] = (220, 20, 20)
        frames.append((color, depth_m, dynamic_scores))
    return frames


def integrator() -> StaticTsdfVolume:
    return StaticTsdfVolume(
        CameraIntrinsics(width=64, height=48, fx=50.0, fy=50.0, cx=31.5, cy=23.5),
        TsdfConfig(
            voxel_length_m=0.02,
            sdf_trunc_m=0.08,
            depth_trunc_m=3.0,
            dynamic_threshold=0.70,
        ),
    )


def cuboid_points() -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, 11)
    ys = np.linspace(-0.4, 0.4, 7)
    zs = np.linspace(-0.2, 0.2, 5)
    local = np.array(np.meshgrid(xs, ys, zs, indexing="ij"), dtype=np.float64)
    local = local.reshape(3, -1).T
    angle = np.deg2rad(30.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return local @ rotation.T + np.array([1.0, 2.0, 3.0])


def object_config():
    return map_module.ObjectAggregationConfig(
        dbscan_eps_m=0.35,
        dbscan_min_samples=3,
        trim_quantile=0.0,
        min_object_points=20,
        degeneracy_ratio=1e-4,
    )


def observation(
    track_id: str,
    points: np.ndarray,
    *,
    strong_dynamic: bool = False,
    timestamp: float = 1.0,
    confidence: float = 0.8,
    label: str = "Office Chair",
):
    return map_module.ObjectPointObservation(
        track_id=track_id,
        label=label,
        confidence=confidence,
        timestamp=timestamp,
        strong_dynamic=strong_dynamic,
        points_world=points,
    )


def sorted_points(points: np.ndarray) -> np.ndarray:
    rounded = np.round(points, decimals=6)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    return rounded[order]


def test_named_pose_round_trip_preserves_transform_and_direction() -> None:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (1.0, -2.0, 0.5)
    pose = TWorldCamera.from_matrix(matrix)

    restored = TWorldCamera.from_json_value(pose.to_json_value())
    world_point = restored.transform_points(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    )

    np.testing.assert_allclose(world_point, [[1.0, -2.0, 1.5]], atol=1e-12)
    np.testing.assert_allclose(
        restored.camera_from_world.matrix @ restored.matrix,
        np.eye(4),
        atol=1e-12,
    )
    assert isinstance(restored.camera_from_world, TCameraWorld)


def test_named_pose_rejects_non_rigid_homogeneous_matrix() -> None:
    reflection = np.eye(4, dtype=np.float64)
    reflection[0, 0] = -1.0

    with pytest.raises(ValueError, match="rotation"):
        TWorldCamera.from_matrix(reflection)


def test_tsdf_requires_world_from_camera_not_its_inverse() -> None:
    color, depth_m, scores = synthetic_frames()[0]
    volume = integrator()

    with pytest.raises(TypeError, match="TWorldCamera"):
        volume.integrate(color, depth_m, scores, TWorldCamera.identity().camera_from_world)


def test_three_frame_masked_tsdf_reconstructs_plane_without_moving_square() -> None:
    volume = integrator()
    pose = TWorldCamera.identity()
    for color, depth_m, scores in synthetic_frames():
        volume.integrate(color, depth_m, scores, pose)

    points = volume.extract_points()

    assert points.shape[0] > 500
    assert abs(float(np.median(points[:, 2])) - 1.0) < 0.02
    assert not np.any(points[:, 2] < 0.85)


def test_pose_serialization_round_trip_produces_identical_tsdf_points() -> None:
    direct = integrator()
    restored = integrator()
    for color, depth_m, scores in synthetic_frames():
        pose = TWorldCamera.identity()
        direct.integrate(color, depth_m, scores, pose)
        restored.integrate(
            color,
            depth_m,
            scores,
            TWorldCamera.from_json_value(pose.to_json_value()),
        )

    np.testing.assert_array_equal(
        sorted_points(direct.extract_points()),
        sorted_points(restored.extract_points()),
    )


def test_object_aggregation_never_fuses_a_track_that_becomes_dynamic() -> None:
    points = cuboid_points()
    records = map_module.aggregate_static_objects(
        [
            observation("track-static", points),
            observation("track-dynamic", points, timestamp=1.0),
            observation("track-dynamic", points, strong_dynamic=True, timestamp=2.0),
        ],
        object_config(),
    )

    assert [record.source_track for record in records] == ["track-static"]


def test_oriented_box_recovers_rotated_cuboid_and_removes_far_outlier() -> None:
    points = np.vstack([cuboid_points(), np.array([[20.0, 20.0, 20.0]])])
    record = map_module.aggregate_static_objects(
        [observation("track-0042", points)], object_config()
    )[0]

    assert record.box_fallback is None
    np.testing.assert_allclose(sorted(record.extent, reverse=True), [2.0, 0.8, 0.4], atol=0.05)
    major_axis = np.asarray(record.orientation)[:, 0]
    expected_axis = np.array([np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0)), 0.0])
    assert abs(float(major_axis @ expected_axis)) > 0.98
    assert record.point_count == cuboid_points().shape[0]


def test_degenerate_object_records_axis_aligned_fallback() -> None:
    xy = np.array(np.meshgrid(np.linspace(-0.5, 0.5, 8), np.linspace(-0.2, 0.2, 5)))
    planar = np.column_stack([xy.reshape(2, -1).T, np.zeros(40)])

    record = map_module.aggregate_static_objects(
        [observation("track-planar", planar)], object_config()
    )[0]

    assert record.box_fallback == "AXIS_ALIGNED_DEGENERATE"
    np.testing.assert_array_equal(record.orientation, np.eye(3))
    assert np.all(np.isfinite(record.centroid))
    assert np.all(np.isfinite(record.extent))


def test_exported_map_artifacts_are_reloadable_and_manifest_bound(tmp_path: Path) -> None:
    volume = integrator()
    for color, depth_m, scores in synthetic_frames():
        volume.integrate(color, depth_m, scores, TWorldCamera.identity())
    records = map_module.aggregate_static_objects(
        [observation("track-static", cuboid_points())], object_config()
    )

    artifact_input = {
        "path": "/reproducible/input",
        "sha256": "0" * 64,
        "size_bytes": 1,
    }
    inputs = {
        name: dict(artifact_input)
        for name in (
            "config", "association", "dataset_manifest", "settings",
            "selected_run_manifest", "trajectory", "semantic_manifest",
            "semantic_index", "semantic_completion", "dynamic_manifest",
            "dynamic_index", "dynamic_tracks", "dynamic_completion",
        )
    }
    inputs["dataset_source_tree_sha256"] = "1" * 64
    manifest_base = {
        "schema": "ovorb.map.v1",
        "schema_version": 1,
        "study_id": "p07-static-map-v1",
        "sequence_id": "synthetic",
        "seed": 23011,
        "producer_commit": "2" * 40,
        "command": ["test-export"],
        "created_utc": "2026-08-29T00:00:00Z",
        "hostname": "test-host",
        "valid": True,
        "pose_convention": "T_world_camera",
        "pose_timestamp_identity": "RGB timestamp rounded to six decimal places",
        "config": json.loads(Path("config/P07_MAP.json").read_text(encoding="utf-8")),
        "inputs": inputs,
        "counts": {
            "association_frames": 3,
            "trajectory_poses": 3,
            "integrated_frames": 3,
            "excluded_frames": {"MISSING_EXACT_POSE": 0},
            "dynamic_pixel_exclusions": 192,
            "invalid_depth_pixels": 0,
            "object_observations": 1,
            "static_objects": 1,
            "dynamic_tracks": 1,
        },
        "cache_completion": {"semantic": {}, "dynamic": {}},
        "semantic_cache_producer": "3" * 40,
        "dynamic_cache_producer": "4" * 40,
    }
    manifest = map_module.export_map_artifacts(
        tmp_path / "map",
        volume=volume,
        objects=records,
        dynamic_track_rows=[
            {
                "frame_id": 0,
                "track_id": 7,
                "strong_dynamic": False,
            },
            {
                "frame_id": 1,
                "track_id": 7,
                "strong_dynamic": True,
            },
        ],
        manifest_base=manifest_base,
        screenshot_views=(
            map_module.ScreenshotView("front", 20.0, -60.0),
            map_module.ScreenshotView("top", 90.0, -90.0),
        ),
    )

    expected = {
        "static_mesh.ply",
        "static_cloud.ply",
        "objects.json",
        "dynamic_tracks.jsonl",
        "map_manifest.json",
        "screenshots/front.png",
        "screenshots/top.png",
        "objects/obj-0001.ply",
    }
    observed = {
        str(path.relative_to(tmp_path / "map"))
        for path in (tmp_path / "map").rglob("*")
        if path.is_file()
    }
    assert expected <= observed

    cloud = o3d.io.read_point_cloud(str(tmp_path / "map/static_cloud.ply"))
    mesh = o3d.io.read_triangle_mesh(str(tmp_path / "map/static_mesh.ply"))
    assert len(cloud.points) > 500
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0
    objects = json.loads((tmp_path / "map/objects.json").read_text(encoding="utf-8"))
    assert objects[0]["object_id"] == "obj-0001"

    on_disk_manifest = json.loads(
        (tmp_path / "map/map_manifest.json").read_text(encoding="utf-8")
    )
    assert on_disk_manifest == manifest
    for relative, metadata in manifest["outputs"].items():
        path = tmp_path / "map" / relative
        assert path.stat().st_size == metadata["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]

    from tools.validate_map import validate_map

    report = validate_map(tmp_path / "map")
    assert report["valid"] is True
    assert report["cloud_points"] == len(cloud.points)
    assert report["mesh_triangles"] == len(mesh.triangles)
    assert report["static_objects"] == 1
    assert report["dynamic_tracks"] == 1
    assert report["dynamic_track_rows"] == 2

    manifest_path = tmp_path / "map/map_manifest.json"
    invalid_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalid_manifest["pose_convention"] = "T_camera_world"
    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pose convention"):
        validate_map(tmp_path / "map")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    (tmp_path / "map/objects.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        validate_map(tmp_path / "map")
