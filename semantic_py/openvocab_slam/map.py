"""Static TSDF and semantic object-map primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .query import normalize_text


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if (type(self.width) is not int or type(self.height) is not int or
                self.width <= 0 or self.height <= 0 or
                not all(math.isfinite(value) for value in values) or
                self.fx <= 0.0 or self.fy <= 0.0):
            raise ValueError("invalid camera intrinsics")


@dataclass(frozen=True)
class TsdfConfig:
    voxel_length_m: float
    sdf_trunc_m: float
    depth_trunc_m: float
    dynamic_threshold: float

    def __post_init__(self) -> None:
        values = (
            self.voxel_length_m,
            self.sdf_trunc_m,
            self.depth_trunc_m,
            self.dynamic_threshold,
        )
        if (not all(math.isfinite(value) for value in values) or
                self.voxel_length_m <= 0.0 or self.sdf_trunc_m <= 0.0 or
                self.depth_trunc_m <= 0.0 or
                not 0.0 <= self.dynamic_threshold <= 1.0):
            raise ValueError("invalid TSDF configuration")


def _rigid_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("pose must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
        raise ValueError("pose must have a homogeneous final row")
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or
            not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8)):
        raise ValueError("pose rotation must be proper orthonormal")
    return np.array(matrix, dtype=np.float64, copy=True)


class TCameraWorld:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = _rigid_matrix(matrix)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self._matrix, copy=True)


class TWorldCamera:
    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = _rigid_matrix(matrix)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "TWorldCamera":
        return cls(matrix)

    @classmethod
    def from_json_value(cls, value: Any) -> "TWorldCamera":
        return cls(_rigid_matrix(value))

    @classmethod
    def identity(cls) -> "TWorldCamera":
        return cls(np.eye(4, dtype=np.float64))

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self._matrix, copy=True)

    @property
    def camera_from_world(self) -> TCameraWorld:
        return TCameraWorld(np.linalg.inv(self._matrix))

    def to_json_value(self) -> list[list[float]]:
        return self._matrix.tolist()

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("points must be a finite Nx3 array")
        return values @ self._matrix[:3, :3].T + self._matrix[:3, 3]


class StaticTsdfVolume:
    def __init__(self, intrinsics: CameraIntrinsics, config: TsdfConfig) -> None:
        import open3d as o3d

        self.intrinsics = intrinsics
        self.config = config
        self._o3d = o3d
        self._intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        )
        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=config.voxel_length_m,
            sdf_trunc=config.sdf_trunc_m,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def integrate(
        self,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        dynamic_scores: np.ndarray,
        world_from_camera: TWorldCamera,
    ) -> None:
        if not isinstance(world_from_camera, TWorldCamera):
            raise TypeError("world_from_camera must be TWorldCamera")
        expected_color = (self.intrinsics.height, self.intrinsics.width, 3)
        expected_scalar = (self.intrinsics.height, self.intrinsics.width)
        color = np.asarray(color_rgb)
        depth = np.asarray(depth_m, dtype=np.float32)
        scores = np.asarray(dynamic_scores, dtype=np.float32)
        if (color.shape != expected_color or color.dtype != np.uint8 or
                depth.shape != expected_scalar or scores.shape != expected_scalar):
            raise ValueError("RGB, depth, or score dimensions do not match intrinsics")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("dynamic scores must be finite in [0, 1]")
        valid_depth = (
            np.isfinite(depth) & (depth > 0.0) &
            (depth <= self.config.depth_trunc_m) &
            (scores < self.config.dynamic_threshold)
        )
        masked_depth = np.where(valid_depth, depth, 0.0).astype(np.float32)
        rgbd = self._o3d.geometry.RGBDImage.create_from_color_and_depth(
            self._o3d.geometry.Image(np.ascontiguousarray(color)),
            self._o3d.geometry.Image(np.ascontiguousarray(masked_depth)),
            depth_scale=1.0,
            depth_trunc=self.config.depth_trunc_m,
            convert_rgb_to_intensity=False,
        )
        self._volume.integrate(
            rgbd,
            self._intrinsic,
            world_from_camera.camera_from_world.matrix,
        )

    def extract_points(self) -> np.ndarray:
        points = np.asarray(self.extract_cloud().points, dtype=np.float64)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise RuntimeError("TSDF produced invalid points")
        return np.array(points, copy=True)

    def extract_cloud(self):
        return self._volume.extract_point_cloud()

    def extract_mesh(self):
        mesh = self._volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh


@dataclass(frozen=True)
class ObjectPointObservation:
    track_id: str
    label: str
    confidence: float
    timestamp: float
    strong_dynamic: bool
    points_world: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_world, dtype=np.float64)
        if (not isinstance(self.track_id, str) or not self.track_id or
                not normalize_text(self.label) or
                type(self.confidence) not in (int, float) or
                not math.isfinite(float(self.confidence)) or
                not 0.0 <= float(self.confidence) <= 1.0 or
                type(self.timestamp) not in (int, float) or
                not math.isfinite(float(self.timestamp)) or
                type(self.strong_dynamic) is not bool or
                points.ndim != 2 or points.shape[1] != 3):
            raise ValueError("invalid object point observation")
        object.__setattr__(self, "points_world", np.array(points, copy=True))


@dataclass(frozen=True)
class ObjectAggregationConfig:
    dbscan_eps_m: float
    dbscan_min_samples: int
    trim_quantile: float
    min_object_points: int
    degeneracy_ratio: float
    max_object_points: int = 20000

    def __post_init__(self) -> None:
        if (not math.isfinite(self.dbscan_eps_m) or self.dbscan_eps_m <= 0.0 or
                type(self.dbscan_min_samples) is not int or self.dbscan_min_samples <= 0 or
                not math.isfinite(self.trim_quantile) or
                not 0.0 <= self.trim_quantile < 0.5 or
                type(self.min_object_points) is not int or self.min_object_points <= 0 or
                type(self.max_object_points) is not int or
                self.max_object_points < self.min_object_points or
                not math.isfinite(self.degeneracy_ratio) or
                not 0.0 <= self.degeneracy_ratio < 1.0):
            raise ValueError("invalid object aggregation configuration")


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    normalized_label: str
    aliases: tuple[str, ...]
    confidence: float
    confidence_history: tuple[float, ...]
    observation_range: tuple[float, float]
    centroid: np.ndarray
    orientation: np.ndarray
    extent: np.ndarray
    point_count: int
    source_track: str
    box_fallback: str | None
    points_world: np.ndarray

    def to_primitive(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "normalized_label": self.normalized_label,
            "aliases": list(self.aliases),
            "confidence": self.confidence,
            "confidence_history": list(self.confidence_history),
            "observation_range": list(self.observation_range),
            "centroid": self.centroid.tolist(),
            "orientation": self.orientation.tolist(),
            "extent": self.extent.tolist(),
            "point_count": self.point_count,
            "source_track": self.source_track,
            "box_fallback": self.box_fallback,
        }


def _clean_object_points(points: np.ndarray, config: ObjectAggregationConfig) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.shape[0] < config.min_object_points:
        return np.empty((0, 3), dtype=np.float64)
    labels = DBSCAN(
        eps=config.dbscan_eps_m,
        min_samples=config.dbscan_min_samples,
    ).fit_predict(finite)
    cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    if cluster_ids.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    best = int(cluster_ids[np.argmax(counts)])
    clustered = finite[labels == best]
    if config.trim_quantile > 0.0:
        lower = np.quantile(clustered, config.trim_quantile, axis=0)
        upper = np.quantile(clustered, 1.0 - config.trim_quantile, axis=0)
        clustered = clustered[np.all((clustered >= lower) & (clustered <= upper), axis=1)]
    if clustered.shape[0] < config.min_object_points:
        return np.empty((0, 3), dtype=np.float64)
    return clustered


def _fit_box(
    points: np.ndarray, degeneracy_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    mean = np.mean(points, axis=0)
    covariance = np.cov(points - mean, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    orientation = eigenvectors[:, order]
    if np.linalg.det(orientation) < 0.0:
        orientation[:, -1] *= -1.0
    degenerate = (
        eigenvalues[0] <= np.finfo(np.float64).eps or
        eigenvalues[-1] / eigenvalues[0] < degeneracy_ratio
    )
    if degenerate:
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        return (lower + upper) / 2.0, np.eye(3), upper - lower, "AXIS_ALIGNED_DEGENERATE"
    local = (points - mean) @ orientation
    lower = np.min(local, axis=0)
    upper = np.max(local, axis=0)
    centroid = mean + orientation @ ((lower + upper) / 2.0)
    return centroid, orientation, upper - lower, None


def aggregate_static_objects(
    observations: list[ObjectPointObservation],
    config: ObjectAggregationConfig,
) -> list[ObjectRecord]:
    grouped: dict[str, list[ObjectPointObservation]] = {}
    for item in observations:
        if not isinstance(item, ObjectPointObservation):
            raise TypeError("observations must contain ObjectPointObservation")
        grouped.setdefault(item.track_id, []).append(item)
    records: list[ObjectRecord] = []
    for track_id in sorted(grouped):
        track = grouped[track_id]
        if any(item.strong_dynamic for item in track):
            continue
        fused = np.concatenate([item.points_world for item in track], axis=0)
        if fused.shape[0] > config.max_object_points:
            fused = fused[np.linspace(
                0, fused.shape[0] - 1, config.max_object_points, dtype=int
            )]
        points = _clean_object_points(fused, config)
        if points.shape[0] == 0:
            continue
        labels = [normalize_text(item.label) for item in track]
        canonical = sorted(set(labels), key=lambda value: (-labels.count(value), value))[0]
        aliases = tuple(sorted(set(labels) - {canonical}))
        ordered = sorted(track, key=lambda item: (item.timestamp, item.label))
        history = tuple(float(item.confidence) for item in ordered)
        centroid, orientation, extent, fallback = _fit_box(
            points, config.degeneracy_ratio
        )
        records.append(ObjectRecord(
            object_id=f"obj-{len(records) + 1:04d}",
            normalized_label=canonical,
            aliases=aliases,
            confidence=max(history),
            confidence_history=history,
            observation_range=(ordered[0].timestamp, ordered[-1].timestamp),
            centroid=centroid,
            orientation=orientation,
            extent=extent,
            point_count=int(points.shape[0]),
            source_track=track_id,
            box_fallback=fallback,
            points_world=points,
        ))
    return records


@dataclass(frozen=True)
class ScreenshotView:
    name: str
    elevation_degrees: float
    azimuth_degrees: float

    def __post_init__(self) -> None:
        if (not isinstance(self.name, str) or not self.name or
                any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
                    for character in self.name) or
                not math.isfinite(self.elevation_degrees) or
                not math.isfinite(self.azimuth_degrees)):
            raise ValueError("invalid screenshot view")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _render_screenshot(
    path: Path,
    points: np.ndarray,
    objects: list[ObjectRecord],
    view: ScreenshotView,
) -> None:
    from PIL import Image, ImageDraw

    width, height = 960, 720
    margin = 40
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    azimuth = math.radians(view.azimuth_degrees)
    elevation = math.radians(view.elevation_degrees)
    right = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    up = np.array([
        -math.sin(elevation) * math.cos(azimuth),
        -math.sin(elevation) * math.sin(azimuth),
        math.cos(elevation),
    ])
    indices = np.linspace(
        0, points.shape[0] - 1, min(points.shape[0], 50000), dtype=int
    ) if points.shape[0] else np.empty((0,), dtype=int)
    sample = points[indices]
    projected = np.column_stack((sample @ right, sample @ up))
    object_projected = np.array([
        [record.centroid @ right, record.centroid @ up] for record in objects
    ], dtype=np.float64).reshape((-1, 2))
    bounds_source = (
        np.vstack((projected, object_projected))
        if object_projected.shape[0] else projected
    )
    if bounds_source.shape[0]:
        lower = np.min(bounds_source, axis=0)
        upper = np.max(bounds_source, axis=0)
        span = np.maximum(upper - lower, 1e-9)
        scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
        center = (lower + upper) / 2.0

        def pixel(values: np.ndarray) -> tuple[int, int]:
            x = int(round((values[0] - center[0]) * scale + width / 2.0))
            y = int(round(height / 2.0 - (values[1] - center[1]) * scale))
            return x, y

        if sample.shape[0]:
            z_lower = float(np.min(sample[:, 2]))
            z_span = max(float(np.max(sample[:, 2])) - z_lower, 1e-9)
            for coordinates, point in zip(projected, sample):
                ratio = (float(point[2]) - z_lower) / z_span
                color = (
                    int(50 + 180 * ratio),
                    int(80 + 120 * (1.0 - abs(2.0 * ratio - 1.0))),
                    int(210 - 160 * ratio),
                )
                draw.point(pixel(coordinates), fill=color)
        for record, coordinates in zip(objects, object_projected):
            x, y = pixel(coordinates)
            draw.line((x - 4, y - 4, x + 4, y + 4), fill="#ef4444", width=2)
            draw.line((x - 4, y + 4, x + 4, y - 4), fill="#ef4444", width=2)
            draw.text((x + 6, y - 6), record.object_id, fill="#991b1b")
    draw.rectangle((margin, margin, width - margin, height - margin), outline="#64748b")
    draw.text((margin, 12), f"{view.name}: fixed world projection", fill="#0f172a")
    image.save(path, format="PNG", optimize=False)


def export_map_artifacts(
    output_root: Path,
    *,
    volume: StaticTsdfVolume,
    objects: list[ObjectRecord],
    dynamic_track_rows: list[dict[str, object]],
    manifest_base: dict[str, object],
    screenshot_views: tuple[ScreenshotView, ...],
) -> dict[str, object]:
    """Atomically export a reloadable map and bind every payload by SHA-256."""
    import open3d as o3d

    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"map output already exists: {root}")
    staging = root.parent / f".{root.name}.partial"
    if staging.exists():
        raise FileExistsError(f"stale map staging directory exists: {staging}")
    staging.mkdir(parents=True)
    (staging / "objects").mkdir()
    (staging / "screenshots").mkdir()
    try:
        cloud = volume.extract_cloud()
        mesh = volume.extract_mesh()
        if len(cloud.points) == 0 or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            raise ValueError("TSDF extraction produced an empty map")
        if not o3d.io.write_point_cloud(
                str(staging / "static_cloud.ply"), cloud, write_ascii=False):
            raise OSError("failed to write static cloud")
        if not o3d.io.write_triangle_mesh(
                str(staging / "static_mesh.ply"), mesh, write_ascii=False):
            raise OSError("failed to write static mesh")
        _write_json(staging / "objects.json", [item.to_primitive() for item in objects])
        with (staging / "dynamic_tracks.jsonl").open("w", encoding="utf-8") as stream:
            for row in dynamic_track_rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        for record in objects:
            object_cloud = o3d.geometry.PointCloud()
            object_cloud.points = o3d.utility.Vector3dVector(record.points_world)
            if not o3d.io.write_point_cloud(
                    str(staging / "objects" / f"{record.object_id}.ply"),
                    object_cloud,
                    write_ascii=False):
                raise OSError(f"failed to write object cloud: {record.object_id}")
        points = np.asarray(cloud.points, dtype=np.float64)
        for view in screenshot_views:
            _render_screenshot(
                staging / "screenshots" / f"{view.name}.png",
                points,
                objects,
                view,
            )
        outputs: dict[str, object] = {}
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            relative = str(path.relative_to(staging))
            outputs[relative] = {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        manifest = {**manifest_base, "outputs": outputs}
        _write_json(staging / "map_manifest.json", manifest)
        root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, root)
        return manifest
    except BaseException:
        # Keep partial outputs for diagnosis; callers may choose a fresh output path.
        raise
