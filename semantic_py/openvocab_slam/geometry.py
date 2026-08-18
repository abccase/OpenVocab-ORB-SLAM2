from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CentroidResult:
    valid: bool
    reason: str | None
    centroid_world: np.ndarray | None
    mad_world: np.ndarray | None
    valid_depth_pixels: int
    diagnostic_points_world: np.ndarray


def backproject_pixel(u: float, v: float, depth_m: float, intrinsics: np.ndarray) -> np.ndarray:
    """Return a camera-frame point using the explicit RGB-D pinhole convention."""
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if not np.isfinite(depth_m) or depth_m <= 0.0 or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("depth and focal lengths must be positive and finite")
    return np.array(
        [
            (float(u) - matrix[0, 2]) * depth_m / matrix[0, 0],
            (float(v) - matrix[1, 2]) * depth_m / matrix[1, 1],
            depth_m,
        ],
        dtype=np.float64,
    )


def transform_point(T_world_camera: np.ndarray, point_camera: np.ndarray) -> np.ndarray:
    matrix = np.asarray(T_world_camera, dtype=np.float64)
    point = np.asarray(point_camera, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or point.shape != (3,)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(point))
    ):
        raise ValueError("T_world_camera and point must be finite 4x4 and 3-vector values")
    homogeneous = matrix @ np.append(point, 1.0)
    if not np.isfinite(homogeneous[3]) or abs(homogeneous[3]) < 1e-12:
        raise ValueError("invalid homogeneous transformed point")
    return homogeneous[:3] / homogeneous[3]


def centroid_from_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    T_world_camera: np.ndarray,
    *,
    min_valid_depth_pixels: int = 100,
    diagnostic_sample_limit: int = 512,
) -> CentroidResult:
    binary_mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    if binary_mask.ndim != 2 or depth.shape != binary_mask.shape:
        raise ValueError("mask and depth must be aligned two-dimensional arrays")
    if min_valid_depth_pixels <= 0 or diagnostic_sample_limit <= 0:
        raise ValueError("minimum valid depth and diagnostic limits must be positive")
    valid = binary_mask & np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    count = len(rows)
    if count < min_valid_depth_pixels:
        return CentroidResult(False, "INSUFFICIENT_VALID_DEPTH", None, None, count, np.empty((0, 3), dtype=np.float64))
    K = np.asarray(intrinsics, dtype=np.float64)
    points_camera = np.column_stack(
        (
            (columns - K[0, 2]) * depth[rows, columns] / K[0, 0],
            (rows - K[1, 2]) * depth[rows, columns] / K[1, 1],
            depth[rows, columns],
        )
    )
    rotation = np.asarray(T_world_camera, dtype=np.float64)[:3, :3]
    translation = np.asarray(T_world_camera, dtype=np.float64)[:3, 3]
    points_world = points_camera @ rotation.T + translation
    centroid = np.median(points_world, axis=0)
    mad = np.median(np.abs(points_world - centroid), axis=0)
    sample_indices = np.linspace(0, count - 1, min(count, diagnostic_sample_limit), dtype=int)
    return CentroidResult(True, None, centroid, mad, count, points_world[sample_indices])
