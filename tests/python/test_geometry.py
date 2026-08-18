import numpy as np

from semantic_py.openvocab_slam.geometry import (
    centroid_from_mask,
    backproject_pixel,
    transform_point,
)


K = np.array([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]])


def test_backproject_then_transform_known_point() -> None:
    # Catches camera/world inversion and millimetre-as-metre depth regressions.
    pose = np.eye(4)
    pose[:3, 3] = [1.0, -2.0, 3.0]
    point_camera = backproject_pixel(420, 190, 2.0, K)

    np.testing.assert_allclose(point_camera, [2.0, -1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(transform_point(pose, point_camera), [3.0, -3.0, 5.0], atol=1e-6)


def test_centroid_rejects_insufficient_valid_depth() -> None:
    # Catches accepting a sparse mask as strong 3D evidence.
    mask = np.array([[True, True], [True, True]], dtype=bool)
    depth = np.array([[1.0, 0.0], [np.nan, -1.0]], dtype=np.float32)

    result = centroid_from_mask(mask, depth, np.eye(3), np.eye(4), min_valid_depth_pixels=2)

    assert result.valid is False
    assert result.reason == "INSUFFICIENT_VALID_DEPTH"
    assert result.valid_depth_pixels == 1


def test_centroid_uses_component_median_and_mad() -> None:
    # Catches an arithmetic-mean implementation that lets one depth outlier move the track.
    mask = np.ones((1, 5), dtype=bool)
    depth = np.array([[1.0, 1.0, 1.0, 1.0, 9.0]], dtype=np.float64)

    result = centroid_from_mask(mask, depth, np.eye(3), np.eye(4), min_valid_depth_pixels=5)

    assert result.valid is True
    np.testing.assert_allclose(result.centroid_world, [2.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(result.mad_world, [1.0, 0.0, 0.0], atol=1e-6)
