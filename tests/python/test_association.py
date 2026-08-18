import numpy as np

from semantic_py.openvocab_slam.association import AssociationObservation, associate_tracks
from semantic_py.openvocab_slam.config import DynamicConfirmationConfig
from semantic_py.openvocab_slam.motion import DynamicTrack


def observation(local_id: int, label: str, x: float, mask: np.ndarray) -> AssociationObservation:
    return AssociationObservation(local_id, label, np.array([x, 0.0, 2.0]), mask)


def track(track_id: int, label: str, x: float) -> DynamicTrack:
    return DynamicTrack.new(track_id, label, np.array([x, 0.0, 2.0]), timestamp=0.0)


def test_crossing_image_masks_keep_3d_track_ids() -> None:
    # Catches association that chooses image overlap over the frozen 3D cost.
    left = np.array([[True, True, False, False]], dtype=bool)
    right = np.array([[False, False, True, True]], dtype=bool)
    result = associate_tracks(
        [track(11, "person", 0.0), track(22, "person", 4.0)],
        [observation(0, "person", 4.1, left), observation(1, "person", 0.1, right)],
        DynamicConfirmationConfig.frozen(),
    )

    assert result.assignments == {11: 1, 22: 0}
    assert result.unassigned_tracks == ()
    assert result.unassigned_observations == ()


def test_association_penalizes_label_mismatch_but_only_gates_distance() -> None:
    # Catches incorrectly treating exact-label similarity as a hard gate.
    result = associate_tracks(
        [track(1, "person", 0.0)],
        [
            observation(0, "chair", 0.1, np.ones((1, 1), dtype=bool)),
            observation(1, "person", 1.01, np.ones((1, 1), dtype=bool)),
        ],
        DynamicConfirmationConfig.frozen(),
    )

    assert result.assignments == {1: 0}
    assert result.unassigned_tracks == ()
    assert result.unassigned_observations == (1,)


def test_exact_label_wins_when_label_penalty_outweighs_small_distance_advantage() -> None:
    # Catches omitting the literal 0.15 mismatch term from the frozen cost.
    mask = np.ones((1, 1), dtype=bool)
    item = track(5, "person", 0.0)
    item.last_mask = mask
    result = associate_tracks(
        [item],
        [observation(0, "chair", 0.0, mask), observation(1, "person", 0.2, mask)],
        DynamicConfirmationConfig.frozen(),
    )

    assert result.assignments == {5: 1}


def test_three_frame_occlusion_reassociates_the_surviving_track_id() -> None:
    # Catches dropping or replacing a track before its five-miss lifetime expires.
    item = track(17, "person", 0.0)
    for timestamp in (1.0, 2.0, 3.0):
        item.mark_missed(timestamp)
    result = associate_tracks(
        [item],
        [observation(0, "person", 0.1, np.ones((1, 1), dtype=bool))],
        DynamicConfirmationConfig.frozen(),
    )

    assert item.terminated is False
    assert result.assignments == {17: 0}


def test_track_is_terminated_after_sixth_miss() -> None:
    # Catches the off-by-one lifecycle bug: config permits five misses, not six.
    item = track(1, "person", 0.0)
    for frame in range(1, 6):
        item.mark_missed(float(frame))
        assert item.terminated is False
    item.mark_missed(6.0)
    assert item.terminated is True
