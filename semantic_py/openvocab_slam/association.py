from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import DynamicConfirmationConfig
from .motion import DynamicTrack


@dataclass(frozen=True)
class AssociationObservation:
    local_id: int
    label: str
    centroid_world: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class AssociationResult:
    assignments: dict[int, int]
    unassigned_tracks: tuple[int, ...]
    unassigned_observations: tuple[int, ...]


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.shape != b.shape:
        return 0.0
    union = np.count_nonzero(a | b)
    return 0.0 if union == 0 else float(np.count_nonzero(a & b) / union)


def associate_tracks(
    tracks: list[DynamicTrack],
    observations: list[AssociationObservation],
    config: DynamicConfirmationConfig,
) -> AssociationResult:
    active = [track for track in tracks if not track.terminated]
    if not active or not observations:
        return AssociationResult({}, tuple(track.track_id for track in active), tuple(range(len(observations))))
    # scipy rejects a matrix when any row has no finite candidate.  Keep
    # forbidden pairs finite for the solver, then discard them below.
    invalid = 1_000_000.0
    costs = np.full((len(active), len(observations)), invalid, dtype=np.float64)
    for track_index, track in enumerate(active):
        for observation_index, observation in enumerate(observations):
            centroid = np.asarray(observation.centroid_world, dtype=np.float64)
            distance = float(np.linalg.norm(track.predicted_position - centroid))
            if distance > config.association_gate_m:
                continue
            label_similarity = float(track.label == observation.label)
            previous_mask = (
                track.last_mask
                if track.last_mask is not None
                else np.zeros_like(observation.mask, dtype=bool)
            )
            costs[track_index, observation_index] = (
                config.centroid_3d_weight * distance / config.association_gate_m
                + config.mask_iou_weight * (1.0 - _mask_iou(previous_mask, observation.mask))
                + config.label_weight * (1.0 - label_similarity)
            )
    rows, columns = linear_sum_assignment(costs)
    assignments: dict[int, int] = {}
    for row, column in zip(rows.tolist(), columns.tolist()):
        if costs[row, column] != invalid:
            assignments[active[row].track_id] = column
    assigned_tracks = set(assignments)
    assigned_observations = set(assignments.values())
    return AssociationResult(
        assignments,
        tuple(track.track_id for track in active if track.track_id not in assigned_tracks),
        tuple(index for index in range(len(observations)) if index not in assigned_observations),
    )
