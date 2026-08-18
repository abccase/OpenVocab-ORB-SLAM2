from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class InferenceConfig:
    schema: str
    image_long_side: int
    box_threshold: float
    text_threshold: float
    mask_threshold: float

    def __post_init__(self) -> None:
        if not self.schema:
            raise ValueError("schema must not be empty")
        if self.image_long_side <= 0:
            raise ValueError("image_long_side must be positive")
        for name in ("box_threshold", "text_threshold", "mask_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

    def to_primitive(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "image_long_side": self.image_long_side,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "mask_threshold": self.mask_threshold,
        }

    def sha256(self) -> str:
        payload = json.dumps(self.to_primitive(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DynamicConfirmationConfig:
    """Frozen, reproducibility-critical dynamic-confirmation parameters."""

    schema: str
    min_valid_depth_pixels: int
    min_confirming_observations: int
    base_motion_threshold_m: float
    robust_sigma_multiplier: float
    dynamic_enter_threshold: float
    dynamic_exit_threshold: float
    unknown_dynamic_probability: float
    uncertain_retention_fraction: float
    max_track_misses: int
    centroid_3d_weight: float
    mask_iou_weight: float
    label_weight: float
    association_gate_m: float
    depth_scale: float
    kalman_initial_position_variance: float
    kalman_initial_velocity_variance: float
    kalman_process_acceleration_variance: float
    kalman_measurement_variance: float
    dynamic_evidence_increment: float
    static_evidence_decrement: float
    diagnostic_sample_limit: int

    @classmethod
    def frozen(cls) -> "DynamicConfirmationConfig":
        return cls(
            schema="ovorb.dynamic-cache.v1",
            min_valid_depth_pixels=100,
            min_confirming_observations=3,
            base_motion_threshold_m=0.10,
            robust_sigma_multiplier=3.0,
            dynamic_enter_threshold=0.70,
            dynamic_exit_threshold=0.40,
            unknown_dynamic_probability=0.25,
            uncertain_retention_fraction=0.50,
            max_track_misses=5,
            centroid_3d_weight=0.55,
            mask_iou_weight=0.30,
            label_weight=0.15,
            association_gate_m=1.0,
            depth_scale=5000.0,
            kalman_initial_position_variance=0.01,
            kalman_initial_velocity_variance=1.0,
            kalman_process_acceleration_variance=0.01,
            kalman_measurement_variance=0.0001,
            dynamic_evidence_increment=0.35,
            static_evidence_decrement=0.20,
            diagnostic_sample_limit=512,
        )

    def __post_init__(self) -> None:
        if not self.schema:
            raise ValueError("schema must not be empty")
        if self.min_valid_depth_pixels <= 0 or self.min_confirming_observations <= 0:
            raise ValueError("minimum counts must be positive")
        if self.max_track_misses < 0 or self.diagnostic_sample_limit <= 0:
            raise ValueError("lifecycle and diagnostic limits are invalid")
        values = self.to_primitive()
        for name, value in values.items():
            if isinstance(value, float) and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if not self.dynamic_exit_threshold < self.dynamic_enter_threshold <= 1.0:
            raise ValueError("dynamic hysteresis thresholds are invalid")
        if not 0.0 <= self.unknown_dynamic_probability < self.dynamic_enter_threshold:
            raise ValueError("unknown dynamic probability must remain below the entry threshold")
        if not 0.0 < self.uncertain_retention_fraction <= 1.0:
            raise ValueError("uncertain retention fraction must be in (0, 1]")
        if not math.isclose(
            self.centroid_3d_weight + self.mask_iou_weight + self.label_weight, 1.0, abs_tol=1e-12
        ):
            raise ValueError("association weights must sum to one")

    def to_primitive(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "min_valid_depth_pixels": self.min_valid_depth_pixels,
            "min_confirming_observations": self.min_confirming_observations,
            "base_motion_threshold_m": self.base_motion_threshold_m,
            "robust_sigma_multiplier": self.robust_sigma_multiplier,
            "dynamic_enter_threshold": self.dynamic_enter_threshold,
            "dynamic_exit_threshold": self.dynamic_exit_threshold,
            "unknown_dynamic_probability": self.unknown_dynamic_probability,
            "uncertain_retention_fraction": self.uncertain_retention_fraction,
            "max_track_misses": self.max_track_misses,
            "centroid_3d_weight": self.centroid_3d_weight,
            "mask_iou_weight": self.mask_iou_weight,
            "label_weight": self.label_weight,
            "association_gate_m": self.association_gate_m,
            "depth_scale": self.depth_scale,
            "kalman_initial_position_variance": self.kalman_initial_position_variance,
            "kalman_initial_velocity_variance": self.kalman_initial_velocity_variance,
            "kalman_process_acceleration_variance": self.kalman_process_acceleration_variance,
            "kalman_measurement_variance": self.kalman_measurement_variance,
            "dynamic_evidence_increment": self.dynamic_evidence_increment,
            "static_evidence_decrement": self.static_evidence_decrement,
            "diagnostic_sample_limit": self.diagnostic_sample_limit,
        }

    def sha256(self) -> str:
        payload = json.dumps(self.to_primitive(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def normalize_formal_prompt(prompt: str) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in prompt.replace("\n", " ").split("."):
        term = " ".join(raw_term.strip().lower().split())
        if term and term not in seen:
            terms.append(term)
            seen.add(term)
    if not terms:
        raise ValueError("formal prompt has no terms")
    return " . ".join(terms) + " ."
