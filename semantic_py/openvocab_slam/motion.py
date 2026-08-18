from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DynamicConfirmationConfig


@dataclass(frozen=True)
class TrackState:
    track_id: int
    dynamic_probability: float
    strong_dynamic: bool
    observation_count: int
    confirming_observations: int
    reason: str | None


@dataclass
class DynamicTrack:
    track_id: int
    label: str
    state: np.ndarray
    covariance: np.ndarray
    timestamp: float
    last_measurement: np.ndarray
    observation_count: int = 1
    confirming_observations: int = 0
    dynamic_probability: float = 0.0
    strong_dynamic: bool = False
    misses: int = 0
    terminated: bool = False
    last_mask: np.ndarray | None = None

    @classmethod
    def new(
        cls,
        track_id: int,
        label: str,
        centroid: np.ndarray,
        *,
        timestamp: float,
        config: DynamicConfirmationConfig | None = None,
    ) -> "DynamicTrack":
        position = np.asarray(centroid, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("track centroid must be finite 3-vector")
        cfg = config or DynamicConfirmationConfig.frozen()
        covariance = np.diag(
            [cfg.kalman_initial_position_variance] * 3
            + [cfg.kalman_initial_velocity_variance] * 3
        )
        return cls(track_id, label, np.r_[position, np.zeros(3)], covariance, float(timestamp), position.copy())

    @property
    def predicted_position(self) -> np.ndarray:
        return self.state[:3].copy()

    def predict(self, timestamp: float, config: DynamicConfirmationConfig) -> np.ndarray:
        """Advance causally to ``timestamp`` and return the predicted centroid."""
        return self._predict(timestamp, config)

    def _predict(self, timestamp: float, config: DynamicConfirmationConfig) -> np.ndarray:
        dt = float(timestamp) - self.timestamp
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("track timestamps must be finite and monotonic")
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        q = config.kalman_process_acceleration_variance
        process = np.zeros((6, 6))
        process[:3, :3] = np.eye(3) * (dt**4 / 4.0) * q
        process[:3, 3:] = np.eye(3) * (dt**3 / 2.0) * q
        process[3:, :3] = process[:3, 3:]
        process[3:, 3:] = np.eye(3) * dt**2 * q
        predicted = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.state = predicted
        self.timestamp = float(timestamp)
        return predicted[:3].copy()

    def update(
        self,
        centroid: np.ndarray,
        *,
        timestamp: float,
        mad: np.ndarray,
        config: DynamicConfirmationConfig,
    ) -> TrackState:
        measurement = np.asarray(centroid, dtype=np.float64)
        uncertainty = np.asarray(mad, dtype=np.float64)
        if (
            measurement.shape != (3,)
            or uncertainty.shape != (3,)
            or not np.all(np.isfinite(measurement))
            or not np.all(np.isfinite(uncertainty))
        ):
            raise ValueError("centroid and MAD must be finite 3-vectors")
        predicted = self._predict(timestamp, config)
        # The last camera-compensated world measurement is the static-scene
        # prediction used for motion evidence.  The Kalman prediction remains
        # responsible for association, but must not turn a stopped object into
        # continued positive evidence merely because its velocity estimate
        # needs time to settle.
        residual = float(np.linalg.norm(measurement - self.last_measurement))
        threshold = max(
            config.base_motion_threshold_m,
            config.robust_sigma_multiplier * float(np.linalg.norm(uncertainty)),
        )
        moving = residual > threshold
        innovation = measurement - predicted
        observation = np.zeros((3, 6))
        observation[:, :3] = np.eye(3)
        innovation_covariance = (
            observation @ self.covariance @ observation.T
            + np.eye(3) * config.kalman_measurement_variance
        )
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(6) - gain @ observation) @ self.covariance
        self.last_measurement = measurement.copy()
        self.observation_count += 1
        self.misses = 0
        if moving:
            self.confirming_observations += 1
            self.dynamic_probability = min(
                1.0, round(self.dynamic_probability + config.dynamic_evidence_increment, 12)
            )
        else:
            self.confirming_observations = 0
            self.dynamic_probability = max(
                0.0, round(self.dynamic_probability - config.static_evidence_decrement, 12)
            )
        if not self.strong_dynamic:
            self.strong_dynamic = (
                self.dynamic_probability >= config.dynamic_enter_threshold
                and self.confirming_observations >= config.min_confirming_observations
            )
        elif self.dynamic_probability < config.dynamic_exit_threshold:
            self.strong_dynamic = False
        return TrackState(
            self.track_id,
            self.dynamic_probability,
            self.strong_dynamic,
            self.observation_count,
            self.confirming_observations,
            None,
        )

    def mark_missed(self, timestamp: float, config: DynamicConfirmationConfig | None = None) -> TrackState:
        cfg = config or DynamicConfirmationConfig.frozen()
        self._predict(timestamp, cfg)
        self.misses += 1
        self.confirming_observations = 0
        self.dynamic_probability = max(
            0.0, round(self.dynamic_probability - cfg.static_evidence_decrement, 12)
        )
        if self.dynamic_probability < cfg.dynamic_exit_threshold:
            self.strong_dynamic = False
        self.terminated = self.misses > cfg.max_track_misses
        return TrackState(
            self.track_id,
            self.dynamic_probability,
            self.strong_dynamic,
            self.observation_count,
            self.confirming_observations,
            "TRACK_MISSED",
        )
