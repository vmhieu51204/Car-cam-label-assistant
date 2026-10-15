"""Constant-velocity Kalman tracking with gated upright 3D GIoU association."""

from dataclasses import dataclass
from math import hypot
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.models.perception import Box3D, Detection, Frame, Observation, Track
from src.services.geometry.boxes import giou3d


@dataclass
class State:
    track: Track
    mean: np.ndarray
    covariance: np.ndarray
    box: Box3D
    missed: int = 0

    def predict(self, elapsed: float) -> None:
        """Advance the association state, never create an unobserved label."""
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * elapsed
        noise = np.vstack([np.eye(3) * elapsed**2 / 2, np.eye(3) * elapsed])
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T + noise @ noise.T
        self.box = self.box.model_copy(update={"center": tuple(self.mean[:3])})

    def update(self, detection: Detection) -> None:
        """Apply a measured center with a numerically stable covariance update."""
        measurement_noise = np.eye(3) * 0.15**2
        innovation_covariance = self.covariance[:3, :3] + measurement_noise
        gain = np.linalg.solve(innovation_covariance, self.covariance[:3, :]).T
        self.mean += gain @ (np.asarray(detection.box.center) - self.mean[:3])
        residual = np.eye(6)
        residual[:, :3] -= gain
        self.covariance = residual @ self.covariance @ residual.T + gain @ measurement_noise @ gain.T
        self.box = detection.box.model_copy(update={"center": tuple(self.mean[:3])})
        self.missed = 0


class Tracker:
    """One scene per instance; global coordinates remove ego-motion ambiguity."""

    def __init__(
        self,
        match_giou: float = 0.0,
        max_missed: int = 2,
        data_source: Literal["mock", "real"] = "mock",
    ) -> None:
        self.match_giou = match_giou
        self.max_missed = max_missed
        self.data_source = data_source
        self.active: list[State] = []
        self.tracks: list[Track] = []
        self.scene: str | None = None
        self.timestamp: float | None = None
        self.frame_index: int | None = None

    def step(self, frame: Frame) -> list[str]:
        """Return track IDs in detection order, preserving identity across short gaps."""
        identities = [""] * len(frame.detections)
        elapsed = self._advance(frame)
        for state in self.active:
            state.predict(elapsed)
        matches = self._match(frame.detections)
        for index, state in enumerate(self.active):
            if index not in matches:
                state.missed += 1
                continue
            detection = frame.detections[matches[index]]
            identities[matches[index]] = state.track.track_id
            state.update(detection)
            self._append(state.track, frame, detection)
        self.active = [state for state in self.active if state.missed <= self.max_missed]
        used = set(matches.values())
        for index, detection in enumerate(frame.detections):
            if index not in used:
                identities[index] = self._create(frame, detection)
        return identities

    def _advance(self, frame: Frame) -> float:
        if self.scene is not None and frame.scene != self.scene:
            raise ValueError("Create a new tracker for each scene")
        if (
            self.timestamp is not None
            and self.frame_index is not None
            and (frame.timestamp_s <= self.timestamp or frame.frame_index <= self.frame_index)
        ):
            raise ValueError("Frames must have strictly increasing timestamps and indices")
        elapsed = 0.0 if self.timestamp is None else frame.timestamp_s - self.timestamp
        self.scene, self.timestamp, self.frame_index = frame.scene, frame.timestamp_s, frame.frame_index
        return elapsed

    def _match(self, detections: list[Detection]) -> dict[int, int]:
        count_tracks, count_detections = len(self.active), len(detections)
        if not count_tracks:
            return {}
        # Dummy assignments allow misses. Gate before Hungarian assignment.
        costs = np.full((count_tracks, count_detections + count_tracks), 3.0)
        for i, state in enumerate(self.active):
            for j, detection in enumerate(detections):
                if not self._candidate(state, detection):
                    continue
                score = giou3d(state.box, detection.box)
                if score >= self.match_giou:
                    costs[i, j] = 1 - score
        rows, cols = linear_sum_assignment(costs)
        return {int(i): int(j) for i, j in zip(rows, cols) if j < count_detections and costs[i, j] < 3}

    def _candidate(self, state: State, detection: Detection) -> bool:
        if state.track.label != detection.label:
            return False
        if self.match_giou < 0:
            return True
        distance = hypot(state.box.center[0] - detection.box.center[0], state.box.center[1] - detection.box.center[1])
        radius = (hypot(*state.box.size_lwh[:2]) + hypot(*detection.box.size_lwh[:2])) / 2
        return distance <= radius

    def _create(self, frame: Frame, detection: Detection) -> str:
        track = Track(
            track_id=f"{frame.scene}:{len(self.tracks) + 1:06d}",
            scene=frame.scene,
            label=detection.label,
            confidence=detection.confidence,
            data_source=self.data_source,
            observations=[],
        )
        self._append(track, frame, detection)
        self.tracks.append(track)
        self.active.append(
            State(
                track=track,
                mean=np.r_[detection.box.center, [0.0, 0.0, 0.0]],
                covariance=np.diag([0.15**2] * 3 + [25.0] * 3),
                box=detection.box,
            )
        )
        return track.track_id

    @staticmethod
    def _append(track: Track, frame: Frame, detection: Detection) -> None:
        track.observations.append(
            Observation(
                sample_token=frame.sample_token,
                frame_index=frame.frame_index,
                timestamp_s=frame.timestamp_s,
                box=detection.box,
                confidence=detection.confidence,
            )
        )
        track.confidence = sum(item.confidence for item in track.observations) / len(track.observations)
