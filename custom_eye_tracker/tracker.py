"""Reusable gaze-estimation pipeline, independent of any poster application."""

from dataclasses import dataclass

import numpy as np

from .gaze_model import RidgeGazeModel
from .smoothing import ExponentialSmoother


@dataclass(frozen=True)
class GazeEstimate:
    timestamp: float
    raw_xy: tuple[float, float]
    filtered_xy: tuple[float, float]
    valid: bool
    quality: float


class CustomEyeTracker:
    """Map landmark features to calibrated display-canvas coordinates."""

    def __init__(self, model: RidgeGazeModel, smoothing_time_constant_seconds: float = 0.10) -> None:
        if not model.fitted:
            raise ValueError("CustomEyeTracker requires a fitted model.")
        self.model = model
        self.smoother = ExponentialSmoother(smoothing_time_constant_seconds)

    def get_gaze(self, features, timestamp: float, quality: float = 1.0, valid: bool = True) -> GazeEstimate:
        if not valid:
            self.smoother.reset()
            return GazeEstimate(timestamp, (float("nan"), float("nan")),
                                (float("nan"), float("nan")), False, quality)
        raw = self.model.predict(np.asarray(features, dtype=float))
        filtered = self.smoother.update(raw, timestamp)
        return GazeEstimate(timestamp, tuple(raw), tuple(filtered), True, float(quality))
