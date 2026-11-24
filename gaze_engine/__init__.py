"""Calibrated webcam gaze engine.

Pipeline (one frame):

    frame -> face landmarks + head pose (solvePnP on a 3D face model)
          -> data normalisation (warp face to a virtual camera; cancels roll/distance)
          -> appearance CNN (ETH-XGaze ResNet-18) -> 3D gaze direction
          -> ray from the 3D face centre, intersected with the camera plane (z = 0)
          -> per-user mapping fitted at calibration -> screen pixels
          -> One-Euro filter

Because head pose and position enter through geometry, moving the head after
calibration no longer looks like an eye movement, which was the main failure of
the previous iris-offset -> pixels linear model.

The heavy model lives behind :class:`gaze_engine.backend.GazeBackend` so the
calibration, filtering and session logic can be tested without PyTorch.
"""

from gaze_engine.observation import FrameObservation

__all__ = ["FrameObservation"]
