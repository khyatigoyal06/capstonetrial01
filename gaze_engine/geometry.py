"""Camera and gaze-ray geometry helpers (pure numpy)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# MediaPipe face-mesh indices (468-point topology): eye corners and lids.
_EYES = (
    ((33, 133), (159, 145)),   # subject's right eye
    ((362, 263), (386, 374)),  # subject's left eye
)


def camera_matrix(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pinhole intrinsics for an uncalibrated webcam with the given field of view."""
    if width <= 0 or height <= 0:
        raise ValueError("Frame size must be positive.")
    if not 10.0 <= hfov_deg <= 170.0:
        raise ValueError("Horizontal field of view must be between 10 and 170 degrees.")
    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def intersect_camera_plane(origin: Sequence[float], direction: Sequence[float]) -> tuple[float, float] | None:
    """Intersect a gaze ray with the camera plane z = 0.

    For a webcam mounted on the screen bezel this plane is (nearly) the screen
    plane, so the intersection moves with true on-screen gaze and is unaffected
    by where the head is. Returns ``None`` when the ray points away from it.
    """
    o = np.asarray(origin, dtype=float).reshape(3)
    d = np.asarray(direction, dtype=float).reshape(3)
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
        return None
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    d = d / norm
    # The ray must travel toward the camera (negative z) at a usable angle;
    # near-parallel rays blow the intersection up to infinity.
    if o[2] <= 0 or d[2] > -0.2:
        return None
    s = -o[2] / d[2]
    point = o + s * d
    return float(point[0]), float(point[1])


def eye_openness(landmarks: np.ndarray) -> float | None:
    """Mean lid-gap / eye-width ratio over both eyes, from 2D face-mesh landmarks."""
    pts = np.asarray(landmarks, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 468:
        return None
    ratios = []
    for (c1, c2), (v1, v2) in _EYES:
        width = float(np.linalg.norm(pts[c2] - pts[c1]))
        if width < 1e-6:
            continue
        ratios.append(float(np.linalg.norm(pts[v2] - pts[v1])) / width)
    return float(np.mean(ratios)) if ratios else None


def approx_visual_angle_deg(error_px: float, distance_m: float | None, px_per_cm: float) -> float | None:
    """Convert an on-screen error to degrees of visual angle (approximate)."""
    if distance_m is None or distance_m <= 0 or px_per_cm <= 0:
        return None
    error_cm = error_px / px_per_cm
    return float(math.degrees(math.atan2(error_cm, distance_m * 100.0)))
