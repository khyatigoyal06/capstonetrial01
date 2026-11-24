"""Per-frame result produced by a gaze backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FrameObservation:
    """What the backend measured in one camera frame.

    ``point_m`` is where the gaze ray meets the camera plane (z = 0), in metres
    in OpenCV camera coordinates (x right, y down). It already accounts for
    head position and rotation, so it is the feature the per-user mapping uses.
    """

    face_detected: bool
    point_m: tuple[float, float] | None = None
    gaze_pitch_yaw: tuple[float, float] | None = None
    head_rotvec: tuple[float, float, float] | None = None
    face_distance_m: float | None = None
    eye_openness: float | None = None
    reason: str = "ok"
    # For the positioning guide: face box in normalised image coords (x1, y1, x2, y2)
    # and head angles in degrees (0, 0 = facing the camera).
    face_box: tuple[float, float, float, float] | None = None
    head_pitch_deg: float | None = None
    head_yaw_deg: float | None = None

    @property
    def usable(self) -> bool:
        """True when geometry is present; blink checks happen in the session."""
        return self.face_detected and self.point_m is not None and self.reason == "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
