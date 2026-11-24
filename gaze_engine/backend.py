"""Appearance-based gaze backend built on ptgaze (ETH-XGaze / MPIIFaceGaze).

ptgaze already implements the parts that are easy to get subtly wrong: the
MediaPipe Face Landmarker, solvePnP head pose against a 3D face model, the
Zhang et al. (ETRA 2018) data normalisation and gaze de-normalisation. This
module adds what the old code was missing:

* real camera intrinsics for the actual frame size (the old path undistorted
  every frame with ptgaze's 640x480 *sample* camera, which skews head pose);
* the gaze-ray / camera-plane intersection, which makes the calibration
  feature head-pose aware;
* eye openness for per-user blink rejection.
"""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from gaze_engine import config
from gaze_engine.geometry import camera_matrix, eye_openness, intersect_camera_plane
from gaze_engine.observation import FrameObservation


class GazeBackendError(RuntimeError):
    """The gaze model could not be loaded or run."""


class GazeBackend(Protocol):
    def process(self, image_bgr: np.ndarray) -> FrameObservation: ...


def backend_availability() -> tuple[bool, str]:
    """Cheap check (no model load) for whether the engine's packages are installed."""
    missing = [name for name in ("torch", "ptgaze", "mediapipe", "timm") if importlib.util.find_spec(name) is None]
    if missing:
        return False, (
            "Missing packages: " + ", ".join(missing)
            + ". Install them with: pip install -r requirements.txt"
        )
    return True, "ok"


class PtgazeBackend:
    """Thread-safe wrapper that keeps one ptgaze estimator per frame size."""

    def __init__(
        self,
        mode: str = config.GAZE_ENGINE_MODE,
        device: str = config.GAZE_ENGINE_DEVICE,
        hfov_deg: float = config.CAMERA_HFOV_DEG,
    ) -> None:
        if mode not in ("eth-xgaze", "mpiifacegaze"):
            raise GazeBackendError("GAZE_ENGINE_MODE must be 'eth-xgaze' or 'mpiifacegaze'.")
        self.mode = mode
        self.device = device
        self.hfov_deg = hfov_deg
        self._lock = threading.Lock()
        self._estimator: Any | None = None
        self._frame_size: tuple[int, int] | None = None
        self._tmpdir = tempfile.TemporaryDirectory(prefix="gaze_engine_")

    # -- model loading -------------------------------------------------
    def _write_camera_params(self, width: int, height: int) -> Path:
        matrix = camera_matrix(width, height, self.hfov_deg)
        path = Path(self._tmpdir.name) / f"camera_{width}x{height}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "image_width": int(width),
                    "image_height": int(height),
                    "camera_matrix": {"rows": 3, "cols": 3, "data": matrix.reshape(-1).tolist()},
                    "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0.0] * 5},
                }
            ),
            encoding="utf-8",
        )
        return path

    def _load(self, width: int, height: int) -> Any:
        if self._estimator is not None and self._frame_size == (width, height):
            return self._estimator
        ok, reason = backend_availability()
        if not ok:
            raise GazeBackendError(reason)
        try:
            from omegaconf import OmegaConf
            from ptgaze.gaze_estimator import GazeEstimator
            from ptgaze.main import load_mode_config
            from ptgaze.utils import (
                check_path_all,
                download_ethxgaze_model,
                download_mediapipe_face_landmarker,
                download_mpiifacegaze_model,
                expanduser_all,
            )

            args = argparse.Namespace(
                mode=self.mode,
                face_detector="mediapipe",
                device=self.device,
                image=None,
                video=None,
                camera=str(self._write_camera_params(width, height)),
                output_dir=None,
                ext=None,
                no_screen=True,
                debug=False,
                config=None,
            )
            cfg = load_mode_config(args)
            cfg.face_detector.mediapipe_max_num_faces = 1
            cfg.gaze_estimator.checkpoint = (
                download_ethxgaze_model() if self.mode == "eth-xgaze" else download_mpiifacegaze_model()
            ).as_posix()
            if self.mode == "mpiifacegaze":
                # The checkpoint holds the full backbone; skip an ImageNet download.
                cfg.model.backbone.pretrained = ""
            download_mediapipe_face_landmarker()
            expanduser_all(cfg)
            check_path_all(cfg)
            OmegaConf.set_readonly(cfg, True)
            estimator = GazeEstimator(cfg)
        except GazeBackendError:
            raise
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            raise GazeBackendError(f"Could not start the gaze model ({self.mode}): {exc}") from exc

        if self._estimator is not None:
            try:
                self._estimator.close()
            except Exception:  # pragma: no cover
                pass
        self._estimator = estimator
        self._frame_size = (width, height)
        return estimator

    def warm_up(self, width: int = 1280, height: int = 720) -> None:
        with self._lock:
            self._load(width, height)

    # -- inference -----------------------------------------------------
    def process(self, image_bgr: np.ndarray) -> FrameObservation:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.size == 0:
            return FrameObservation(face_detected=False, reason="bad_frame")
        height, width = image_bgr.shape[:2]
        with self._lock:
            estimator = self._load(width, height)
            faces = estimator.detect_faces(image_bgr)
            if not faces:
                return FrameObservation(face_detected=False, reason="no_face")
            # Several people in view: the largest face is the participant.
            face = max(faces, key=lambda f: float(np.prod(np.ptp(np.asarray(f.bbox, float), axis=0))))
            try:
                estimator.estimate_gaze(image_bgr, face)
            except Exception:  # pragma: no cover - solvePnP / warp failures
                return FrameObservation(face_detected=True, reason="pose_failed")
            gaze = np.asarray(face.gaze_vector, dtype=float).reshape(-1)
            centre = np.asarray(face.center, dtype=float).reshape(-1)
            rotvec = np.asarray(face.head_pose_rot.as_rotvec(), dtype=float).reshape(-1)
            angles = np.asarray(face.normalized_gaze_angles, dtype=float).reshape(-1)
            openness = eye_openness(face.landmarks)
            # Same convention as ptgaze's demo: (pitch, yaw, roll) with 0 = facing the camera.
            pitch, yaw, _roll = face.change_coordinate_system(face.head_pose_rot.as_euler("XYZ", degrees=True))
            box = np.asarray(face.bbox, dtype=float).reshape(2, 2)
            face_box = (
                float(box[0, 0] / width), float(box[0, 1] / height),
                float(box[1, 0] / width), float(box[1, 1] / height),
            )

        return observation_from_geometry(
            centre, gaze, rotvec, angles, openness,
            face_box=face_box, head_angles=(float(pitch), float(yaw)),
        )


def observation_from_geometry(
    centre: np.ndarray,
    gaze: np.ndarray,
    rotvec: np.ndarray,
    angles: np.ndarray,
    openness: float | None,
    *,
    face_box: tuple[float, float, float, float] | None = None,
    head_angles: tuple[float, float] | None = None,
) -> FrameObservation:
    """Turn ptgaze's 3D outputs into a :class:`FrameObservation` (pure; testable)."""
    pose = {
        "face_box": face_box,
        "head_pitch_deg": head_angles[0] if head_angles else None,
        "head_yaw_deg": head_angles[1] if head_angles else None,
    }
    if not (np.all(np.isfinite(centre)) and np.all(np.isfinite(gaze))):
        return FrameObservation(face_detected=True, reason="pose_failed", face_box=face_box)
    distance = float(np.linalg.norm(centre))
    if not config.MIN_FACE_DISTANCE_M <= distance <= config.MAX_FACE_DISTANCE_M:
        return FrameObservation(face_detected=True, face_distance_m=distance, eye_openness=openness, reason="bad_distance", **pose)
    point = intersect_camera_plane(centre, gaze)
    if point is None:
        return FrameObservation(face_detected=True, face_distance_m=distance, eye_openness=openness, reason="looking_away", **pose)
    return FrameObservation(
        face_detected=True,
        point_m=point,
        gaze_pitch_yaw=(float(angles[0]), float(angles[1])) if angles.size >= 2 else None,
        head_rotvec=tuple(float(v) for v in rotvec[:3]),
        face_distance_m=distance,
        eye_openness=openness,
        reason="ok",
        **pose,
    )


_backend: PtgazeBackend | None = None
_backend_lock = threading.Lock()


def get_backend() -> PtgazeBackend:
    """Process-wide backend singleton (the model is loaded on first frame)."""
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = PtgazeBackend()
        return _backend
