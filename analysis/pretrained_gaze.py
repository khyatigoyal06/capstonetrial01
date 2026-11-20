"""Local wrapper around the pretrained MPIIFaceGaze estimator.

The browser never sends frames to an external service.  A JPEG is posted to
the running FastAPI process, which derives a compact gaze/head-pose feature
vector and immediately discards the image.
"""

from __future__ import annotations

import argparse
import threading
from typing import Any

import cv2
import numpy as np


class PretrainedGazeError(RuntimeError):
    """The local pretrained gaze model could not be initialized or used."""


class PretrainedGazeEstimator:
    """Load MPIIFaceGaze once and expose a small browser-friendly feature vector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._estimator: Any | None = None

    def _load(self) -> Any:
        if self._estimator is not None:
            return self._estimator
        try:
            from omegaconf import OmegaConf
            from ptgaze.gaze_estimator import GazeEstimator
            from ptgaze.main import load_mode_config
            from ptgaze.utils import (
                check_path_all,
                download_mediapipe_face_landmarker,
                download_mpiifacegaze_model,
                expanduser_all,
            )

            args = argparse.Namespace(
                mode="mpiifacegaze",
                face_detector="mediapipe",
                device="cpu",
                image=None,
                video=None,
                camera=None,
                output_dir=None,
                ext=None,
                no_screen=True,
                debug=False,
                config=None,
            )
            config = load_mode_config(args)
            # The gaze checkpoint contains the complete backbone.  Skipping an
            # unrelated ImageNet download also keeps the runtime fully local.
            config.model.backbone.pretrained = ""
            config.gaze_estimator.checkpoint = download_mpiifacegaze_model().as_posix()
            download_mediapipe_face_landmarker()
            expanduser_all(config)
            check_path_all(config)
            OmegaConf.set_readonly(config, True)
            self._estimator = GazeEstimator(config)
            return self._estimator
        except Exception as exc:  # pragma: no cover - dependent on local model runtime
            raise PretrainedGazeError(f"Could not start the local MPIIFaceGaze model: {exc}") from exc

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise PretrainedGazeError("The camera frame could not be decoded.")

        with self._lock:
            estimator = self._load()
            # ptgaze accepts BGR input and needs the same undistortion used by its demo.
            undistorted = cv2.undistort(
                image, estimator.camera.camera_matrix, estimator.camera.dist_coefficients
            )
            faces = estimator.detect_faces(undistorted)
            if not faces:
                return {"face_detected": False}
            face = faces[0]
            estimator.estimate_gaze(undistorted, face)
            # MPIIFaceGaze predicts from the normalized full face, so its
            # gaze vector is stored on the face rather than either eye.
            gaze = np.asarray(face.gaze_vector, dtype=float).reshape(-1)
            # ptgaze stores head pose as scipy's Rotation object.
            head_rotation = np.asarray(face.head_pose_rot.as_rotvec(), dtype=float).reshape(-1)

        # Gaze direction and head rotation are stable across the calibration
        # targets.  Screen-position values add noise and overfit a small
        # per-session calibration set, so they deliberately stay out.
        features = np.concatenate((gaze, head_rotation))
        if not np.all(np.isfinite(features)):
            return {"face_detected": False}
        return {
            "face_detected": True,
            "features": [round(float(value), 7) for value in features],
        }


pretrained_gaze_estimator = PretrainedGazeEstimator()
