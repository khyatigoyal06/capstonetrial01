from pathlib import Path

class FaceLandmarker:
    """Lazy wrapper for MediaPipe Tasks VIDEO mode; images remain in process."""
    def __init__(self, model_path: Path):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is missing. Use the documented Python 3.11/3.12 virtual environment and install requirements.txt.") from exc
        if not model_path.is_file():
            raise RuntimeError(f"Face Landmarker model not found: {model_path}. Download face_landmarker.task as described in README.md.")
        self.mp=mp; self.last_timestamp=-1
        options=mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1, output_facial_transformation_matrixes=False)
        self.task=mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, bgr, timestamp_ms):
        timestamp_ms=max(int(timestamp_ms), self.last_timestamp+1); self.last_timestamp=timestamp_ms
        rgb=self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=bgr[:,:,::-1].copy())
        result=self.task.detect_for_video(rgb,timestamp_ms)
        return result.face_landmarks[0] if result.face_landmarks else [], result

    def close(self): self.task.close()
