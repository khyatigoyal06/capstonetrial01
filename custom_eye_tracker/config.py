from dataclasses import asdict, dataclass
from pathlib import Path
import platform

@dataclass(frozen=True)
class TrackerConfig:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    model_path: Path = Path("models/face_landmarker.task")
    output_dir: Path = Path("records")
    sample_window_s: float = 2.0
    settle_s: float = 0.7
    min_samples: int = 8
    fixation_radius_px: float = 70.0
    fixation_min_duration_s: float = 0.1
    smoothing_tau_s: float = 0.12

    def metadata(self) -> dict:
        return {**asdict(self), "model_path": str(self.model_path), "output_dir": str(self.output_dir),
                "python": platform.python_version(), "platform": platform.platform()}
