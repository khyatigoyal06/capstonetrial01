"""Tunable constants for the gaze engine, overridable through environment variables."""

from __future__ import annotations

import os


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# Which ptgaze model to use. ETH-XGaze generalises across people and head poses
# better than MPIIFaceGaze because it was trained on far larger pose/gaze ranges.
GAZE_ENGINE_MODE = os.getenv("GAZE_ENGINE_MODE", "eth-xgaze")
GAZE_ENGINE_DEVICE = os.getenv("GAZE_ENGINE_DEVICE", "cpu")

# Horizontal field of view used to build the camera matrix when the webcam has
# not been calibrated. Typical laptop webcams are 60-80 degrees. A wrong value
# mostly becomes a scale error, which the per-user mapping absorbs.
CAMERA_HFOV_DEG = _float("GAZE_CAMERA_HFOV_DEG", 70.0)

# Calibration capture: frames in the first SETTLE seconds after a target appears
# are the saccade + reaction time and are ignored.
TARGET_SETTLE_S = _float("GAZE_TARGET_SETTLE_S", 0.6)
MIN_TARGET_SAMPLES = int(_float("GAZE_MIN_TARGET_SAMPLES", 5))
# Robust spread of the camera-plane gaze point (metres) above which the
# participant was not holding a fixation. 0.035 m is about 3.3 deg at 60 cm.
MAX_TARGET_DISPERSION_M = _float("GAZE_MAX_TARGET_DISPERSION_M", 0.035)

# Blink rejection relative to this participant's own open-eye baseline.
BLINK_RATIO = _float("GAZE_BLINK_RATIO", 0.55)
ABSOLUTE_MIN_OPENNESS = _float("GAZE_ABSOLUTE_MIN_OPENNESS", 0.08)

# Plausible face distance in metres; outside this the pose fit is unreliable.
MIN_FACE_DISTANCE_M = 0.25
MAX_FACE_DISTANCE_M = 1.6
# Warn when the participant moves this much closer/further than at calibration.
DISTANCE_WARNING_RATIO = _float("GAZE_DISTANCE_WARNING_RATIO", 0.2)

# Session quality bands, judged on held-out validation targets (screen px).
GOOD_MEDIAN_PX = _float("GAZE_GOOD_MEDIAN_PX", 100.0)
GOOD_P90_PX = _float("GAZE_GOOD_P90_PX", 180.0)
FAIR_MEDIAN_PX = _float("GAZE_FAIR_MEDIAN_PX", 180.0)
FAIR_P90_PX = _float("GAZE_FAIR_P90_PX", 320.0)

# Drift checks during recording. Offsets below the limit are corrected; above
# it the participant is asked to recalibrate.
DRIFT_CORRECT_LIMIT_PX = _float("GAZE_DRIFT_CORRECT_LIMIT_PX", 160.0)
DRIFT_BLEND = _float("GAZE_DRIFT_BLEND", 0.7)

# One-Euro filter (Casiez et al., CHI 2012), screen-pixel units.
FILTER_MIN_CUTOFF_HZ = _float("GAZE_FILTER_MIN_CUTOFF_HZ", 0.6)
FILTER_BETA = _float("GAZE_FILTER_BETA", 0.003)
FILTER_D_CUTOFF_HZ = _float("GAZE_FILTER_D_CUTOFF_HZ", 1.0)

# Used only to report an approximate angular error next to pixel error.
ASSUMED_CSS_PX_PER_CM = _float("GAZE_ASSUMED_CSS_PX_PER_CM", 96 / 2.54)
