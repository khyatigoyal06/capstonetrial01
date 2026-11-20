"""Normalized geometry from MediaPipe Face Landmarker (468/478 topology).

Iris coordinates are coarse image landmarks, not precise pupil measurements.
Eye corner and iris index pairs follow the standard Face Mesh topology.
"""
from math import hypot
import numpy as np

EYES = {"left": {"corners": (362, 263), "vertical": (386, 374), "iris": (473, 474, 475, 476, 477)},
        "right": {"corners": (33, 133), "vertical": (159, 145), "iris": (468, 469, 470, 471, 472)}}
FEATURE_NAMES = ("left_iris_x", "left_iris_y", "left_eye_height", "right_iris_x", "right_iris_y", "right_eye_height")

def extract_features(landmarks):
    """Return fixed-length normalized vector and quality dict; invalid eyes are NaN."""
    values=[]; quality={"face": bool(landmarks), "left_eye": False, "right_eye": False, "blink": False,
                        "landmark_quality": 0.0, "reason": "no_face"}
    if not landmarks:
        return np.full(6, np.nan, dtype=float), quality
    lm=landmarks
    for name,spec in EYES.items():
        if len(lm) <= max(spec["iris"]):
            values.extend((np.nan,np.nan,np.nan)); continue
        a,b=(lm[i] for i in spec["corners"]); width=hypot(b.x-a.x,b.y-a.y)
        if width < 1e-5:
            values.extend((np.nan,np.nan,np.nan)); continue
        ix=float(np.mean([lm[i].x for i in spec["iris"]])); iy=float(np.mean([lm[i].y for i in spec["iris"]]))
        v1,v2=(lm[i] for i in spec["vertical"]); openness=hypot(v2.x-v1.x,v2.y-v1.y)/width
        values.extend(((ix-a.x)/(b.x-a.x), (iy-(a.y+b.y)/2)/width, openness))
        quality[f"{name}_eye"] = True
    visible=sum(quality[k] for k in ("left_eye","right_eye"));
    # Simplified eye-aspect heuristic; not a clinical or universal blink measure.
    openness=[values[i] for i in (2,5) if np.isfinite(values[i])]
    blink=bool(openness) and max(openness)<0.09
    reason="no_face" if not quality["face"] else "eyes_unavailable" if not visible else "blink" if blink else "ok"
    quality.update(landmark_quality=visible/2, blink=blink, reason=reason)
    return np.asarray(values,float), quality

def valid_feature_vector(features, quality):
    return bool(quality.get("face") and not quality.get("blink") and np.isfinite(features).sum() >= 3)
