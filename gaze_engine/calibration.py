"""Per-user mapping from camera-plane gaze points to screen pixels.

The CNN's gaze direction is already head-pose aware, so what remains per person
is small and smooth: the offset between the eye's optical and visual axes
(kappa, ~5 deg and person-specific), the unknown camera-to-screen placement and
the field-of-view guess. An affine map captures most of that; a quadratic map is
tried too and kept only when leave-one-target-out error says it generalises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from gaze_engine import config

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
_MIN_POINTS = {"affine": 4, "quadratic": 9}


@dataclass(frozen=True)
class TargetCapture:
    """Robust summary of the frames recorded while one target was shown."""

    target_id: str
    target_px: tuple[float, float]
    feature: tuple[float, float]
    n_samples: int
    dispersion_m: float
    face_distance_m: float | None = None


def summarise_target(
    samples: Sequence[tuple[float, tuple[float, float], float | None]],
    shown_at_ms: float,
    *,
    settle_s: float = config.TARGET_SETTLE_S,
    min_samples: int = config.MIN_TARGET_SAMPLES,
    max_dispersion_m: float = config.MAX_TARGET_DISPERSION_M,
) -> tuple[tuple[tuple[float, float], int, float, float | None] | None, str]:
    """Summarise ``(t_ms, (X, Y), distance_m)`` samples captured for one target.

    Frames in the first ``settle_s`` after the target appeared are the saccade
    and reaction time, so they are dropped *from the start* of the window. The
    remaining points are trimmed with a median-absolute-deviation rule, and the
    target is rejected if the participant clearly was not holding still.
    Returns ``((feature, n, dispersion, distance), "ok")`` or ``(None, reason)``.
    """
    settled = [s for s in samples if s[0] - shown_at_ms >= settle_s * 1000.0]
    if len(settled) < min_samples:
        return None, "too_few_samples"
    pts = np.asarray([s[1] for s in settled], dtype=float)
    centre = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - centre, axis=1)
    mad = float(np.median(dist)) or 1e-6
    keep = dist <= max(3.0 * mad, 0.004)
    if keep.sum() < min_samples:
        return None, "unstable_fixation"
    kept = pts[keep]
    centre = np.median(kept, axis=0)
    dispersion = float(np.median(np.linalg.norm(kept - centre, axis=1)))
    if dispersion > max_dispersion_m:
        return None, "unstable_fixation"
    distances = [s[2] for s, k in zip(settled, keep) if k and s[2] is not None]
    distance = float(np.median(distances)) if distances else None
    return ((float(centre[0]), float(centre[1])), int(keep.sum()), dispersion, distance), "ok"


def _design(features: np.ndarray, kind: str) -> np.ndarray:
    x, y = features[:, 0], features[:, 1]
    columns = [np.ones_like(x), x, y]
    if kind == "quadratic":
        columns += [x * x, x * y, y * y]
    return np.stack(columns, axis=1)


@dataclass
class GazeMapping:
    kind: str
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray, kind: str, alpha: float) -> "GazeMapping":
        features = np.asarray(features, float)
        targets = np.asarray(targets, float)
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        design = _design((features - mean) / scale, kind)
        penalty = np.eye(design.shape[1]) * alpha * len(features)
        penalty[0, 0] = 0.0  # never shrink the intercept
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        return cls(kind=kind, alpha=alpha, mean=mean, scale=scale, coef=coef)

    def predict(self, features: Iterable[Sequence[float]] | Sequence[float]) -> np.ndarray:
        arr = np.asarray(features, float)
        single = arr.ndim == 1
        arr = np.atleast_2d(arr)
        out = _design((arr - self.mean) / self.scale, self.kind) @ self.coef + self.offset
        return out[0] if single else out


def _loo_errors(features: np.ndarray, targets: np.ndarray, kind: str, alpha: float) -> np.ndarray:
    errors = []
    for i in range(len(features)):
        mask = np.arange(len(features)) != i
        model = GazeMapping.fit(features[mask], targets[mask], kind, alpha)
        errors.append(float(np.linalg.norm(model.predict(features[i]) - targets[i])))
    return np.asarray(errors)


def fit_mapping(captures: Sequence[TargetCapture]) -> tuple[GazeMapping, dict]:
    """Choose model form and ridge strength by leave-one-target-out error, then fit.

    One target whose residual is far worse than the rest (the participant looked
    away) is dropped and the selection repeated.
    """
    if len(captures) < _MIN_POINTS["affine"]:
        raise ValueError(f"Need at least {_MIN_POINTS['affine']} accepted calibration targets.")
    captures = list(captures)
    dropped: list[str] = []

    for _ in range(2):
        features = np.asarray([c.feature for c in captures], float)
        targets = np.asarray([c.target_px for c in captures], float)
        scored = []
        for kind, minimum in _MIN_POINTS.items():
            if len(captures) <= minimum:
                continue
            for alpha in ALPHAS:
                errs = _loo_errors(features, targets, kind, alpha)
                scored.append((float(np.median(errs)), float(np.percentile(errs, 90)), kind, alpha, errs))
        if not scored:
            raise ValueError("Too few calibration targets to cross-validate.")
        median, p90, kind, alpha, errs = min(scored, key=lambda item: (item[0], item[1]))
        worst = int(np.argmax(errs))
        if (
            not dropped
            and len(captures) - 1 > _MIN_POINTS["affine"]
            and errs[worst] > max(3.0 * median, 150.0)
        ):
            dropped.append(captures[worst].target_id)
            captures.pop(worst)
            continue
        break

    mapping = GazeMapping.fit(features, targets, kind, alpha)
    report = {
        "kind": kind,
        "alpha": alpha,
        "points_used": len(captures),
        "dropped_targets": dropped,
        "cv_median_px": round(median, 1),
        "cv_p90_px": round(p90, 1),
        # Leave-one-out error per target: lets the client re-measure the worst ones.
        "residuals": {c.target_id: round(float(e), 1) for c, e in zip(captures, errs)},
    }
    return mapping, report


def summarise_errors(errors: Sequence[float]) -> dict:
    errs = np.asarray(list(errors), float)
    if errs.size == 0:
        return {"n": 0, "median_px": None, "p90_px": None, "mean_px": None, "max_px": None}
    return {
        "n": int(errs.size),
        "median_px": round(float(np.median(errs)), 1),
        "p90_px": round(float(np.percentile(errs, 90)), 1),
        "mean_px": round(float(np.mean(errs)), 1),
        "max_px": round(float(np.max(errs)), 1),
    }


def quality_band(median_px: float | None, p90_px: float | None) -> str:
    if median_px is None or p90_px is None:
        return "poor"
    if median_px <= config.GOOD_MEDIAN_PX and p90_px <= config.GOOD_P90_PX:
        return "good"
    if median_px <= config.FAIR_MEDIAN_PX and p90_px <= config.FAIR_P90_PX:
        return "fair"
    return "poor"
