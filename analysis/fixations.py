"""Fixation detection and uncertainty-aware AOI attention.

Webcam gaze has an accuracy error of tens to hundreds of pixels. Counting each
raw sample as a hit on whichever rectangle contains it treats that noise as
attention, and a small CTA box becomes a coin flip. Here:

1. samples are grouped into fixations (I-DT, Salvucci & Goldberg 2000);
2. each fixation is spread as a 2D Gaussian whose sigma is the session's
   *measured* validation error, and its duration is shared between AOIs in
   proportion to how much of that Gaussian falls on each one;
3. AOIs smaller than the error are flagged as unreliable instead of being
   reported with false precision.

Overlapping AOIs use the same smallest-box-wins ownership rule as the saliency
model, so shares always sum to 100%.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from analysis.aoi import AOIRecord, get_present_aoi_labels
from config.settings import BACKGROUND_LABEL

# Work on a downscaled ownership grid; attention shares do not need full resolution.
_GRID_MAX_EDGE = 320


@dataclass(frozen=True)
class Fixation:
    x: float
    y: float
    start_ms: float
    end_ms: float
    n_samples: int

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict:
        return {**asdict(self), "duration_ms": self.duration_ms}


def detect_fixations(
    samples: Sequence[tuple[float, float, float]],
    *,
    dispersion_px: float,
    min_duration_ms: float = 100.0,
    max_gap_ms: float = 250.0,
) -> list[Fixation]:
    """I-DT fixation detection on ``(t_ms, x, y)`` samples sorted by time.

    A window is a fixation while (max x - min x) + (max y - min y) stays within
    ``dispersion_px`` and it lasts at least ``min_duration_ms``. A sampling gap
    longer than ``max_gap_ms`` (blink, lost face) always ends a window.
    """
    pts = sorted((float(t), float(x), float(y)) for t, x, y in samples if np.isfinite([t, x, y]).all())
    fixations: list[Fixation] = []
    i, n = 0, len(pts)

    def dispersion(window: list[tuple[float, float, float]]) -> float:
        xs = [p[1] for p in window]
        ys = [p[2] for p in window]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    while i < n:
        j = i
        # Grow an initial window covering min_duration_ms without crossing a gap.
        while j + 1 < n and pts[j + 1][0] - pts[j][0] <= max_gap_ms and pts[j][0] - pts[i][0] < min_duration_ms:
            j += 1
        window = pts[i : j + 1]
        if window[-1][0] - window[0][0] < min_duration_ms or dispersion(window) > dispersion_px:
            i += 1
            continue
        while j + 1 < n and pts[j + 1][0] - pts[j][0] <= max_gap_ms and dispersion(pts[i : j + 2]) <= dispersion_px:
            j += 1
        window = pts[i : j + 1]
        # The last sample's own duration is about one sampling interval.
        interval = (window[-1][0] - window[0][0]) / max(len(window) - 1, 1)
        fixations.append(
            Fixation(
                x=float(np.mean([p[1] for p in window])),
                y=float(np.mean([p[2] for p in window])),
                start_ms=window[0][0],
                end_ms=window[-1][0] + interval,
                n_samples=len(window),
            )
        )
        i = j + 1
    return fixations


def _ownership_grid(records: Sequence[AOIRecord], width: int, height: int, scale: float) -> np.ndarray:
    gw, gh = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    owner = np.full((gh, gw), -1, dtype=np.int32)
    order = sorted(range(len(records)), key=lambda i: records[i].width * records[i].height, reverse=True)
    for index in order:
        r = records[index]
        x1, y1 = int(np.floor(max(0, r.x1) * scale)), int(np.floor(max(0, r.y1) * scale))
        x2, y2 = int(np.ceil(min(width, r.x2) * scale)), int(np.ceil(min(height, r.y2) * scale))
        if x2 > x1 and y2 > y1:
            owner[y1:y2, x1:x2] = index
    return owner


def probabilistic_aoi_attention(
    fixations: Iterable[Fixation],
    records: Sequence[AOIRecord],
    image_shape: tuple[int, ...],
    sigma_px: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Share fixation time between AOIs by Gaussian overlap.

    Returns ``(attention_ms_by_label, stats)``. Labels include
    ``BACKGROUND_LABEL``. Fixations whose centre is off the poster are not
    attributed (they are counted in ``stats["off_poster_ms"]``); the Gaussian
    mass of on-poster fixations is renormalised to the poster area.
    """
    height, width = image_shape[:2]
    labels = get_present_aoi_labels(list(records))
    attention = {label: 0.0 for label in labels}
    attention[BACKGROUND_LABEL] = 0.0
    stats = {"on_poster_ms": 0.0, "off_poster_ms": 0.0, "fixations_used": 0}

    scale = min(1.0, _GRID_MAX_EDGE / max(width, height, 1))
    owner = _ownership_grid(records, width, height, scale)
    gh, gw = owner.shape
    # Pixel-centre coordinates of the grid in image space.
    xs = (np.arange(gw) + 0.5) / scale
    ys = (np.arange(gh) + 0.5) / scale
    sigma = max(float(sigma_px), 1.0)
    owner_flat = owner.ravel() + 1  # background -> bin 0
    record_labels = [r.class_name for r in records]

    for fix in fixations:
        duration = max(0.0, fix.duration_ms)
        if not (0 <= fix.x < width and 0 <= fix.y < height):
            stats["off_poster_ms"] += duration
            continue
        wx = np.exp(-0.5 * ((xs - fix.x) / sigma) ** 2)
        wy = np.exp(-0.5 * ((ys - fix.y) / sigma) ** 2)
        weights = np.outer(wy, wx).ravel()
        total = float(weights.sum())
        if total <= 0:
            continue
        mass = np.bincount(owner_flat, weights=weights, minlength=len(records) + 1) / total
        attention[BACKGROUND_LABEL] += duration * float(mass[0])
        for index, share in enumerate(mass[1:]):
            if share > 0:
                attention[record_labels[index]] += duration * float(share)
        stats["on_poster_ms"] += duration
        stats["fixations_used"] += 1
    return attention, stats


def aoi_reliability(records: Sequence[AOIRecord], sigma_px: float, factor: float = 1.5) -> list[dict]:
    """Flag AOI labels whose largest box is too small for the measured gaze error.

    A box narrower than ``factor * sigma`` on its short side cannot be told
    apart from its surroundings at this accuracy.
    """
    best: dict[str, int] = {}
    for record in records:
        side = int(min(record.width, record.height))
        best[record.class_name] = max(best.get(record.class_name, 0), side)
    threshold = factor * float(sigma_px)
    return [
        {"label": label, "min_side_px": side, "required_px": round(threshold, 1), "reliable": side >= threshold}
        for label, side in sorted(best.items())
    ]


def render_fixation_map(
    image_bgr: np.ndarray,
    fixations: Sequence[Fixation],
    sigma_px: float,
    *,
    alpha: float = 0.62,
) -> np.ndarray:
    """Heatmap of fixation time blurred by the session's measured error, with
    numbered circles showing viewing order (circle size grows with duration)."""
    import cv2

    height, width = image_bgr.shape[:2]
    heat = np.zeros((height, width), np.float32)
    on_poster = [f for f in fixations if 0 <= f.x < width and 0 <= f.y < height]
    for fix in on_poster:
        heat[int(fix.y), int(fix.x)] += max(fix.duration_ms, 1.0)
    canvas = image_bgr.copy()
    if on_poster:
        heat = cv2.GaussianBlur(heat, (0, 0), max(float(sigma_px) / 2.0, 2.0))
        peak = float(heat.max())
        if peak > 0:
            norm = np.clip(heat / peak, 0, 1)
            colours = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            weight = (norm * alpha)[..., None]
            canvas = np.clip(image_bgr * (1 - weight) + colours * weight, 0, 255).astype(np.uint8)
    scale = max(width, height) / 1000.0
    for index, fix in enumerate(fixations, start=1):
        if not (0 <= fix.x < width and 0 <= fix.y < height):
            continue
        centre = (int(fix.x), int(fix.y))
        radius = int((10 + min(fix.duration_ms, 1500) / 50) * scale)
        cv2.circle(canvas, centre, radius, (255, 255, 255), max(2, int(3 * scale)))
        cv2.circle(canvas, centre, radius, (20, 20, 20), max(1, int(1.5 * scale)))
        text = str(index)
        font_scale = 0.6 * scale
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.putText(canvas, text, (centre[0] - tw // 2, centre[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), max(2, int(2 * scale)), cv2.LINE_AA)
    return canvas
