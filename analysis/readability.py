"""Readability of text elements (Headline, CTA, Price).

A designer acts on two things the attention model cannot see: whether text
stands out from what is behind it, and whether it is big enough to read at a
glance. For each text box we estimate:

* **contrast ratio** - the box is clustered into its two dominant colours
  (k-means, k=2) and the WCAG 2.x contrast ratio between them is computed.
  WCAG asks for 4.5:1 for normal text and 3:1 for large text;
* **size** - box height relative to the poster height (CTA/Price also by area).

These are estimates: a box can contain a photo, gradients or several colours,
so treat them as signals, not measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

TEXT_LABELS = ("Headline", "CTA", "Price")
# Minimum box height as a share of poster height to count as comfortably readable.
MIN_HEIGHT_SHARE = {"Headline": 0.07, "CTA": 0.035, "Price": 0.035}
CONTRAST_FLOOR, CONTRAST_TARGET = 1.5, 4.5


@dataclass(frozen=True)
class TextReadability:
    label: str
    contrast_ratio: float
    height_share: float
    contrast_score: float
    size_score: float
    score: float

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(self).items()}


def _relative_luminance(bgr: np.ndarray) -> float:
    rgb = np.asarray(bgr, dtype=float)[::-1] / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])


def contrast_ratio(colour_a: Sequence[float], colour_b: Sequence[float]) -> float:
    """WCAG contrast ratio between two BGR colours (1.0 - 21.0)."""
    la, lb = _relative_luminance(np.asarray(colour_a)), _relative_luminance(np.asarray(colour_b))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def dominant_contrast(crop_bgr: np.ndarray) -> float:
    """Contrast ratio between the two dominant colours of an image region."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 1.0
    h, w = crop_bgr.shape[:2]
    scale = min(1.0, 64.0 / max(h, w))
    small = cv2.resize(crop_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 4 or float(pixels.std(axis=0).max()) < 1.0:
        return 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    cv2.setRNGSeed(7)
    _compactness, _labels, centres = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return float(contrast_ratio(centres[0], centres[1]))


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def assess_text_elements(image_bgr: np.ndarray, boxes: Iterable[tuple[str, int, int, int, int]]) -> list[TextReadability]:
    """Readability for every text box; the best-scoring box per label is kept."""
    height, width = image_bgr.shape[:2]
    best: dict[str, TextReadability] = {}
    for label, x1, y1, x2, y2 in boxes:
        if label not in TEXT_LABELS:
            continue
        x1, x2 = max(0, int(x1)), min(width, int(x2))
        y1, y2 = max(0, int(y1)), min(height, int(y2))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        ratio = dominant_contrast(image_bgr[y1:y2, x1:x2])
        height_share = (y2 - y1) / max(height, 1)
        contrast_score = _clamp((ratio - CONTRAST_FLOOR) / (CONTRAST_TARGET - CONTRAST_FLOOR) * 100)
        size_score = _clamp(height_share / MIN_HEIGHT_SHARE[label] * 100)
        item = TextReadability(label, ratio, height_share, contrast_score, size_score, 0.65 * contrast_score + 0.35 * size_score)
        if label not in best or item.score > best[label].score:
            best[label] = item
    return [best[label] for label in TEXT_LABELS if label in best]
