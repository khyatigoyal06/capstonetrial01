"""Analyze CSV exports recorded with the calibrated gaze engine.

Engine exports carry the session's own validation error and the on-screen
poster scale, so attention can be computed from fixations with an uncertainty
that matches *this* session instead of a hard-coded guess. Older CSVs (legacy
iris tracker, Tobii) keep using :class:`analysis.gaze.GazeAnalyzer`.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any, Sequence

from analysis.aoi import AOIRecord
from analysis.fixations import Fixation, aoi_reliability, detect_fixations, probabilistic_aoi_attention

ENGINE_MODEL = "engine"
REQUIRED_COLUMNS = {"timestamp_ms", "gaze_x", "gaze_y", "gaze_model", "session_median_error_px", "poster_display_scale"}

# I-DT dispersion threshold in *screen* pixels: ~1.5-2 deg at a typical laptop
# distance, wide enough for webcam jitter, narrow enough to split saccades.
FIXATION_DISPERSION_SCREEN_PX = 80.0
FIXATION_MIN_DURATION_MS = 100.0
# Used when a CSV row is missing its own error estimate.
FALLBACK_SIGMA_SCREEN_PX = 120.0


@dataclass
class EngineSession:
    samples: list[tuple[float, float, float]]  # (t_ms, x, y) in the CSV's poster coordinates
    poster_width: float | None
    poster_height: float | None
    median_error_screen_px: float
    p90_error_screen_px: float | None
    display_scale: float  # rendered poster CSS px per poster image px
    quality: str | None
    rows_read: int
    rows_used: int


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # drop NaN


def load_engine_session(gaze_bytes: bytes, *, delimiter: str = ",") -> EngineSession | None:
    """Return an :class:`EngineSession` if the export came from the gaze engine."""
    try:
        text = gaze_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
        return None

    samples: list[tuple[float, float, float]] = []
    errors, p90s, scales = [], [], []
    width = height = None
    quality = None
    rows_read = 0
    for row in reader:
        rows_read += 1
        if (row.get("gaze_model") or "").strip() != ENGINE_MODEL:
            continue
        if (row.get("gaze_valid") or "1").strip().lower() in ("0", "false", "no"):
            continue
        t, x, y = _float(row.get("timestamp_ms")), _float(row.get("gaze_x")), _float(row.get("gaze_y"))
        if t is None or x is None or y is None:
            continue
        samples.append((t, x, y))
        for target, key in ((errors, "session_median_error_px"), (p90s, "session_p90_error_px"), (scales, "poster_display_scale")):
            value = _float(row.get(key))
            if value is not None and value > 0:
                target.append(value)
        width = width or _float(row.get("poster_width"))
        height = height or _float(row.get("poster_height"))
        quality = quality or (row.get("session_quality") or None)
    if not samples:
        return None

    def median(values: Sequence[float], default: float | None) -> float | None:
        if not values:
            return default
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    return EngineSession(
        samples=samples,
        poster_width=width,
        poster_height=height,
        median_error_screen_px=median(errors, FALLBACK_SIGMA_SCREEN_PX) or FALLBACK_SIGMA_SCREEN_PX,
        p90_error_screen_px=median(p90s, None),
        display_scale=median(scales, 1.0) or 1.0,
        quality=quality,
        rows_read=rows_read,
        rows_used=len(samples),
    )


def analyze_engine_session(
    session: EngineSession,
    records: Sequence[AOIRecord],
    image_shape: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, Any], list[tuple[int, int, float]]]:
    """Return ``(attention_ms, stats, points)`` with points in uploaded-image pixels."""
    height, width = image_shape[:2]
    sx = width / session.poster_width if session.poster_width else 1.0
    sy = height / session.poster_height if session.poster_height else 1.0
    # Screen px -> poster px -> uploaded-image px.
    screen_to_image = (sx + sy) / 2.0 / max(session.display_scale, 1e-6)

    points = [(t, x * sx, y * sy) for t, x, y in session.samples]
    sigma_px = session.median_error_screen_px * screen_to_image
    fixations = detect_fixations(
        points,
        dispersion_px=FIXATION_DISPERSION_SCREEN_PX * screen_to_image,
        min_duration_ms=FIXATION_MIN_DURATION_MS,
    )
    attention, attention_stats = probabilistic_aoi_attention(fixations, records, image_shape, sigma_px)
    reliability = aoi_reliability(records, sigma_px)
    stats = {
        "rows_read": session.rows_read,
        "rows_used": session.rows_used,
        "events_used": attention_stats["fixations_used"],
        "rows_quality_rejected": session.rows_read - session.rows_used,
        "fixations": len(fixations),
        "fixation_time_ms": round(sum(f.duration_ms for f in fixations), 1),
        "on_poster_ms": round(attention_stats["on_poster_ms"], 1),
        "off_poster_ms": round(attention_stats["off_poster_ms"], 1),
        "mode": "engine_fixations",
    }
    quality = {
        "session_quality": session.quality,
        "validation_median_error_screen_px": round(session.median_error_screen_px, 1),
        "validation_p90_error_screen_px": None if session.p90_error_screen_px is None else round(session.p90_error_screen_px, 1),
        "sigma_image_px": round(sigma_px, 1),
        "aoi_reliability": reliability,
        "unreliable_aois": [item["label"] for item in reliability if not item["reliable"]],
    }
    stats["quality"] = quality
    stats["fixation_list"] = [f.to_dict() for f in fixations]
    point_triples = [(int(round(x)), int(round(y)), t) for t, x, y in points if 0 <= x < width and 0 <= y < height]
    return attention, stats, point_triples


def fixations_from_stats(stats: dict[str, Any]) -> list[Fixation]:
    return [
        Fixation(x=f["x"], y=f["y"], start_ms=f["start_ms"], end_ms=f["end_ms"], n_samples=f["n_samples"])
        for f in stats.get("fixation_list") or []
    ]
