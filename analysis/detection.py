"""High-level poster element detection API."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from analysis.aoi import AOIRecord, build_aoi_record, parse_yolo_label_lines, parse_yolo_labels
from config.settings import CLASS_NAMES, DATASET_LABEL_DIR, DETECTION_CONFIDENCE, DUPLICATE_IOU
from models.detector import ModelNotFoundError, PosterDetector

# One loaded detector per (weights file, detector class). Loading YOLO weights
# takes ~0.5-1 s, so it must not happen on every request.
_DETECTOR_CACHE: dict[tuple[str, int], PosterDetector] = {}
_DETECTOR_LOCK = threading.Lock()


def get_detector(weights_path: str | Path | None = None, conf: float = DETECTION_CONFIDENCE) -> PosterDetector:
    detector_class = PosterDetector  # looked up at call time so tests can substitute it
    key = (str(weights_path or ""), id(detector_class))
    with _DETECTOR_LOCK:
        detector = _DETECTOR_CACHE.get(key)
        if detector is None:
            detector = detector_class(weights_path=weights_path, conf_threshold=conf)
            _DETECTOR_CACHE[key] = detector
        return detector


def _iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = lambda d: max(0, d["x2"] - d["x1"]) * max(0, d["y2"] - d["y1"])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def merge_duplicate_detections(detections: list[dict], iou_threshold: float = DUPLICATE_IOU) -> list[dict]:
    """Drop same-class boxes that mostly cover a higher-confidence box of that class.

    YOLO's NMS can leave several overlapping boxes for one element (e.g. three
    'logo' boxes on one logo), which would split or inflate that element's
    attention share.
    """
    kept: list[dict] = []
    for det in sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True):
        if any(k["class_id"] == det["class_id"] and _iou(k, det) >= iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def detections_to_aoi_records(
    detections: list[dict],
    image_shape: tuple[int, ...],
) -> list[AOIRecord]:
    """Convert YOLO inference output to standardized AOI records."""
    height, width = image_shape[:2]
    records: list[AOIRecord] = []

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        box_w = (x2 - x1) / max(width, 1)
        box_h = (y2 - y1) / max(height, 1)
        x_center = ((x1 + x2) / 2) / max(width, 1)
        y_center = ((y1 + y2) / 2) / max(height, 1)

        records.append(
            build_aoi_record(
                class_id=det["class_id"],
                x_center_norm=x_center,
                y_center_norm=y_center,
                width_norm=box_w,
                height_norm=box_h,
                image_width=width,
                image_height=height,
                confidence=det["confidence"],
            )
        )

    return records


def load_aois_from_labels(
    poster_name: str,
    image_shape: tuple[int, ...],
    label_dir: str | Path | None = None,
) -> list[AOIRecord]:
    """Load ground-truth AOIs from a YOLO label file matching the poster."""
    label_dir = Path(label_dir or DATASET_LABEL_DIR)
    stem = os.path.splitext(poster_name)[0]
    label_path = label_dir / f"{stem}.txt"

    if not label_path.is_file():
        raise FileNotFoundError(f"Matching label file not found: {label_path.name}")

    return parse_yolo_labels(str(label_path), image_shape)


def detect_poster_elements(
    image: np.ndarray,
    *,
    image_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    conf: float = DETECTION_CONFIDENCE,
    fallback_to_labels: bool = False,
    poster_name: str | None = None,
    label_dir: str | Path | None = None,
) -> tuple[list[AOIRecord], str]:
    """
    Detect poster elements using YOLO inference.

    Returns (aoi_records, source) where source is 'model' or 'labels'.

    If weights are missing and fallback_to_labels=True, uses dataset labels
    when poster_name is provided.
    """
    detector = get_detector(weights_path, conf)

    if detector.is_available:
        if image_path is not None:
            detections = detector.predict(image_path, conf=conf)
        else:
            detections = detector.predict_array(image, conf=conf)
        detections = merge_duplicate_detections(detections)
        return detections_to_aoi_records(detections, image.shape), "model"

    if fallback_to_labels and poster_name:
        records = load_aois_from_labels(poster_name, image.shape, label_dir)
        return records, "labels"

    raise ModelNotFoundError(
        "No trained model weights available. "
        "Train with: python training/train.py "
        "Or pass fallback_to_labels=True with a labelled dataset poster."
    )


def load_label_lines(label_dir: str | Path, poster_name: str) -> list[str]:
    """Load raw YOLO label lines for a poster (legacy helper)."""
    label_dir = Path(label_dir)
    stem = os.path.splitext(poster_name)[0]
    label_path = label_dir / f"{stem}.txt"

    if not label_path.is_file():
        raise FileNotFoundError(f"Matching label file not found: {label_path.name}")

    with open(label_path, encoding="utf-8") as file:
        return file.readlines()


def label_lines_to_aoi_records(lines: list[str], image_shape: tuple[int, ...]) -> list[AOIRecord]:
    return parse_yolo_label_lines(lines, image_shape, CLASS_NAMES, confidence=1.0)
