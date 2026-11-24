"""Poster Effectiveness Score (PES), version 2.

PES combines five 0-100 components (weights in config/scoring.yaml):

* **Attention Fit** - each detected element's attention share against a
  healthy band, weighted by how much that element matters.
* **Visual Hierarchy** - pairwise agreement between the observed attention
  order and the expected order (Product, then Headline/CTA, then Price, logo).
* **Focus** - penalises attention lost to the background and one element
  swallowing everything.
* **Completeness** - which key ad elements are present at all. A missing CTA
  costs its share here instead of zeroing a whole component, as v1 did.
* **Readability** - WCAG contrast and size of text elements (needs the image).

Components that cannot be measured are excluded and the other weights rescaled.
The same function scores estimated attention (saliency) and measured attention
(gaze), so the two are directly comparable.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from analysis.aoi import AOIRecord
from config.settings import BACKGROUND_LABEL, CORE_AOI_ORDER, PROJECT_ROOT

SCORING_CONFIG_PATH = PROJECT_ROOT / "config" / "scoring.yaml"

_DEFAULTS: dict[str, Any] = {
    "pes_version": 2,
    "pes_weights": {"Attention Fit": 30, "Visual Hierarchy": 20, "Focus": 15, "Completeness": 15, "Readability": 20},
    "ideal_attention_ranges": {"Product": (30, 45), "Headline": (15, 30), "CTA": (10, 20), "Price": (5, 15), "logo": (3, 10)},
    "element_importance": {"Product": 0.35, "Headline": 0.25, "CTA": 0.25, "Price": 0.10, "logo": 0.05},
    "completeness_points": {"Product": 35, "Headline": 25, "CTA": 25, "logo": 10, "Price": 5},
    "expected_rank": {"Product": 1, "Headline": 2, "CTA": 2, "Price": 3, "logo": 4},
    "pes_categories": {"excellent": 80, "good": 65, "average": 50},
}


def _load_scoring_config() -> dict[str, Any]:
    try:
        import yaml

        with open(SCORING_CONFIG_PATH, encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
    except Exception:
        loaded = {}
    config = {key: loaded.get(key) or value for key, value in _DEFAULTS.items()}
    config["pes_weights"] = {str(k): float(v) for k, v in config["pes_weights"].items()}
    config["ideal_attention_ranges"] = {str(k): (float(v[0]), float(v[1])) for k, v in config["ideal_attention_ranges"].items()}
    config["element_importance"] = {str(k): float(v) for k, v in config["element_importance"].items()}
    config["completeness_points"] = {str(k): float(v) for k, v in config["completeness_points"].items()}
    config["expected_rank"] = {str(k): float(v) for k, v in config["expected_rank"].items()}
    config["pes_categories"] = {str(k): float(v) for k, v in config["pes_categories"].items()}
    return config


_CONFIG = _load_scoring_config()
PES_VERSION: int = int(_CONFIG["pes_version"])
PES_WEIGHTS: dict[str, float] = _CONFIG["pes_weights"]
IDEAL_ATTENTION_RANGES: dict[str, tuple[float, float]] = _CONFIG["ideal_attention_ranges"]
ELEMENT_IMPORTANCE: dict[str, float] = _CONFIG["element_importance"]
COMPLETENESS_POINTS: dict[str, float] = _CONFIG["completeness_points"]
EXPECTED_RANK: dict[str, float] = _CONFIG["expected_rank"]
PES_CATEGORIES: dict[str, float] = _CONFIG["pes_categories"]


def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    return max(minimum, min(maximum, value))


def _boxes(items: Iterable[Any]) -> list[tuple[str, int, int, int, int]]:
    """Normalise AOIRecords or legacy (label, x1, y1, x2, y2) tuples."""
    out = []
    for item in items:
        if isinstance(item, AOIRecord):
            out.append((item.class_name, item.x1, item.y1, item.x2, item.y2))
        else:
            label, x1, y1, x2, y2 = item[:5]
            out.append((str(label), int(x1), int(y1), int(x2), int(y2)))
    return out


def _present_labels(boxes: list[tuple[str, int, int, int, int]]) -> list[str]:
    labels = {box[0] for box in boxes}
    return [label for label in CORE_AOI_ORDER if label in labels] + sorted(labels - set(CORE_AOI_ORDER))


def score_for_ideal_range(value: float, low: float, high: float) -> float:
    """100 inside [low, high]; falls linearly to 0 at 0% (below) or 100% (above)."""
    if low <= value <= high:
        return 100.0
    if value < low:
        return clamp(value / low * 100) if low else 100.0
    remaining = 100 - high
    return 0.0 if remaining <= 0 else clamp(100 - (value - high) / remaining * 100)


def attention_fit(percentages: Dict[str, float], labels: List[str]) -> tuple[float | None, dict[str, dict]]:
    scored = [label for label in labels if label in IDEAL_ATTENTION_RANGES]
    details: dict[str, dict] = {}
    if not scored:
        return None, details
    total_weight = sum(ELEMENT_IMPORTANCE.get(label, 0.1) for label in scored)
    score = 0.0
    for label in scored:
        low, high = IDEAL_ATTENTION_RANGES[label]
        share = float(percentages.get(label, 0.0))
        band_score = score_for_ideal_range(share, low, high)
        status = "healthy" if band_score >= 100 else ("too low" if share < low else "too high")
        details[label] = {"attention_pct": round(share, 2), "ideal_range": [low, high], "fit": round(band_score, 1), "status": status}
        score += band_score * ELEMENT_IMPORTANCE.get(label, 0.1) / total_weight
    return score, details


def visual_hierarchy(percentages: Dict[str, float], labels: List[str]) -> tuple[float | None, list[str]]:
    """Share of element pairs whose attention order matches the expected order."""
    ranked = [label for label in labels if label in EXPECTED_RANK]
    observed_order = sorted(ranked, key=lambda label: percentages.get(label, 0.0), reverse=True)
    if len(ranked) < 2:
        if not ranked:
            return None, observed_order
        return (100.0 if ranked[0] == "Product" else 60.0), observed_order
    agree = total = 0.0
    for i, a in enumerate(ranked):
        for b in ranked[i + 1 :]:
            if EXPECTED_RANK[a] == EXPECTED_RANK[b]:
                continue
            first, second = (a, b) if EXPECTED_RANK[a] < EXPECTED_RANK[b] else (b, a)
            gap = percentages.get(first, 0.0) - percentages.get(second, 0.0)
            total += 1
            agree += 1.0 if gap > 2 else (0.5 if gap >= -2 else 0.0)  # near-ties count half
    return (agree / total * 100 if total else 100.0), observed_order


def focus(percentages: Dict[str, float], labels: List[str]) -> float:
    background = float(percentages.get(BACKGROUND_LABEL, 0.0))
    background_score = clamp(100 - max(0.0, background - 15) * 2.5)  # 15% ok, 55% -> 0
    element_total = sum(percentages.get(label, 0.0) for label in labels)
    if element_total <= 0 or len(labels) < 2:
        return background_score
    top_share = max(percentages.get(label, 0.0) for label in labels) / element_total * 100
    dominance_score = clamp(100 - max(0.0, top_share - 60) / 35 * 100)  # 60% ok, 95% -> 0
    return 0.6 * background_score + 0.4 * dominance_score


def completeness(labels: List[str]) -> tuple[float, list[str]]:
    present = set(labels)
    missing = [label for label in COMPLETENESS_POINTS if label not in present]
    return clamp(sum(points for label, points in COMPLETENESS_POINTS.items() if label in present)), missing


def readability(image: np.ndarray | None, boxes: list[tuple[str, int, int, int, int]]) -> tuple[float | None, list[dict]]:
    if image is None:
        return None, []
    from analysis.readability import assess_text_elements

    items = assess_text_elements(image, boxes)
    if not items:
        return None, []
    weights = {"Headline": 0.4, "CTA": 0.4, "Price": 0.2}
    total = sum(weights[item.label] for item in items)
    return sum(item.score * weights[item.label] for item in items) / total, [item.to_dict() for item in items]


def get_pes_category(score: float) -> str:
    if score >= PES_CATEGORIES.get("excellent", 80):
        return "Excellent"
    if score >= PES_CATEGORIES.get("good", 65):
        return "Good"
    if score >= PES_CATEGORIES.get("average", 50):
        return "Average"
    return "Needs Improvement"


def _insights(components: dict[str, float], elements: dict, order: list[str], missing: list[str],
              text: list[dict], percentages: Dict[str, float]) -> list[str]:
    notes: list[str] = []
    for label, detail in elements.get("attention", {}).items():
        if detail["status"] == "healthy":
            continue
        low, high = detail["ideal_range"]
        notes.append(f"{label} gets {detail['attention_pct']:.0f}% of attention ({detail['status']}; healthy is {low:.0f}-{high:.0f}%).")
    if missing:
        notes.append("Missing elements: " + ", ".join(missing) + ".")
    if order:
        notes.append("Attention order: " + " > ".join(order) + ".")
    background = percentages.get(BACKGROUND_LABEL, 0.0)
    if background > 25:
        notes.append(f"{background:.0f}% of attention falls outside the marked elements (background or clutter).")
    for item in text:
        if item["contrast_ratio"] < 3:
            notes.append(f"{item['label']} contrast is about {item['contrast_ratio']:.1f}:1, below the 3:1 minimum for large text.")
        if item["size_score"] < 60:
            notes.append(f"{item['label']} is small: {item['height_share'] * 100:.1f}% of the poster height.")
    if not notes:
        notes.append("All measured components are in healthy ranges.")
    return notes


def calculate_pes(
    percentages: Dict[str, float],
    aoi_records: Iterable[Any],
    image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calculate PES from attention shares (incl. Background), AOIs and, optionally, the image."""
    boxes = _boxes(aoi_records)
    labels = _present_labels(boxes)

    fit, attention_details = attention_fit(percentages, labels)
    hierarchy, order = visual_hierarchy(percentages, labels)
    complete, missing = completeness(labels)
    read, text_details = readability(image, boxes)
    raw = {
        "Attention Fit": fit,
        "Visual Hierarchy": hierarchy,
        "Focus": focus(percentages, labels) if labels else None,
        "Completeness": complete,
        "Readability": read,
    }
    measured = {name: value for name, value in raw.items() if value is not None}
    used_weight = sum(PES_WEIGHTS.get(name, 0.0) for name in measured)
    effective_weights = {
        name: (PES_WEIGHTS.get(name, 0.0) / used_weight * 100 if name in measured and used_weight else 0.0)
        for name in PES_WEIGHTS
    }
    score = sum(measured[name] * effective_weights[name] / 100 for name in measured) if used_weight else 0.0
    components = {name: round(value, 2) for name, value in measured.items()}
    elements = {"attention": attention_details, "missing": missing, "attention_order": order, "text": text_details}
    return {
        "version": PES_VERSION,
        "score": round(clamp(score), 2),
        "category": get_pes_category(score),
        "components": components,
        "weights": {name: round(weight, 2) for name, weight in effective_weights.items() if name in measured},
        "configured_weights": PES_WEIGHTS,
        "not_measured": [name for name, value in raw.items() if value is None],
        "elements": elements,
        "insights": _insights(components, elements, order, missing, text_details, percentages),
    }
