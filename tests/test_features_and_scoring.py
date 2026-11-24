import cv2
import numpy as np
import pytest

from analysis.aoi import build_aoi_record
from analysis.features import aggregate_features
from analysis.readability import assess_text_elements, contrast_ratio
from analysis.scoring import PES_WEIGHTS, calculate_pes, score_for_ideal_range, visual_hierarchy


def make_record(class_id, cx_norm, cy_norm, w_norm, h_norm):
    return build_aoi_record(class_id, cx_norm, cy_norm, w_norm, h_norm, 800, 600, confidence=0.9)


def test_features_aggregate_by_class():
    feats = aggregate_features([make_record(3, 0.35, 0.5, 0.2, 0.3), make_record(0, 0.6, 0.55, 0.15, 0.1)], (600, 800))
    assert {"Product", "CTA"} <= set(feats)


def test_ideal_range_scoring():
    assert score_for_ideal_range(35, 30, 45) == 100
    assert score_for_ideal_range(15, 30, 45) == 50
    assert score_for_ideal_range(100, 30, 45) == 0


def test_missing_cta_costs_completeness_not_a_whole_component():
    records = [make_record(3, 0.5, 0.5, 0.4, 0.4), make_record(1, 0.5, 0.1, 0.6, 0.1)]
    pes = calculate_pes({"Product": 40, "Headline": 20, "Background": 10}, records)
    assert pes["components"]["Completeness"] == pytest.approx(60)  # Product 35 + Headline 25
    assert "CTA" in pes["elements"]["missing"]
    assert pes["components"]["Attention Fit"] == 100  # present elements are healthy
    assert pes["score"] > 70


def test_readability_is_skipped_without_image_and_weights_rescale():
    pes = calculate_pes({"Product": 40, "Background": 10}, [make_record(3, 0.5, 0.5, 0.4, 0.4)])
    assert "Readability" in pes["not_measured"]
    assert sum(pes["weights"].values()) == pytest.approx(100)
    assert sum(PES_WEIGHTS.values()) == pytest.approx(100)


def test_hierarchy_rewards_product_first():
    labels = ["Product", "Headline", "CTA", "logo"]
    good, order = visual_hierarchy({"Product": 40, "Headline": 25, "CTA": 15, "logo": 5}, labels)
    bad, _ = visual_hierarchy({"Product": 5, "Headline": 15, "CTA": 25, "logo": 40}, labels)
    assert good == 100 and order[0] == "Product"
    assert bad < 20


def test_contrast_ratio_matches_wcag_extremes():
    assert contrast_ratio((0, 0, 0), (255, 255, 255)) == pytest.approx(21, abs=0.1)
    assert contrast_ratio((128, 128, 128), (128, 128, 128)) == pytest.approx(1)


def test_readability_detects_low_contrast_text():
    image = np.full((600, 800, 3), 200, np.uint8)
    cv2.rectangle(image, (100, 20), (700, 100), (0, 0, 0), -1)
    cv2.putText(image, "BIG SALE", (150, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5)  # white on black
    cv2.putText(image, "order now", (300, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (190, 190, 190), 2)  # grey on grey
    items = {i.label: i for i in assess_text_elements(image, [("Headline", 100, 20, 700, 100), ("CTA", 290, 525, 480, 575)])}
    assert items["Headline"].contrast_ratio > 7  # WCAG AAA level
    assert items["CTA"].contrast_ratio < 1.5
    assert items["Headline"].score > items["CTA"].score
