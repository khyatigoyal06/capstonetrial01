import numpy as np
import pytest

from analysis.aoi import build_aoi_record
from analysis.fixations import Fixation, aoi_reliability, detect_fixations, probabilistic_aoi_attention


def _cluster(t0, x, y, n=10, step=50, jitter=5, seed=0):
    rng = np.random.default_rng(seed)
    return [(t0 + i * step, x + rng.normal(0, jitter), y + rng.normal(0, jitter)) for i in range(n)]


def test_idt_finds_two_fixations_and_ignores_a_saccade_sample():
    samples = _cluster(0, 100, 100) + [(520, 400, 300)] + _cluster(560, 700, 500, seed=1)
    fixations = detect_fixations(samples, dispersion_px=60, min_duration_ms=100)
    assert len(fixations) == 2
    assert fixations[0].x == pytest.approx(100, abs=6) and fixations[1].x == pytest.approx(700, abs=6)
    assert fixations[0].duration_ms == pytest.approx(500)


def test_idt_splits_on_sampling_gap():
    samples = _cluster(0, 100, 100) + _cluster(1500, 100, 100, seed=2)
    assert len(detect_fixations(samples, dispersion_px=60, max_gap_ms=250)) == 2


def test_probabilistic_attention_conserves_time_and_handles_overlap():
    shape = (400, 400, 3)
    product = build_aoi_record(3, 0.5, 0.5, 0.5, 0.5, 400, 400)   # 100..300
    cta = build_aoi_record(0, 0.5, 0.5, 0.1, 0.1, 400, 400)       # 180..220, inside product
    fixations = [Fixation(200, 200, 0, 1000, 10), Fixation(390, 20, 1000, 1500, 5), Fixation(500, 500, 1500, 1800, 3)]
    attention, stats = probabilistic_aoi_attention(fixations, [product, cta], shape, sigma_px=5)
    assert sum(attention.values()) == pytest.approx(1500)
    assert attention["CTA"] == pytest.approx(1000, rel=0.02)  # small sigma: centre fixation is the CTA
    assert attention["Background"] == pytest.approx(500, rel=0.02)
    assert stats["off_poster_ms"] == pytest.approx(300)

    blurred, _ = probabilistic_aoi_attention(fixations[:1], [product, cta], shape, sigma_px=60)
    # 40 px CTA under a 60 px error: only ~7% of the fixation can honestly be credited to it.
    assert 0.03 < blurred["CTA"] / 1000 < 0.15
    assert sum(blurred.values()) == pytest.approx(1000)


def test_reliability_flags_aois_smaller_than_the_error():
    product = build_aoi_record(3, 0.5, 0.5, 0.5, 0.5, 400, 400)
    cta = build_aoi_record(0, 0.5, 0.9, 0.3, 0.05, 400, 400)
    flags = {item["label"]: item["reliable"] for item in aoi_reliability([product, cta], sigma_px=40)}
    assert flags == {"CTA": False, "Product": True}
