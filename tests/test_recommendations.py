from analysis.aoi import build_aoi_record
from analysis.recommendations import build_recommendations
from analysis.scoring import calculate_pes


def make_record(class_id, cx, cy, w, h):
    return build_aoi_record(class_id, cx, cy, w, h, 800, 600, confidence=0.9)


def test_recommendations_are_ranked_and_actionable():
    records = [make_record(3, 0.4, 0.5, 0.2, 0.2)]
    percentages = {"Product": 15.0, "Background": 85.0}
    recs = build_recommendations(calculate_pes(percentages, records), percentages)
    titles = [r["title"] for r in recs]
    assert any("Missing CTA" in t for t in titles)
    assert any(t.startswith("Product: attention too low") for t in titles)
    assert any("clutter" in t.lower() for t in titles)
    impacts = [r["impact"] for r in recs]
    assert impacts == sorted(impacts, reverse=True)
    assert all(r["detail"] and r["component"] for r in recs)


def test_healthy_poster_gets_positive_feedback():
    records = [make_record(i, 0.2 + 0.15 * i, 0.5, 0.1, 0.1) for i in range(5)]
    percentages = {"Product": 38, "Headline": 22, "CTA": 15, "Price": 10, "logo": 6, "Background": 9}
    recs = build_recommendations(calculate_pes(percentages, records), percentages)
    assert recs[0]["title"] == "Strong poster"
