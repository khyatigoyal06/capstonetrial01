"""Turn a PES result into prioritised, concrete design recommendations.

Each recommendation names the component it improves and an *impact*: the PES
points available if that component were fixed (its weight x the points it is
missing). Recommendations are sorted by impact, so the first one is the change
most worth making.
"""

from __future__ import annotations

from typing import Any

from config.settings import BACKGROUND_LABEL

_ATTENTION_ADVICE = {
    ("Product", "too low"): "Make the food the hero: enlarge it, move it toward the centre, and give it the strongest light and colour contrast on the poster.",
    ("Product", "too high"): "The product swallows the poster. Give the headline and call-to-action more weight (size, contrast, clear space) so the message gets read too.",
    ("Headline", "too low"): "Strengthen the headline: shorter wording, larger and bolder type, and a calm area behind it.",
    ("Headline", "too high"): "The headline competes with the product. Reduce its size or weight so the product leads.",
    ("CTA", "too low"): "Make the call-to-action pop: a solid button shape in a colour that contrasts with everything around it, close to the product.",
    ("CTA", "too high"): "The call-to-action is louder than the offer. Tone it down so the product and headline lead.",
    ("Price", "too low"): "Make the price or offer easier to spot: badge or burst shape, high contrast, near the product.",
    ("Price", "too high"): "The price dominates. Reduce it if the poster is meant to sell taste or brand rather than discount.",
    ("logo", "too low"): "Brand recall is weak. Give the logo a clean corner with enough clear space.",
    ("logo", "too high"): "The logo pulls too much attention. Make it smaller and keep it in a corner.",
}

_MISSING_ADVICE = {
    "Product": "Add a clear product image. Food posters are judged mainly on how appetising the product looks.",
    "Headline": "Add a short headline (3-6 words) that says what is on offer.",
    "CTA": "Add a call-to-action (e.g. \"Order now\", \"Visit today\") so viewers know what to do next.",
    "logo": "Add the brand logo so the poster builds recognition.",
    "Price": "Consider showing the price or offer; it is one of the strongest triggers in food advertising.",
}


def build_recommendations(pes: dict[str, Any], percentages: dict[str, float] | None = None) -> list[dict[str, Any]]:
    weights = pes.get("weights") or {}
    components = pes.get("components") or {}
    elements = pes.get("elements") or {}
    percentages = percentages or {}

    def impact(component: str, share: float = 1.0) -> float:
        return round(weights.get(component, 0.0) * (100 - components.get(component, 100.0)) / 100 * share, 1)

    recs: list[dict[str, Any]] = []

    attention = elements.get("attention") or {}
    off = {label: d for label, d in attention.items() if d["status"] != "healthy"}
    for label, detail in off.items():
        text = _ATTENTION_ADVICE.get((label, detail["status"]))
        if text:
            recs.append({
                "title": f"{label}: attention {detail['status']} ({detail['attention_pct']:.0f}%)",
                "detail": text,
                "component": "Attention Fit",
                "impact": impact("Attention Fit", (100 - detail["fit"]) / max(1, sum(100 - d["fit"] for d in off.values()))),
            })

    missing = elements.get("missing") or []
    for label in missing:
        recs.append({
            "title": f"Missing {label}",
            "detail": _MISSING_ADVICE.get(label, f"Add a {label}."),
            "component": "Completeness",
            "impact": impact("Completeness", 1 / max(1, len(missing))),
        })

    for item in elements.get("text") or []:
        problems = []
        if item["contrast_ratio"] < 4.5:
            problems.append(f"contrast is about {item['contrast_ratio']:.1f}:1 (aim for 4.5:1; a light text on dark band or vice versa fixes it)")
        if item["size_score"] < 80:
            problems.append(f"it is only {item['height_share'] * 100:.1f}% of the poster height - make it larger")
        if problems:
            recs.append({
                "title": f"Readability of {item['label']}",
                "detail": f"{item['label']}: " + "; ".join(problems) + ".",
                "component": "Readability",
                "impact": impact("Readability", (100 - item["score"]) / max(1, sum(100 - t["score"] for t in elements.get("text") or []))),
            })

    if components.get("Visual Hierarchy", 100) < 70:
        order = " > ".join(elements.get("attention_order") or [])
        recs.append({
            "title": "Fix the viewing order",
            "detail": f"Viewers currently look at {order}. Lead with the product, then the headline/offer, then the call-to-action; "
                      "use size and contrast steps between them rather than making everything loud.",
            "component": "Visual Hierarchy",
            "impact": impact("Visual Hierarchy"),
        })

    if components.get("Focus", 100) < 70:
        background = percentages.get(BACKGROUND_LABEL, 0.0)
        recs.append({
            "title": "Reduce clutter",
            "detail": f"{background:.0f}% of attention lands outside the key elements. Remove decorative detail, "
                      "simplify the background or blur it, and group related text.",
            "component": "Focus",
            "impact": impact("Focus"),
        })

    recs.sort(key=lambda rec: rec["impact"], reverse=True)
    if not recs:
        recs.append({
            "title": "Strong poster",
            "detail": "Every measured component is healthy. A/B test small variants (CTA wording, colour) to optimise further.",
            "component": "Overall",
            "impact": 0.0,
        })
    return recs


def recommendation_texts(recs: list[dict[str, Any]]) -> list[str]:
    return [f"{rec['title']} - {rec['detail']}" for rec in recs]
