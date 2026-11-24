from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from analysis.aoi import AOIRecord
from analysis.attention import calculate_attention_percentages, summarize_attention
from analysis.recommendations import build_recommendations
from analysis.scoring import calculate_pes


def build_structured_report(
    records: List[AOIRecord],
    attention_scores: Dict[str, float],
    image: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Machine-friendly report: attention summary, PES v2 and ranked recommendations."""
    summary = summarize_attention(attention_scores)
    percentages = calculate_attention_percentages(attention_scores, records)
    pes = calculate_pes(percentages, records, image)
    return {
        "summary": summary,
        "pes": pes,
        "recommendations": build_recommendations(pes, percentages),
        "records": [record.to_dict() for record in records],
    }


def report_to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2)
