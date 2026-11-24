"""PDF exports: a full analysis report, plus the original minimal helper."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any


def create_pdf_with_image_and_text(image_bytes: bytes, text: str) -> bytes:
    """Minimal PDF with one image and plain text (kept for backwards compatibility)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 100
    try:
        image = ImageReader(io.BytesIO(image_bytes))
        iw, ih = image.getSize()
        scaled_h = ih * width / iw
        pdf.drawImage(image, 0, height - scaled_h, width=width, height=scaled_h)
        y = height - scaled_h - 40
    except Exception:
        pass
    for line in text.splitlines():
        if y < 40:
            pdf.showPage()
            y = height - 40
        pdf.drawString(40, y, line)
        y -= 14
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _image_from_uri(uri: str | None) -> tuple[io.BytesIO, int, int] | None:
    if not uri or "," not in uri:
        return None
    from PIL import Image as PILImage

    raw = base64.b64decode(uri.split(",", 1)[1])
    with PILImage.open(io.BytesIO(raw)) as image:
        width, height = image.size
    return io.BytesIO(raw), width, height


def build_analysis_pdf(payload: dict[str, Any]) -> bytes:
    """A shareable report: score, components, attention, recommendations and both images."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=9, leading=12)
    pes = payload.get("pes") or {}
    story: list[Any] = []
    is_gaze = str(payload.get("attention_source", "")).startswith(("gaze", "webgazer"))

    story.append(Paragraph("Poster Effectiveness Report", styles["Title"]))
    created = payload.get("created_at")
    when = datetime.fromtimestamp(created).strftime("%d %b %Y %H:%M") if created else datetime.now().strftime("%d %b %Y %H:%M")
    story.append(Paragraph(
        f"{payload.get('filename') or 'Poster'} &middot; {when} &middot; "
        f"attention from {'measured eye-tracking' if is_gaze else 'estimated saliency'}", small))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(f"<b>PES {pes.get('score', 0):.0f}/100 &mdash; {pes.get('category', '')}</b>", styles["Heading2"]))

    rows = [["Component", "Score", "Weight"]]
    for name, value in (pes.get("components") or {}).items():
        rows.append([name, f"{value:.0f}", f"{(pes.get('weights') or {}).get(name, 0):.0f}%"])
    for name in pes.get("not_measured") or []:
        rows.append([name, "not measured", "-"])
    table = Table(rows, colWidths=[70 * mm, 30 * mm, 30 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(table)
    story.append(Spacer(1, 5 * mm))

    percentages = pes.get("attention_percentages") or {}
    if percentages:
        story.append(Paragraph("Attention share", styles["Heading3"]))
        comparison = payload.get("attention_comparison") or {}
        predicted = comparison.get("predicted") or {}
        header = ["Element", "Measured" if is_gaze else "Estimated"] + (["Saliency estimate"] if predicted else [])
        data = [header] + [
            [label, f"{pct:.1f}%"] + ([f"{predicted.get(label, 0):.1f}%"] if predicted else [])
            for label, pct in sorted(percentages.items(), key=lambda kv: -kv[1])
        ]
        attention_table = Table(data, colWidths=[50 * mm] + [35 * mm] * (len(header) - 1))
        attention_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        story.append(attention_table)
        story.append(Spacer(1, 5 * mm))

    recs = payload.get("recommendations") or []
    if recs:
        story.append(Paragraph("Recommendations (highest impact first)", styles["Heading3"]))
        for index, rec in enumerate(recs, start=1):
            if isinstance(rec, dict):
                story.append(Paragraph(f"<b>{index}. {rec.get('title', '')}</b> (+{rec.get('impact', 0):.1f} PES) &mdash; {rec.get('detail', '')}", small))
            else:
                story.append(Paragraph(f"{index}. {rec}", small))
            story.append(Spacer(1, 2 * mm))

    insights = pes.get("insights") or []
    if insights:
        story.append(Paragraph("Findings", styles["Heading3"]))
        for item in insights:
            story.append(Paragraph(f"&bull; {item}", small))

    images = [(_image_from_uri(payload.get("annotated_image")), "Detected elements"),
              (_image_from_uri(payload.get("saliency_image")), "Measured gaze" if is_gaze else "Estimated attention")]
    cells = []
    for found, caption in images:
        if found is None:
            continue
        stream, iw, ih = found
        h = min(110 * mm, 80 * mm * ih / iw)
        cells.append([Image(stream, width=h * iw / ih, height=h), Paragraph(caption, small)])
    if cells:
        story.append(Spacer(1, 5 * mm))
        story.append(Table([[c[0] for c in cells], [c[1] for c in cells]]))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "PES is a configurable design heuristic (see config/scoring.yaml), not a validated sales prediction.", small))
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm).build(story)
    return buffer.getvalue()
