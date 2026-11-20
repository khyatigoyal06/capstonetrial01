from __future__ import annotations

import base64
import csv
import json
import math
import io
import re
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import chat
from analysis.aoi import AOIRecord, build_aoi_record, draw_aoi_boxes
from analysis.attention import simple_heatmap
from analysis.detection import detect_poster_elements
from analysis.gaze import GazeAnalyzer, parse_gaze_points
from analysis.pretrained_gaze import PretrainedGazeError, pretrained_gaze_estimator
from analysis.recommendations import build_recommendations
from analysis.saliency import (
    attention_percentages_from_saliency,
    saliency_heatmap_overlay,
)
from analysis.scoring import calculate_pes, score_poster
from config.settings import CLASS_NAMES, DATASET_IMAGE_DIR, GROQ_MODEL
from models.detector import ModelNotFoundError
from monitoring.metrics import (
    gaze_analysis_requests,
    inference_requests,
    processing_seconds,
    start_metrics_server,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_metrics_server()
    yield


app = FastAPI(title="Poster PES API", version="0.2.0", lifespan=lifespan)
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observe_request_duration(request, call_next):
    import time

    started = time.perf_counter()
    response = await call_next(request)
    processing_seconds.observe(time.perf_counter() - started)
    return response

# Analyses are kept in-process so the chat turn can reference the poster and its
# numbers without the browser re-uploading either. Bounded because this is a
# long-running server, not a request-scoped cache; a restart clears it, which is
# acceptable for a single-node deployment.
_MAX_CACHED_ANALYSES = 32
_analyses: OrderedDict[str, dict[str, Any]] = OrderedDict()


class ChatRequest(BaseModel):
    analysis_id: str
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "chat_configured": chat.is_configured(),
        "chat_model": GROQ_MODEL,
    }


@app.get("/api/chat/models")
def chat_models() -> dict[str, Any]:
    """List model ids the configured Groq key can reach (troubleshooting aid)."""
    try:
        return {"configured_model": GROQ_MODEL, "available": chat.list_models()}
    except chat.ChatConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except chat.ChatUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/analyze-poster")
async def analyze_poster(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")

    image_bytes = await file.read()
    image = decode_image(image_bytes)
    inference_requests.inc()

    try:
        records, source = detect_poster_elements(image, conf=0.25)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Poster detection failed. Check MODEL_WEIGHTS and model compatibility. {exc}",
        ) from exc

    # Attention share drives PES, so it is estimated from saliency rather than
    # box area — see analysis/saliency.py for why.
    percentages = attention_percentages_from_saliency(image, records)
    pes = calculate_pes(percentages, records)
    legacy_scores = score_poster(records, percentages)
    recommendations = build_recommendations(percentages, records, legacy_scores)

    analysis_id = uuid.uuid4().hex
    payload = {
        "analysis_id": analysis_id,
        "filename": file.filename,
        "source": source,
        "detections": [record.to_dict() for record in records],
        "pes": {
            "mode": "saliency_attention",
            "score": pes["score"],
            "category": pes["category"],
            "components": pes["components"],
            "weights": pes["weights"],
            "insights": pes["insights"],
            "attention_percentages": percentages,
        },
        "recommendations": recommendations,
        "annotated_image": encode_annotated_image(image, records),
        "saliency_image": encode_image_data_uri(saliency_heatmap_overlay(image)),
    }

    _remember_analysis(analysis_id, payload, image)
    return payload


@app.post("/api/analyze-gaze")
async def analyze_gaze(
    image_file: UploadFile = File(...),
    gaze_file: UploadFile = File(...),
    aoi_json: str | None = Form(None),
) -> dict[str, Any]:
    """Analyze an uploaded WebGazer/Tobii export against a poster image."""
    if not image_file.content_type or not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid poster image.")
    gaze_name = (gaze_file.filename or "").lower()
    if not gaze_name.endswith((".csv", ".tsv")):
        raise HTTPException(status_code=400, detail="Upload a gaze export in CSV or TSV format.")

    image = decode_image(await image_file.read())
    gaze_bytes = await gaze_file.read()
    if not gaze_bytes:
        raise HTTPException(status_code=400, detail="The gaze file is empty.")

    inference_requests.inc()
    gaze_analysis_requests.inc()
    if not aoi_json:
        aoi_json = _annotations_from_webgazer_csv(gaze_bytes, suffix=".tsv" if gaze_name.endswith(".tsv") else ".csv")

    if aoi_json:
        try:
            records = _records_from_study_annotations(aoi_json, image.shape)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid study AOI annotations: {exc}") from exc
    else:
        records = []

    if records:
        detection_source = "study_annotations"
    else:
        try:
            records, detection_source = detect_poster_elements(image, conf=0.25)
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Poster detection failed: {exc}") from exc

    suffix = ".tsv" if gaze_name.endswith(".tsv") else ".csv"
    gaze_temp_path: Path | None = None
    try:
        # Close the temp handle before the existing path-based parser reopens
        # it; Windows does not allow reopening NamedTemporaryFile(delete=True).
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as gaze_temp:
            gaze_temp_path = Path(gaze_temp.name)
            gaze_temp.write(gaze_bytes)
        analyzer = GazeAnalyzer(records)
        stats = analyzer.analyze_raw_gaze_file(str(gaze_temp_path), image.shape)
        points = parse_gaze_points(str(gaze_temp_path), image.shape)
    except (UnicodeDecodeError, ValueError, csv.Error) as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse gaze export: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not analyze gaze export: {exc}") from exc
    finally:
        if gaze_temp_path is not None:
            gaze_temp_path.unlink(missing_ok=True)

    percentages = _attention_percentages(analyzer.attention_scores)
    pes = calculate_pes(percentages, records)
    legacy_scores = score_poster(records, percentages)
    recommendations = build_recommendations(percentages, records, legacy_scores)
    analysis_id = uuid.uuid4().hex
    payload = {
        "analysis_id": analysis_id,
        "filename": image_file.filename,
        "source": detection_source,
        "attention_source": "webgazer_gaze",
        "detections": [record.to_dict() for record in records],
        "gaze_stats": stats,
        "gaze_sample_count": len(points),
        "attention_scores_ms": analyzer.attention_scores,
        "pes": {
            "mode": "webgazer_gaze",
            "score": pes["score"],
            "category": pes["category"],
            "components": pes["components"],
            "weights": pes["weights"],
            "insights": pes["insights"],
            "attention_percentages": percentages,
        },
        "recommendations": recommendations,
        "annotated_image": encode_annotated_image(image, records),
        "saliency_image": encode_image_data_uri(encode_gaze_heatmap(image, points)),
    }
    _remember_analysis(analysis_id, payload, image)
    return payload


def _records_from_study_annotations(raw_annotations: str, image_shape: tuple[int, ...]) -> list[AOIRecord]:
    """Scale WebGazer's saved AOI rectangles to the uploaded image dimensions."""
    annotations = json.loads(raw_annotations)
    if isinstance(annotations, list):
        raise ValueError("poster dimensions are missing from the AOI metadata")
    if not isinstance(annotations, dict):
        raise ValueError("AOI metadata must be an object")
    source_width = int(annotations.get("width", 0))
    source_height = int(annotations.get("height", 0))
    source_aois = annotations.get("aois", [])
    if not isinstance(source_aois, list):
        raise ValueError("the AOI list must be an array")
    if not source_aois:
        return []
    if source_width <= 0 or source_height <= 0:
        raise ValueError("poster dimensions are missing")

    image_height, image_width = image_shape[:2]
    sx, sy = image_width / source_width, image_height / source_height
    records: list[AOIRecord] = []
    for item in source_aois:
        label = str(item.get("label", "")).strip()
        if not label or len(label) > 64:
            continue
        coords = [float(item[key]) for key in ("x1", "y1", "x2", "y2")]
        if not all(math.isfinite(value) for value in coords):
            continue
        x1 = min(image_width, max(0.0, coords[0] * sx))
        y1 = min(image_height, max(0.0, coords[1] * sy))
        x2 = min(image_width, max(0.0, coords[2] * sx))
        y2 = min(image_height, max(0.0, coords[3] * sy))
        if x2 <= x1 or y2 <= y1:
            continue

        class_id = next(
            (idx for idx, class_label in CLASS_NAMES.items() if class_label.casefold() == label.casefold()),
            -1,
        )
        class_names = {**CLASS_NAMES, class_id: label}
        records.append(build_aoi_record(
            class_id=class_id,
            x_center_norm=((x1 + x2) / 2) / image_width,
            y_center_norm=((y1 + y2) / 2) / image_height,
            width_norm=(x2 - x1) / image_width,
            height_norm=(y2 - y1) / image_height,
            image_width=image_width,
            image_height=image_height,
            confidence=1.0,
            class_names=class_names,
        ))
    return records


def _annotations_from_webgazer_csv(gaze_bytes: bytes, *, suffix: str) -> str | None:
    """Recover AOI geometry embedded in recent WebGazer CSV exports."""
    try:
        text = gaze_bytes.decode("utf-8-sig")
        delimiter = "\t" if suffix == ".tsv" else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        first_row = next(reader, None)
    except (UnicodeDecodeError, csv.Error):
        return None
    if not first_row:
        return None
    raw_aois = first_row.get("aoi_bounds_json")
    if not raw_aois:
        return None
    try:
        aois = json.loads(raw_aois)
    except (TypeError, json.JSONDecodeError):
        return raw_aois
    if isinstance(aois, dict):
        return json.dumps(aois)
    if isinstance(aois, list):
        return json.dumps({
            "width": first_row.get("poster_width", 0),
            "height": first_row.get("poster_height", 0),
            "aois": aois,
        })
    return raw_aois


@app.get("/", include_in_schema=False)
def frontend_index():
    index_path = FRONTEND_DIST / "index.html"
    if not index_path.is_file():
        return HTMLResponse(
            "<h1>Frontend is not built yet</h1><p>Run <code>npm run build</code> in frontend/.</p>",
            status_code=503,
        )
    return FileResponse(index_path)


@app.get("/api/gaze/capture", response_class=HTMLResponse)
def gaze_capture_page() -> HTMLResponse:
    """Serve the existing WebGazer gallery inside the main React application."""
    page_path = PROJECT_ROOT / "webgazer_choose_poster.html"
    if not page_path.is_file():
        raise HTTPException(status_code=404, detail="WebGazer capture page is missing.")
    page = page_path.read_text(encoding="utf-8")
    # The source gallery also contains two stale entries that no longer exist
    # in any split. Remove missing files before the browser builds its picker.
    gallery_pattern = re.compile(r"(const POSTERS = )(\[.*?\])(\s*;\s*const SAMPLE_INTERVAL_MS)", re.DOTALL)
    match = gallery_pattern.search(page)
    if match:
        posters = json.loads(match.group(2))
        dataset_root = DATASET_IMAGE_DIR.parent.parent
        available = [
            poster for poster in posters
            if Path(str(poster.get("name", ""))).name == poster.get("name")
            and any((dataset_root / split / "images" / poster["name"]).is_file() for split in ("train", "valid", "test"))
        ]
        page = gallery_pattern.sub(
            lambda found: f"{found.group(1)}{json.dumps(available, ensure_ascii=False)}{found.group(3)}",
            page,
            count=1,
        )
    # Its poster links are root-relative when framed under /api/gaze/capture.
    page = page.replace("<head>", '<head>\n  <base href="/">', 1)
    return HTMLResponse(page)


@app.post("/api/gaze/predict")
async def predict_gaze(frame: UploadFile = File(...)) -> dict[str, Any]:
    """Estimate a local gaze-direction feature vector from one camera frame."""
    if not frame.content_type or not frame.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Send a JPEG or PNG camera frame.")
    image_bytes = await frame.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The camera frame is empty.")
    try:
        return pretrained_gaze_estimator.predict(image_bytes)
    except PretrainedGazeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/chat")
def chat_turn(request: ChatRequest) -> dict[str, str]:
    cached = _analyses.get(request.analysis_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="That analysis is no longer in memory. Re-analyze the poster to continue.",
        )

    try:
        reply = chat.ask(
            analysis=cached["payload"],
            history=request.history,
            user_message=request.message,
            image_data_uri=cached["image_data_uri"],
        )
    except chat.ChatConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except chat.ChatUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"reply": reply}


def _remember_analysis(analysis_id: str, payload: dict[str, Any], image: np.ndarray) -> None:
    _analyses[analysis_id] = {
        "payload": payload,
        "image_data_uri": chat.encode_image_for_chat(image),
    }
    while len(_analyses) > _MAX_CACHED_ANALYSES:
        _analyses.popitem(last=False)


def _attention_percentages(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in scores.values())
    if total <= 0:
        return {label: 0.0 for label in scores}
    return {label: round(max(0.0, float(value)) / total * 100, 2) for label, value in scores.items()}


def encode_gaze_heatmap(image: np.ndarray, points: list[tuple[int, int, float]]) -> np.ndarray:
    """Create a softened gaze-density overlay in the same color space as the image."""
    if not points:
        return image.copy()
    heat = simple_heatmap([(x, y) for x, y, _ in points], image.shape).astype(np.float32)
    kernel = max(3, (min(image.shape[:2]) // 35) | 1)
    heat = cv2.GaussianBlur(heat, (kernel, kernel), 0)
    peak = float(heat.max())
    if peak <= 0:
        return image.copy()
    normalized = np.clip(heat / peak, 0, 1)
    colors = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (normalized * 0.68)[..., None]
    return np.clip(image * (1 - alpha) + colors * alpha, 0, 255).astype(np.uint8)


def decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
    return image


def encode_image_data_uri(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode image.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def encode_annotated_image(image: np.ndarray, records: list[AOIRecord]) -> str:
    preview_rgb, _ = draw_aoi_boxes(image, records)
    return encode_image_data_uri(cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))


@app.get("/Capstone.yolov8/train/images/{image_name:path}", include_in_schema=False)
def webgazer_poster_image(image_name: str) -> FileResponse:
    """Serve a gallery poster regardless of its train/valid/test split.

    The legacy WebGazer gallery stores stable train/images URLs, while its
    curated list can include files from any dataset split. Resolve only a
    basename within the configured dataset folders so a broken train URL can
    still locate its source image without exposing arbitrary filesystem paths.
    """
    if not image_name or Path(image_name).name != image_name:
        raise HTTPException(status_code=404, detail="Poster image not found.")

    dataset_root = DATASET_IMAGE_DIR.parent.parent
    for split in ("train", "valid", "test"):
        candidate = dataset_root / split / "images" / image_name
        if candidate.is_file():
            return FileResponse(candidate)

    raise HTTPException(status_code=404, detail="Poster image not found in the dataset.")

if (FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="frontend-assets",
    )
