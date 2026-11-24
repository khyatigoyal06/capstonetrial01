from __future__ import annotations

import asyncio
import base64
import csv
import json
import math
import io
import re
import struct
import sys
import tempfile
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import chat
from analysis.aoi import AOIRecord, build_aoi_record, draw_aoi_boxes, parse_yolo_labels
from analysis.attention import simple_heatmap
from analysis.detection import detect_poster_elements
from analysis.gaze import GazeAnalyzer, parse_gaze_points
from analysis.fixations import render_fixation_map
from analysis.gaze_session import analyze_engine_session, fixations_from_stats, load_engine_session
from analysis.recommendations import build_recommendations
from analysis.saliency import (
    attention_percentages_from_saliency,
    saliency_heatmap_overlay,
)
from analysis.scoring import calculate_pes
from analysis.store import AnalysisStore
from config.settings import (
    CHAT_RATE_LIMIT_PER_MINUTE,
    CLASS_NAMES,
    DATASET_IMAGE_DIR,
    DEFAULT_MODEL_WEIGHTS,
    GROQ_MODEL,
    MAX_UPLOAD_MB,
)
from utils.pdf_export import build_analysis_pdf
from gaze_engine import config as gaze_config
from gaze_engine.backend import GazeBackendError, backend_availability, get_backend
from gaze_engine.session import GazeSession
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


app = FastAPI(title="Poster PES API", version="0.3.0", lifespan=lifespan)
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
    started = time.perf_counter()
    response = await call_next(request)
    processing_seconds.observe(time.perf_counter() - started)
    return response

# Analyses are stored in SQLite (data/analyses.sqlite3) so chat, PDF reports
# and history keep working after a restart.
_store: AnalysisStore | None = None


def get_store() -> AnalysisStore:
    global _store
    if _store is None:
        _store = AnalysisStore()
    return _store


# Simple per-client limit so a public deployment cannot burn the Groq key.
_chat_calls: dict[str, deque[float]] = defaultdict(deque)


def _check_chat_rate(client: str) -> None:
    now = time.monotonic()
    calls = _chat_calls[client]
    while calls and now - calls[0] > 60:
        calls.popleft()
    if len(calls) >= CHAT_RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many chat messages; wait a minute and try again.")
    calls.append(now)


async def read_upload(upload: UploadFile, what: str) -> bytes:
    """Read an upload, refusing anything over MAX_UPLOAD_MB."""
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"The {what} is larger than {MAX_UPLOAD_MB:g} MB.")
    if not data:
        raise HTTPException(status_code=400, detail=f"The {what} is empty.")
    return data


class ChatRequest(BaseModel):
    analysis_id: str
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list)


@app.get("/api/health")
def health() -> dict[str, Any]:
    gaze_ok, gaze_reason = backend_availability()
    return {
        "status": "ok",
        "version": app.version,
        "detector_weights": DEFAULT_MODEL_WEIGHTS.name,
        "detector_available": DEFAULT_MODEL_WEIGHTS.is_file(),
        "gaze_engine_available": gaze_ok,
        "gaze_engine_reason": gaze_reason,
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

    image = decode_image(await read_upload(file, "poster"))
    inference_requests.inc()

    try:
        records, source = detect_poster_elements(image)
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
    pes = calculate_pes(percentages, records, image)

    payload = {
        "analysis_id": uuid.uuid4().hex,
        "filename": file.filename,
        "source": source,
        "attention_source": "saliency_estimate",
        "detections": [record.to_dict() for record in records],
        "pes": {**pes, "mode": "saliency_attention", "attention_percentages": percentages},
        "recommendations": build_recommendations(pes, percentages),
        "annotated_image": encode_annotated_image(image, records),
        "saliency_image": encode_image_data_uri(saliency_heatmap_overlay(image)),
    }
    _remember_analysis(payload, image)
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

    image = decode_image(await read_upload(image_file, "poster"))
    gaze_bytes = await read_upload(gaze_file, "gaze file")

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
            records, detection_source = detect_poster_elements(image)
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Poster detection failed: {exc}") from exc

    suffix = ".tsv" if gaze_name.endswith(".tsv") else ".csv"
    engine_session = load_engine_session(gaze_bytes, delimiter="\t" if suffix == ".tsv" else ",")
    if engine_session is not None:
        attention_ms, stats, points = analyze_engine_session(engine_session, records, image.shape)
        heatmap = render_fixation_map(image, fixations_from_stats(stats), stats["quality"]["sigma_image_px"])
        return _gaze_payload(
            image_file.filename, image, records, detection_source, attention_ms, stats, points,
            attention_source="gaze_engine_fixations", heatmap_image=heatmap,
        )

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

    return _gaze_payload(
        image_file.filename, image, records, detection_source, analyzer.attention_scores, stats, points,
        attention_source="webgazer_gaze",
    )


def _gaze_payload(
    filename: str | None,
    image: np.ndarray,
    records: list[AOIRecord],
    detection_source: str,
    attention_scores: dict[str, float],
    stats: dict[str, Any],
    points: list[tuple[int, int, float]],
    *,
    attention_source: str,
    heatmap_image: np.ndarray | None = None,
) -> dict[str, Any]:
    percentages = _attention_percentages(attention_scores)
    pes = calculate_pes(percentages, records, image)
    recommendations = build_recommendations(pes, percentages)
    comparison = attention_comparison(percentages, attention_percentages_from_saliency(image, records))
    quality = stats.get("quality") or {}
    insights = list(pes["insights"])
    unreliable = quality.get("unreliable_aois") or []
    if unreliable:
        insights.append(
            "Gaze accuracy in this session is too coarse to measure "
            + ", ".join(unreliable)
            + " reliably; treat their attention share as approximate."
        )
    if quality.get("session_quality") == "fair":
        insights.append("This session passed validation with fair (not good) accuracy; small elements are approximate.")
    stats = {key: value for key, value in stats.items() if key != "fixation_list"} | (
        {"fixation_count": len(stats.get("fixation_list") or [])} if "fixation_list" in stats else {}
    )
    payload = {
        "analysis_id": uuid.uuid4().hex,
        "filename": filename,
        "source": detection_source,
        "attention_source": attention_source,
        "detections": [record.to_dict() for record in records],
        "gaze_stats": stats,
        "gaze_quality": quality or None,
        "gaze_sample_count": len(points),
        "attention_scores_ms": {label: round(float(value), 1) for label, value in attention_scores.items()},
        "attention_comparison": comparison,
        "pes": {**pes, "mode": attention_source, "insights": insights, "attention_percentages": percentages},
        "recommendations": recommendations,
        "annotated_image": encode_annotated_image(image, records),
        "saliency_image": encode_image_data_uri(
            heatmap_image if heatmap_image is not None else encode_gaze_heatmap(image, points)
        ),
    }
    _remember_analysis(payload, image)
    return payload


def attention_comparison(measured: dict[str, float], predicted: dict[str, float]) -> dict[str, Any]:
    """How well the saliency estimate (Poster Review mode) matched measured gaze."""
    labels = sorted(set(measured) | set(predicted))
    m = np.array([measured.get(label, 0.0) for label in labels])
    p = np.array([predicted.get(label, 0.0) for label in labels])
    correlation = None
    if len(labels) >= 3 and m.std() > 0 and p.std() > 0:
        correlation = round(float(np.corrcoef(m, p)[0, 1]), 3)
    return {
        "measured": {label: round(float(v), 2) for label, v in zip(labels, m)},
        "predicted": {label: round(float(v), 2) for label, v in zip(labels, p)},
        "mean_abs_error_pct": round(float(np.mean(np.abs(m - p))), 2) if labels else None,
        "pearson_r": correlation,
    }


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
        for poster in available:
            refreshed = _gallery_aois_from_labels(dataset_root, poster)
            if refreshed is not None:
                poster["aois"] = refreshed
        page = gallery_pattern.sub(
            lambda found: f"{found.group(1)}{json.dumps(available, ensure_ascii=False)}{found.group(3)}",
            page,
            count=1,
        )
    # Its poster links are root-relative when framed under /api/gaze/capture.
    page = page.replace("<head>", '<head>\n  <base href="/">', 1)
    return HTMLResponse(page)


def get_gaze_backend():
    """Indirection so tests can substitute a fake backend."""
    return get_backend()


@app.get("/api/gaze/engine")
def gaze_engine_status() -> dict[str, Any]:
    available, reason = backend_availability()
    return {"available": available, "reason": reason, "mode": gaze_config.GAZE_ENGINE_MODE}


@app.post("/api/gaze/predict")
async def predict_gaze(frame: UploadFile = File(...)) -> dict[str, Any]:
    """Run the gaze engine on one camera frame (uncalibrated; for diagnostics)."""
    if not frame.content_type or not frame.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Send a JPEG or PNG camera frame.")
    image_bytes = await frame.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The camera frame is empty.")
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The camera frame could not be decoded.")
    try:
        observation = await asyncio.to_thread(get_gaze_backend().process, image)
    except GazeBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return observation.to_dict()


@app.websocket("/api/gaze/stream")
async def gaze_stream(websocket: WebSocket) -> None:
    """Live calibrated gaze.

    Binary messages are frames: 8-byte little-endian float64 capture time (ms)
    followed by JPEG bytes. Text messages are JSON control messages, see
    gaze_engine/session.py. The client sends the next frame only after it gets
    the reply to the previous one, so throughput adapts to the CPU.
    """
    await websocket.accept()
    session = GazeSession()
    try:
        backend = get_gaze_backend()
        await websocket.send_json({"type": "status", "stage": "loading_model"})
        warm_up = getattr(backend, "warm_up", None)
        if warm_up is not None:
            await asyncio.to_thread(warm_up)
        await websocket.send_json({"type": "status", "stage": "model_ready"})
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                data = message["bytes"]
                if len(data) <= 8:
                    await websocket.send_json({"type": "error", "detail": "frame message too short"})
                    continue
                (t_ms,) = struct.unpack("<d", data[:8])
                image = cv2.imdecode(np.frombuffer(data[8:], np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    await websocket.send_json({"type": "gaze", "t": t_ms, "valid": False, "reason": "bad_frame"})
                    continue
                started = time.perf_counter()
                observation = await asyncio.to_thread(backend.process, image)
                reply = session.handle_frame(observation, t_ms)
                reply["proc_ms"] = round((time.perf_counter() - started) * 1000, 1)
                await websocket.send_json(reply)
            elif message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "detail": "invalid JSON"})
                    continue
                for reply in session.handle_control(control if isinstance(control, dict) else {}):
                    await websocket.send_json(reply)
    except GazeBackendError as exc:
        await websocket.send_json({"type": "error", "fatal": True, "detail": str(exc)})
        await websocket.close(code=1011)
    except WebSocketDisconnect:
        pass


def _gallery_aois_from_labels(dataset_root: Path, poster: dict[str, Any]) -> list[dict[str, Any]] | None:
    """AOIs for a gallery poster, read from its current YOLO label file.

    The AOI list embedded in the capture page was generated once and still has
    boxes from before the labels were cleaned (negative or off-image
    coordinates), so the label files are the source of truth.
    """
    width, height = int(poster.get("width") or 0), int(poster.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    stem = Path(poster["name"]).stem
    for split in ("train", "valid", "test"):
        label_path = dataset_root / split / "labels" / f"{stem}.txt"
        if label_path.is_file():
            records = parse_yolo_labels(str(label_path), (height, width))
            return [
                {
                    "label": r.class_name,
                    "x1": max(0, min(width, r.x1)), "y1": max(0, min(height, r.y1)),
                    "x2": max(0, min(width, r.x2)), "y2": max(0, min(height, r.y2)),
                }
                for r in records
                if min(width, r.x2) - max(0, r.x1) > 1 and min(height, r.y2) - max(0, r.y1) > 1
            ]
    return None


@app.get("/api/analyses")
def list_analyses(limit: int = 30) -> dict[str, Any]:
    return {"analyses": get_store().list(limit)}


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> dict[str, Any]:
    found = get_store().get(analysis_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return found[0]


@app.delete("/api/analyses/{analysis_id}")
def delete_analysis(analysis_id: str) -> dict[str, bool]:
    if not get_store().delete(analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {"deleted": True}


@app.get("/api/analyses/{analysis_id}/report.pdf")
def analysis_report_pdf(analysis_id: str) -> Response:
    found = get_store().get(analysis_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    payload = found[0]
    stem = Path(payload.get("filename") or "poster").stem[:40] or "poster"
    return Response(
        build_analysis_pdf(payload),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{stem}_pes_report.pdf"'},
    )


@app.post("/api/chat")
def chat_turn(request: ChatRequest, http_request: Request) -> dict[str, str]:
    _check_chat_rate(http_request.client.host if http_request.client else "local")
    found = get_store().get(request.analysis_id)
    if found is None:
        raise HTTPException(status_code=404, detail="That analysis was not found. Re-analyze the poster to continue.")
    payload, poster = found

    try:
        reply = chat.ask(
            analysis=payload,
            history=request.history,
            user_message=request.message,
            image_data_uri=chat.encode_image_for_chat(poster) if poster is not None else None,
        )
    except chat.ChatConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except chat.ChatUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"reply": reply}


def _remember_analysis(payload: dict[str, Any], image: np.ndarray) -> None:
    try:
        get_store().save(payload, image)
    except Exception as exc:  # storage must never break an analysis
        print(f"warning: could not store analysis {payload.get('analysis_id')}: {exc}")


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
        # Pillow covers formats some OpenCV builds lack (AVIF, some WebP/PNG variants).
        try:
            from PIL import Image

            with Image.open(io.BytesIO(image_bytes)) as pil_image:
                image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            image = None
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
