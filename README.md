# Poster PES Analyzer

Analyse a food & beverage advertisement poster and get a **Poster Effectiveness
Score (PES)**, where people look (estimated or measured with a webcam eye
tracker), how readable the text is, and a ranked list of concrete fixes.

| Mode | What it does |
|---|---|
| **Poster review** | YOLOv8 finds the Product, Headline, CTA, Price and logo; a colour-aware saliency model estimates attention; PES v2 + recommendations; chat assistant; PDF report. |
| **Live eye-tracking study** | Webcam gaze engine (ETH-XGaze + head pose) with a positioning guide, automatic dwell calibration, held-out accuracy check and drift correction. Shows a fixation heatmap right after you stop. |
| **Analyze gaze file** | Upload a study CSV (or a Tobii export) and the poster: fixations, measured attention per element, PES, and how well the saliency estimate matched real gaze. |

Every analysis is saved locally (SQLite) and listed under **Recent analyses**,
so you can reopen it, chat about it or export the PDF later.

## Quick start

**Windows, one click:** double-click `scripts\start_windows.bat`. It finds
Python 3.10-3.12, installs everything into `.venv`, builds the frontend, stops
any old server on port 8001, opens <http://localhost:8001> and logs to
`server.log`. Delete `.venv\.deps_ok` to force a dependency reinstall.

**Manual (any OS):**

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                   # Python 3.10-3.12 (MediaPipe / PyTorch)
cd frontend && npm ci && npm run build && cd ..
python -m uvicorn api.main:app --port 8001
```

Open <http://localhost:8001>. For frontend development run `npm run dev` in
`frontend/` (port 5173; it proxies `/api`, including the gaze WebSocket, to 8001).

**Docker:** `docker compose up --build`, then <http://localhost:8001>. The image
uses CPU-only PyTorch; history and model downloads live in named volumes.

## Configuration

Copy `.env.example` to `.env`. The main settings:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | – | Enables the chat assistant (server-side only). |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Vision model; falls back automatically if Groq retires it. |
| `MODEL_WEIGHTS` | `models/weights/best.pt` | YOLO checkpoint. |
| `DETECTION_CONFIDENCE` | `0.25` | Detector threshold. |
| `GAZE_CAMERA_HFOV_DEG` | `70` | Webcam field of view guess (the study page can also correct distance per camera). |
| `MAX_UPLOAD_MB` | `20` | Upload limit. |
| `METRICS_PORT` | `8000` | Prometheus endpoint; `0` disables. |

## How PES v2 is computed

PES is the weighted mean of five 0-100 components (weights in
`config/scoring.yaml`):

| Component | Weight | Measures |
|---|---|---|
| Attention Fit | 30 | Each element's attention share against a healthy band (Product 30-45 %, Headline 15-30 %, CTA 10-20 %, Price 5-15 %, logo 3-10 %), weighted by importance. |
| Visual Hierarchy | 20 | Pairwise agreement between observed attention order and the expected order (Product → Headline/CTA → Price → logo). |
| Focus | 15 | Little attention lost to background; no single element taking > 60 %. |
| Completeness | 15 | Which key elements exist (Product 35, Headline 25, CTA 25, logo 10, Price 5 points). |
| Readability | 20 | WCAG contrast ratio of each text element's two dominant colours (target 4.5:1) and text size. |

A component that cannot be measured (e.g. Readability with no text elements) is
left out and the other weights rescaled. Categories: ≥ 80 Excellent, ≥ 65 Good,
≥ 50 Average. Recommendations are ranked by **impact** = the PES points the fix
could win back.

Changes from v1: a missing CTA now costs its Completeness share instead of
zeroing a 20-point component; Price and logo are scored; readability is new;
hierarchy uses pairwise order agreement instead of fixed rank bonuses.

**Caveat:** bands and weights are design heuristics, not fitted to sales or
click data. Attention in Poster review is an *estimate*; validate it against
real gaze with `tools/validate_saliency.py` before relying on it.

## Eye tracking: the gaze engine

`gaze_engine/` runs on the API; the browser only captures frames and shows
targets.

```
webcam frame (720p) ──WebSocket──▶ /api/gaze/stream
  MediaPipe Face Landmarker → solvePnP head pose (real camera intrinsics)
  → data normalisation → ETH-XGaze ResNet-18 → 3D gaze direction
  → gaze ray from the 3D face centre ∩ camera plane      (head-pose aware)
  → per-user affine/quadratic map (leave-one-target-out CV) → screen px
  → One-Euro filter → fixations (I-DT) → Gaussian AOI attention
```

Study flow: positioning guide (distance 45-70 cm, centred, facing the screen;
one-time "I'm at 50 cm" distance correction per webcam) → camera exposure and
white balance locked → 13 dwell-captured calibration dots (worst ones
re-measured) → 8 held-out accuracy dots (a missed dot is re-checked) →
recording, with a drift check every 20 s or immediately when the head moves.
Drift is corrected retroactively when gradual; samples after a head movement
are dropped. Recording is blocked when accuracy is poor.

Realistic accuracy: about 1.5-3° (60-120 px on a laptop) after calibration.
Large elements are measured reliably; elements smaller than the session's
error are flagged "approximate" instead of reported with false precision.
Click **↓ Report** after each session and run `python tools/gaze_benchmark.py <folder>`
to get pooled accuracy across participants.

## Research and evaluation tools

| Command | Output |
|---|---|
| `python training/evaluate.py` | Per-class precision, recall, mAP@0.5 and mAP@0.5:0.95 of the detector on the validation split (`reports/detector_eval.md`). |
| `python training/dataset_report.py` | Images and boxes per class and split, box sizes, label problems. |
| `python tools/validate_saliency.py <csv-folder>` | Correlation and error between saliency-estimated and gaze-measured attention across sessions. |
| `python tools/gaze_benchmark.py <report-folder>` | Pooled eye-tracker accuracy from study reports. |
| `python training/train.py --epochs 100 --copy-weights` | Retrain the detector. |

## Project structure

```
api/main.py            FastAPI: analysis, gaze stream, chat, history, PDF
analysis/              detection, saliency, scoring (PES v2), readability,
                       recommendations, fixations, gaze parsing, chat, store
gaze_engine/           webcam gaze engine (backend, calibration, session, filters)
frontend/              React + Vite app (src/components/*)
webgazer_choose_poster.html   study capture page (served at /api/gaze/capture)
config/                settings.py (env), scoring.yaml (PES weights and bands)
models/                YOLO wrapper + weights
training/              train, evaluate, dataset report, split, inference
tools/                 gaze benchmark, saliency validation
tests/                 pytest suite (run: python -m pytest -q)
legacy/                old Streamlit app and CLI, kept for reference
custom_eye_tracker/    superseded standalone iris tracker (reference only)
```

## Tests and CI

`python -m pytest -q` runs the backend suite (no GPU, webcam or model downloads
needed; the heavy models are faked). GitHub Actions runs the tests with
CPU PyTorch, builds the frontend and the Docker image, and pushes the image
only when `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` secrets are set.

## Known limitations

- The detector was trained on 63 images (11 validation); CTA and Price are the
  rarest classes. More labelled posters will help more than any code change.
- No held-out test split yet, so detector numbers are validation numbers.
- Readability uses two-colour clustering inside each box; photos or gradients
  behind text make the contrast estimate approximate.
- Webcam eye tracking cannot match an infrared tracker; treat small-element
  attention as approximate.
