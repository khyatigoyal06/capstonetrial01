# Architecture

```
React app (frontend/)                       Capture page (webgazer_choose_poster.html)
  Poster review / Gaze file / History         positioning → calibration → validation → recording
        │  REST                                         │ WebSocket (JPEG frames + control)
        ▼                                               ▼
FastAPI (api/main.py) ─────────────────────────────────────────────────────────────
  /api/analyze-poster   detect → saliency attention → PES v2 → recommendations → store
  /api/analyze-gaze     AOIs (study/labels/detector) → fixations → measured attention
                        → PES v2 → recommendations → saliency comparison → store
  /api/gaze/stream      gaze_engine.GazeSession per connection
  /api/analyses[...]    history list / reopen / delete / PDF report
  /api/chat             grounded Groq chat on a stored analysis
        │
        ├─ analysis/detection.py     cached YOLOv8 detector + same-class duplicate merge
        ├─ analysis/saliency.py      spectral residual on CIE Lab + centre bias
        ├─ analysis/scoring.py       PES v2 (config/scoring.yaml)
        ├─ analysis/readability.py   WCAG contrast + size of text boxes
        ├─ analysis/recommendations.py  impact-ranked fixes
        ├─ analysis/fixations.py     I-DT fixations, Gaussian AOI attention, fixation map
        ├─ analysis/gaze_session.py  engine CSV → attention, quality, reliability
        ├─ analysis/store.py         SQLite history (payload JSON + JPEG blobs)
        └─ gaze_engine/              ptgaze backend, calibration, One-Euro filter, session
```

## Data flow of one gaze study

1. The page streams 720p frames; the server returns face box, head angles,
   distance and (after calibration) a filtered screen point per frame.
2. Calibration targets are captured by dwell; the server summarises each
   target robustly and fits the per-user mapping by leave-one-out CV.
3. Validation targets measure accuracy on locations never used for fitting.
4. During recording the page logs samples with the session's error, display
   scale and head pose; drift checks correct or invalidate past samples.
5. The CSV goes to `/api/analyze-gaze`, which detects fixations and spreads
   each one by the measured error over the AOIs.

Historical planning documents are in `docs/archive/`.
