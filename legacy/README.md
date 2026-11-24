# Legacy tools

Kept for reference; the maintained app is the React frontend + FastAPI (`api/`).

| File | What it was |
|---|---|
| `aoi_visualizer.py` | Original 2,300-line CLI: AOI preview, Tobii/WebGazer import, v1 PES, capture-page generator. Its parsing code now lives in `analysis/gaze.py`. **Do not use `--webgazer-gallery`**: it would overwrite the maintained `webgazer_choose_poster.html`. |
| `streamlit_app/app.py` | First web prototype (Streamlit). Run with `streamlit run legacy/streamlit_app/app.py` after `pip install -r requirements-dev.txt`. |
| `run_server.py` | Wrapper that served the Streamlit app in an iframe for the Playwright UI test. |
| `tests_ui/` | Playwright test for the Streamlit app (not run in CI). |
