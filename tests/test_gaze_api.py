import cv2
import numpy as np
from fastapi.testclient import TestClient

from analysis.aoi import build_aoi_record
from api import main


def test_gaze_upload_returns_shared_analysis_payload(monkeypatch):
    image = np.full((120, 160, 3), 220, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    records = [
        build_aoi_record(3, 0.25, 0.25, 0.3, 0.3, 160, 120, confidence=0.9),
        build_aoi_record(0, 0.75, 0.75, 0.25, 0.2, 160, 120, confidence=0.9),
    ]
    monkeypatch.setattr(main, "detect_poster_elements", lambda _image, conf: (records, "model"))
    client = TestClient(main.app)
    gaze_csv = (
        "timestamp_ms,gaze_x,gaze_y,duration_ms,aoi_hit\n"
        "100,40,30,100,Product\n"
        "200,120,90,100,CTA\n"
    )

    response = client.post(
        "/api/analyze-gaze",
        files={
            "image_file": ("poster.png", encoded.tobytes(), "image/png"),
            "gaze_file": ("session.csv", gaze_csv.encode(), "text/csv"),
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["attention_source"] == "webgazer_gaze"
    assert payload["gaze_sample_count"] == 2
    assert payload["pes"]["mode"] == "webgazer_gaze"
    assert sum(payload["pes"]["attention_percentages"].values()) == 100
    assert payload["saliency_image"].startswith("data:image/png;base64,")


def test_embedded_webgazer_page_is_available():
    client = TestClient(main.app)
    response = client.get("/api/gaze/capture")
    assert response.status_code == 200
    assert '<base href="/">' in response.text
    if (main.FRONTEND_DIST / "index.html").is_file():
        assert client.get("/").status_code == 200
        asset = next((main.FRONTEND_DIST / "assets").iterdir())
        assert client.get(f"/assets/{asset.name}").status_code == 200
    poster = next(main.DATASET_IMAGE_DIR.glob("*.jpg"))
    assert client.get(f"/Capstone.yolov8/train/images/{poster.name}").status_code == 200
