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
    monkeypatch.setattr(main, "detect_poster_elements", lambda _image, **_kwargs: (records, "model"))
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


def test_poster_analysis_is_stored_reportable_and_listed(monkeypatch, tmp_path):
    from analysis.store import AnalysisStore

    monkeypatch.setattr(main, "_store", AnalysisStore(tmp_path / "db.sqlite3"))
    image = np.full((300, 200, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (40, 220), (160, 260), (0, 0, 200), -1)
    ok, encoded = cv2.imencode(".png", image)
    records = [build_aoi_record(3, 0.5, 0.35, 0.6, 0.4, 200, 300), build_aoi_record(0, 0.5, 0.8, 0.6, 0.13, 200, 300)]
    monkeypatch.setattr(main, "detect_poster_elements", lambda _image, **_kwargs: (records, "model"))
    client = TestClient(main.app)
    payload = client.post("/api/analyze-poster", files={"file": ("p.png", encoded.tobytes(), "image/png")}).json()
    assert payload["pes"]["version"] == 2 and "Readability" in payload["pes"]["components"]
    assert payload["recommendations"] and "impact" in payload["recommendations"][0]

    listed = client.get("/api/analyses").json()["analyses"]
    assert listed[0]["analysis_id"] == payload["analysis_id"]
    again = client.get(f"/api/analyses/{payload['analysis_id']}").json()
    assert again["pes"]["score"] == payload["pes"]["score"] and again["annotated_image"].startswith("data:image/jpeg")
    pdf = client.get(f"/api/analyses/{payload['analysis_id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert client.delete(f"/api/analyses/{payload['analysis_id']}").json() == {"deleted": True}


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0.001)
    response = TestClient(main.app).post("/api/analyze-poster", files={"file": ("big.png", b"x" * 5000, "image/png")})
    assert response.status_code == 413


def test_gallery_aois_come_from_label_files():
    html = TestClient(main.app).get("/api/gaze/capture").text
    import json as _json
    import re as _re

    posters = _json.loads(_re.search(r"const POSTERS = (\[.*?\]);", html, _re.S).group(1))
    for poster in posters:
        for aoi in poster["aois"]:
            assert 0 <= aoi["x1"] < aoi["x2"] <= poster["width"]
            assert 0 <= aoi["y1"] < aoi["y2"] <= poster["height"]
