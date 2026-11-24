import json
import struct

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api import main
from gaze_engine.observation import FrameObservation


class FakeBackend:
    """Returns the camera-plane point encoded in the frame's pixel colour."""

    def warm_up(self):
        pass

    def process(self, image):
        b, g, _ = image[0, 0].astype(float)
        return FrameObservation(face_detected=True, point_m=((b - 128) / 1000, (g - 128) / 1000),
                                face_distance_m=0.6, eye_openness=0.3)


def _frame_bytes(t_ms, point):
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, :] = (int(point[0] * 1000 + 128), int(point[1] * 1000 + 128), 0)
    ok, jpg = cv2.imencode(".png", img)  # lossless so the encoded point survives
    return struct.pack("<d", t_ms) + jpg.tobytes()


def test_gaze_stream_calibrates_and_streams_screen_points(monkeypatch):
    monkeypatch.setattr(main, "get_gaze_backend", lambda: FakeBackend())
    client = TestClient(main.app)
    targets = [(100, 100), (500, 100), (900, 100), (100, 400), (500, 400), (900, 400), (100, 700), (500, 700), (900, 700)]
    to_plane = lambda p: ((p[0] - 500) / 10000, (p[1] - 400) / 10000)  # 1 px = 0.1 mm
    with client.websocket_connect("/api/gaze/stream") as ws:
        assert ws.receive_json()["stage"] == "loading_model"
        assert ws.receive_json()["stage"] == "model_ready"
        ws.send_text(json.dumps({"type": "hello", "screen": {"width": 1000, "height": 800}}))
        assert ws.receive_json()["type"] == "ready"
        clock = 0.0
        for i, target in enumerate(targets):
            ws.send_text(json.dumps({"type": "target_start", "phase": "calibration", "id": str(i), "x": target[0], "y": target[1], "t": clock}))
            for k in range(30):
                ws.send_bytes(_frame_bytes(clock + 60 * k, to_plane(target)))
                assert ws.receive_json()["type"] == "gaze"
            ws.send_text(json.dumps({"type": "target_end", "id": str(i)}))
            assert ws.receive_json()["accepted"]
            clock += 3000
        ws.send_text(json.dumps({"type": "fit"}))
        fit = ws.receive_json()
        assert fit["ok"], fit
        ws.send_bytes(_frame_bytes(clock, to_plane((300, 250))))
        gaze = ws.receive_json()
        assert gaze["valid"] and abs(gaze["x"] - 300) < 15 and abs(gaze["y"] - 250) < 15


def test_engine_status_endpoint_reports_availability():
    payload = TestClient(main.app).get("/api/gaze/engine").json()
    assert set(payload) == {"available", "reason", "mode"}


def test_engine_csv_uses_fixations_and_reports_quality(monkeypatch):
    image = np.full((400, 400, 3), 230, np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    aois = {"width": 400, "height": 400, "aois": [
        {"label": "Product", "x1": 100, "y1": 100, "x2": 300, "y2": 300},
        {"label": "CTA", "x1": 150, "y1": 360, "x2": 250, "y2": 380},
    ]}
    header = ["timestamp_ms", "gaze_x", "gaze_y", "gaze_model", "gaze_valid", "poster", "poster_width", "poster_height",
              "aoi_bounds_json", "session_median_error_px", "session_p90_error_px", "session_quality", "poster_display_scale"]
    rows = []
    for i in range(40):
        rows.append([i * 50, 200, 200, "engine", 1, "p", 400, 400, json.dumps(aois), 40, 70, "good", 1.0])
    csv_text = ",".join(header) + "\n" + "\n".join(
        ",".join('"' + str(v).replace('"', '""') + '"' for v in row) for row in rows)
    response = TestClient(main.app).post(
        "/api/analyze-gaze",
        files={"image_file": ("p.png", encoded.tobytes(), "image/png"),
               "gaze_file": ("s.csv", csv_text.encode(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["attention_source"] == "gaze_engine_fixations"
    assert payload["gaze_stats"]["fixations"] == 1
    assert payload["pes"]["attention_percentages"]["Product"] > 90
    assert payload["gaze_quality"]["unreliable_aois"] == ["CTA"]
    assert any("CTA" in item for item in payload["pes"]["insights"])
