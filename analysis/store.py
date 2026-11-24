"""SQLite store for analyses, so chat, PDF reports and history survive restarts.

Images are kept as JPEG blobs next to a JSON payload without the (large)
base64 images; `get` rebuilds the data URIs the frontend expects.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config.settings import ANALYSIS_DB_MAX_ROWS, ANALYSIS_DB_PATH

IMAGE_FIELDS = ("annotated_image", "saliency_image")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    kind TEXT NOT NULL,
    filename TEXT,
    score REAL,
    category TEXT,
    payload TEXT NOT NULL,
    poster BLOB,
    annotated BLOB,
    heatmap BLOB
)
"""


def _jpeg(image_bgr: np.ndarray | None, quality: int = 85) -> bytes | None:
    if image_bgr is None:
        return None
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None


def _data_uri_to_bgr(uri: str | None) -> np.ndarray | None:
    if not uri or "," not in uri:
        return None
    data = np.frombuffer(base64.b64decode(uri.split(",", 1)[1]), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _blob_to_uri(blob: bytes | None) -> str | None:
    return None if blob is None else "data:image/jpeg;base64," + base64.b64encode(blob).decode("ascii")


class AnalysisStore:
    def __init__(self, path: Path | str = ANALYSIS_DB_PATH, max_rows: int = ANALYSIS_DB_MAX_ROWS) -> None:
        self.path = Path(path)
        self.max_rows = max_rows
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def save(self, payload: dict[str, Any], poster_bgr: np.ndarray) -> None:
        stripped = {k: v for k, v in payload.items() if k not in IMAGE_FIELDS}
        pes = payload.get("pes") or {}
        row = (
            payload["analysis_id"], time.time(), payload.get("attention_source", "saliency"),
            payload.get("filename"), pes.get("score"), pes.get("category"), json.dumps(stripped),
            _jpeg(poster_bgr, 92),
            _jpeg(_data_uri_to_bgr(payload.get("annotated_image"))),
            _jpeg(_data_uri_to_bgr(payload.get("saliency_image"))),
        )
        with self._lock, self._connect() as db:
            db.execute("INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?)", row)
            db.execute(
                "DELETE FROM analyses WHERE id NOT IN (SELECT id FROM analyses ORDER BY created_at DESC LIMIT ?)",
                (self.max_rows,),
            )

    def get(self, analysis_id: str) -> tuple[dict[str, Any], np.ndarray | None] | None:
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT payload, poster, annotated, heatmap, created_at FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        payload["annotated_image"] = _blob_to_uri(row[2])
        payload["saliency_image"] = _blob_to_uri(row[3])
        payload["created_at"] = row[4]
        poster = cv2.imdecode(np.frombuffer(row[1], np.uint8), cv2.IMREAD_COLOR) if row[1] else None
        return payload, poster

    def list(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT id, created_at, kind, filename, score, category FROM analyses ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 200)),),
            ).fetchall()
        return [
            {"analysis_id": r[0], "created_at": r[1], "kind": r[2], "filename": r[3], "score": r[4], "category": r[5]}
            for r in rows
        ]

    def delete(self, analysis_id: str) -> bool:
        with self._lock, self._connect() as db:
            return db.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,)).rowcount > 0
