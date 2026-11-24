"""Does the saliency estimate (Poster Review mode) predict where people really look?

Collect gaze-engine sessions for several posters (CSV from the study page),
put them in one folder and run:

    python tools/validate_saliency.py path/to/csvs/ [--images extra/poster/folder]

For each session the poster is found by name in the dataset (or --images),
AOIs come from the CSV, measured attention from fixations, and predicted
attention from the saliency model on the same AOIs. Output: per-session and
pooled correlation and mean absolute error, plus reports/saliency_validation.csv.

A pooled Pearson r near 1 means saliency is a usable stand-in for eye
tracking on these posters; near 0 means Poster Review scores should not be
trusted without gaze data.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.aoi import build_aoi_record  # noqa: E402
from analysis.gaze_session import analyze_engine_session, load_engine_session  # noqa: E402
from analysis.poster import read_image  # noqa: E402
from analysis.saliency import attention_percentages_from_saliency  # noqa: E402
from config.settings import CLASS_NAMES, DATASET_DIR, PROJECT_ROOT  # noqa: E402


def find_poster(name: str, extra: list[Path]) -> Path | None:
    for folder in extra + [DATASET_DIR / s / "images" for s in ("train", "valid", "test")]:
        if (folder / name).is_file():
            return folder / name
    return None


def records_from_csv(raw: bytes, image_shape):
    first = next(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    aois = json.loads(first.get("aoi_bounds_json") or "[]")
    sw, sh = float(first.get("poster_width") or 0), float(first.get("poster_height") or 0)
    h, w = image_shape[:2]
    sx, sy = (w / sw if sw else 1.0), (h / sh if sh else 1.0)
    name_to_id = {v.casefold(): k for k, v in CLASS_NAMES.items()}
    out = []
    for a in aois:
        x1, y1, x2, y2 = a["x1"] * sx, a["y1"] * sy, a["x2"] * sx, a["y2"] * sy
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(build_aoi_record(name_to_id.get(str(a["label"]).casefold(), -1), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                                    (x2 - x1) / w, (y2 - y1) / h, w, h, class_names={**CLASS_NAMES, -1: a["label"]}))
    return first.get("poster", ""), out


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 and a.std() > 0 and b.std() > 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--images", type=Path, action="append", default=[])
    args = parser.parse_args()

    rows, pooled_m, pooled_p = [], [], []
    for path in sorted(args.folder.glob("*.csv")):
        raw = path.read_bytes()
        session = load_engine_session(raw)
        if session is None:
            print(f"skip {path.name}: not a gaze-engine export")
            continue
        poster_name = next(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))).get("poster", "")
        poster_path = find_poster(poster_name, args.images)
        if poster_path is None:
            print(f"skip {path.name}: poster {poster_name!r} not found (use --images)")
            continue
        image = read_image(poster_path)
        _, records = records_from_csv(raw, image.shape)
        attention_ms, _stats, _points = analyze_engine_session(session, records, image.shape)
        total = sum(attention_ms.values()) or 1.0
        measured = {k: v / total * 100 for k, v in attention_ms.items()}
        predicted = attention_percentages_from_saliency(image, records)
        labels = sorted(set(measured) | set(predicted))
        m = np.array([measured.get(l, 0.0) for l in labels])
        p = np.array([predicted.get(l, 0.0) for l in labels])
        pooled_m.extend(m)
        pooled_p.extend(p)
        r = pearson(m, p)
        print(f"{path.name[:48]:48s} r={'-' if r is None else f'{r:+.2f}'}  MAE={np.mean(np.abs(m - p)):5.1f} pts  "
              f"(accuracy {session.median_error_screen_px:.0f}px, {session.quality})")
        for label in labels:
            rows.append({"session": path.name, "poster": poster_name, "element": label,
                         "measured_pct": round(measured.get(label, 0.0), 2), "predicted_pct": round(predicted.get(label, 0.0), 2)})

    if not rows:
        print("No usable sessions.")
        return 1
    m, p = np.array(pooled_m), np.array(pooled_p)
    r = pearson(m, p)
    print(f"\nPooled over {len({r_['session'] for r_ in rows})} sessions, {len(rows)} element shares: "
          f"Pearson r = {'n/a' if r is None else f'{r:.2f}'}, mean absolute error = {np.mean(np.abs(m - p)):.1f} percentage points")
    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    with open(out / "saliency_validation.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out / 'saliency_validation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
