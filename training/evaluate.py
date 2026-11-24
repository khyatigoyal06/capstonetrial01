"""Evaluate the poster-element detector on a labelled split.

    python training/evaluate.py                         # models/weights/best.pt on the valid split
    python training/evaluate.py --weights runs/detect/x/weights/best.pt --split test

Writes reports/detector_eval.json and a Markdown table you can paste into the
capstone report: per-class precision, recall, mAP@0.5 and mAP@0.5:0.95.
With only 11 validation images (and few CTA/Price boxes), per-class numbers
have wide uncertainty - say so when you report them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATASET_YAML, DEFAULT_MODEL_WEIGHTS, PROJECT_ROOT  # noqa: E402
from training.train import write_resolved_dataset_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_MODEL_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DATASET_YAML)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args()


def summarise(metrics, names: dict[int, str]) -> dict:
    box = metrics.box
    per_class = {}
    class_ids = list(getattr(box, "ap_class_index", range(len(names))))
    for position, class_id in enumerate(class_ids):
        per_class[names[int(class_id)]] = {
            "precision": round(float(box.p[position]), 4),
            "recall": round(float(box.r[position]), 4),
            "mAP50": round(float(box.ap50[position]), 4),
            "mAP50_95": round(float(box.ap[position]), 4),
        }
    return {
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
        },
        "per_class": per_class,
    }


def to_markdown(report: dict) -> str:
    lines = ["| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|"]
    rows = list(report["per_class"].items()) + [("**All**", report["overall"])]
    for name, m in rows:
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['mAP50']:.3f} | {m['mAP50_95']:.3f} |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        print(f"Weights not found: {args.weights}")
        return 1
    from ultralytics import YOLO

    resolved = write_resolved_dataset_yaml(args.data, args.out)
    model = YOLO(str(args.weights))
    metrics = model.val(data=str(resolved), split=args.split, imgsz=args.imgsz, plots=False, verbose=False)
    report = {"weights": str(args.weights), "split": args.split, **summarise(metrics, model.names)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "detector_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = to_markdown(report)
    (args.out / "detector_eval.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nSaved to {args.out / 'detector_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
