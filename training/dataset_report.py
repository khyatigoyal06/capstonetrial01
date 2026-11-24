"""Dataset health check: images and boxes per split and class, box sizes and label problems.

    python training/dataset_report.py

Use it before and after adding images to see whether the rare classes (CTA,
Price) are catching up.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CLASS_NAMES, DATASET_DIR, SUPPORTED_EXTENSIONS  # noqa: E402


def analyse(dataset_dir: Path = DATASET_DIR) -> dict:
    report: dict = {"splits": {}, "problems": []}
    for split in ("train", "valid", "test"):
        image_dir, label_dir = dataset_dir / split / "images", dataset_dir / split / "labels"
        if not image_dir.is_dir():
            continue
        images = [p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        boxes: Counter = Counter()
        images_with: defaultdict = defaultdict(int)
        areas: defaultdict = defaultdict(list)
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                report["problems"].append(f"{split}/{image.name}: no label file")
                continue
            seen = set()
            for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    report["problems"].append(f"{split}/{label.name}:{number}: expected 5 values, got {len(parts)}")
                    continue
                cls = CLASS_NAMES.get(int(parts[0]), f"class_{parts[0]}")
                xc, yc, w, h = map(float, parts[1:])
                if xc - w / 2 < -0.01 or yc - h / 2 < -0.01 or xc + w / 2 > 1.01 or yc + h / 2 > 1.01:
                    report["problems"].append(f"{split}/{label.name}:{number}: {cls} box extends outside the image")
                boxes[cls] += 1
                areas[cls].append(w * h)
                seen.add(cls)
            for cls in seen:
                images_with[cls] += 1
        report["splits"][split] = {
            "images": len(images),
            "boxes": dict(boxes),
            "images_containing": dict(images_with),
            "median_box_area_pct": {cls: round(sorted(v)[len(v) // 2] * 100, 2) for cls, v in areas.items()},
        }
    return report


def main() -> int:
    report = analyse()
    for split, data in report["splits"].items():
        print(f"\n{split}: {data['images']} images")
        for cls in CLASS_NAMES.values():
            print(f"  {cls:9s} boxes={data['boxes'].get(cls, 0):4d}  images={data['images_containing'].get(cls, 0):4d}  "
                  f"median area={data['median_box_area_pct'].get(cls, 0):5.2f}%")
    if "test" not in report["splits"]:
        print("\nNo test split: final numbers are reported on the validation split only.")
    print(f"\n{len(report['problems'])} label problems")
    for problem in report["problems"][:30]:
        print("  " + problem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
