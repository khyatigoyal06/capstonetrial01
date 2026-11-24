"""Summarise gaze accuracy across recorded sessions.

Each study session can download a ``*_gaze_report.json`` (the "↓ Report"
button). Collect them from several participants / lighting conditions and run:

    python tools/gaze_benchmark.py path/to/reports/            # or explicit files

The numbers come from held-out validation dots that were never used for
fitting, so this is the honest accuracy of the tracker on your setup. Use it to
compare changes (camera, model, lighting) instead of eyeballing the gaze dot.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(paths: list[Path]) -> list[tuple[Path, dict]]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.glob("*_gaze_report.json")) if path.is_dir() else [path])
    reports = []
    for file in files:
        try:
            reports.append((file, json.loads(file.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip {file}: {exc}")
    return reports


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[50, 100, 150])
    args = parser.parse_args()

    reports = _load(args.paths)
    if not reports:
        print("No reports found.")
        return 1

    pooled: list[float] = []
    print(f"{'session':40s} {'model':8s} {'median':>7s} {'p90':>7s} {'deg':>5s} {'quality':>8s} {'drift fixes':>11s}")
    for file, report in reports:
        validation = report.get("validation") or {}
        errors = list((validation.get("errors") or {}).values())
        pooled.extend(float(e) for e in errors)
        drift = [d for d in report.get("drift_checks") or [] if d.get("action") == "corrected"]
        deg = validation.get("approx_median_deg")
        print(
            f"{file.stem[:40]:40s} {str(report.get('gaze_model'))[:8]:8s} "
            f"{validation.get('median_px', '-')!s:>7s} {validation.get('p90_px', '-')!s:>7s} "
            f"{('%.1f' % deg) if isinstance(deg, (int, float)) else '-':>5s} {str(validation.get('quality', '-')):>8s} {len(drift):>11d}"
        )

    if pooled:
        print(f"\nPooled over {len(pooled)} validation targets from {len(reports)} sessions:")
        print(f"  median {statistics.median(pooled):.1f} px   P90 {_percentile(pooled, 90):.1f} px   mean {statistics.mean(pooled):.1f} px")
        for threshold in args.thresholds:
            share = sum(e <= threshold for e in pooled) / len(pooled)
            print(f"  within {threshold:.0f} px: {share:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
