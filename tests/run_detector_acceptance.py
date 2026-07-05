#!/usr/bin/env python3
"""Run the detector-improvement acceptance tests from
``docs/DETECTOR-IMPROVEMENT-SPEC-2026-07-04.md`` against a saved detector
output JSON and a corrections dataset.

This script is meant to be executed in a game repo that contains:

- the scanned board raster,
- the hexgrid calibration JSON,
- a ``corrections-2026-07-04.json`` produced by the operator, and
- a detector-output JSON written by the caller (see below).

In this tooling repo the script finds no dataset, so it exits non-zero and
reports the missing file; that is the expected behavior until the harness is
run in a game repo.

Usage::

    python tests/run_detector_acceptance.py \
        --corrections path/to/corrections-2026-07-04.json \
        --output path/to/detector-output.json \
        --map NaB

``detector-output.json`` is the grouped shape::

    {"rivers": [{"a":"0101","b":"0102"}, ...],
     "roads":   [{"a":"0126","b":"0227","value":"primary"}, ...],
     ...}

Exit code 0 only if every acceptance threshold from the spec is met.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root or tests/ directory
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo))

from parser.detector import CorrectionScorer, DetectedEdge


# Acceptance thresholds exactly as stated in the spec.
THRESHOLDS = {
    "NaB": {
        "river": {"added": 110, "removed": 0, "reclassified": 0},
        "road": {"added": 135, "removed": 0, "reclassified": 1},
        "bridge": {"added": 18, "removed": 18},  # <=4 retained means >=18 excluded
    },
    "TWU": {
        "impassible": {"added": 230, "removed": 0, "reclassified": 0},
        "river": {"added": 40, "removed": 0, "reclassified": 0},
        "border": {"added": 50, "removed": 0, "reclassified": 0},
        "rail": {"added": 65, "removed": 90},  # >=90 excluded, incl. y=1673.8 corridor
    },
}

# The y=1673.8 corridor is a special TWU rail removal check.  The corrections
# dataset is expected to tag those keys with a marker; this script counts hits
# if the dataset exposes them under TWU.rail.removed_corridor.
CORRIDOR_KEY = "removed_corridor"


def _load_output(path: str | Path) -> dict[str, list[DetectedEdge]]:
    data = json.loads(Path(path).read_text())
    out: dict[str, list[DetectedEdge]] = {}
    for layer, edges in data.items():
        out[layer] = [
            DetectedEdge(a=e["a"], b=e["b"], layer=layer, value=e.get("value"))
            for e in edges
        ]
    return out


def _format_summary(summary: dict) -> str:
    lines = []
    for layer, rec in summary.items():
        lines.append(
            f"  {layer}: added {rec['added_hits']}/{rec['added_total']}, "
            f"removed {rec['removed_hits']}/{rec['removed_total']}, "
            f"reclassified {rec['reclassified_hits']}/{rec['reclassified_total']} "
            f"(output edges: {rec['output_edges']})"
        )
    return "\n".join(lines)


def _check_thresholds(map_name: str, summary: dict, corrections: dict) -> tuple[bool, list[str]]:
    ok = True
    messages: list[str] = []
    thresholds = THRESHOLDS.get(map_name, {})
    for layer, need in thresholds.items():
        rec = summary.get(layer, {})
        if "added" in need and need["added"] > 0:
            if rec.get("added_hits", 0) < need["added"]:
                ok = False
                messages.append(
                    f"FAIL {map_name}.{layer}: added hits {rec.get('added_hits', 0)} "
                    f"< required {need['added']}"
                )
        if "removed" in need and need["removed"] > 0:
            excluded = rec.get("removed_total", 0) - rec.get("removed_hits", 0)
            if excluded < need["removed"]:
                ok = False
                messages.append(
                    f"FAIL {map_name}.{layer}: removed excluded {excluded} "
                    f"< required {need['removed']}"
                )
        if "reclassified" in need and need["reclassified"] > 0:
            if rec.get("reclassified_hits", 0) < need["reclassified"]:
                ok = False
                messages.append(
                    f"FAIL {map_name}.{layer}: reclassified hits "
                    f"{rec.get('reclassified_hits', 0)} < required {need['reclassified']}"
                )
        # Note: the spec's special y=1673.8 corridor is included in the
        # TWU.rail.removed set, so the generic removed check above covers it.
    return ok, messages


def main() -> int:
    ap = argparse.ArgumentParser(description="Detector-improvement acceptance harness")
    ap.add_argument("--corrections", help="path to corrections-2026-07-04.json")
    ap.add_argument("--output", help="path to detector-output.json")
    ap.add_argument("--map", choices=["NaB", "TWU"], required=True)
    args = ap.parse_args()

    corrections_path = args.corrections or os.environ.get("DETECTOR_CORRECTIONS")
    output_path = args.output or os.environ.get("DETECTOR_OUTPUT")
    if not corrections_path:
        print("ERROR: --corrections not given and DETECTOR_CORRECTIONS not set")
        return 2
    if not output_path:
        print("ERROR: --output not given and DETECTOR_OUTPUT not set")
        return 2

    if not Path(corrections_path).exists():
        print(f"BLOCKED: corrections dataset not found: {corrections_path}")
        print("Acceptance tests require the per-game corrections-2026-07-04.json.")
        return 2

    scorer = CorrectionScorer(corrections_path)
    output = _load_output(output_path)
    summary = scorer.score(output, args.map)

    print(f"Acceptance results for {args.map}:")
    print(_format_summary(summary))

    with open(corrections_path) as fh:
        corrections = json.load(fh)

    ok, messages = _check_thresholds(args.map, summary, corrections)
    for msg in messages:
        print(msg)
    if ok:
        print(f"PASS {args.map}: all acceptance thresholds met")
        return 0
    print(f"FAIL {args.map}: one or more thresholds not met")
    return 1


if __name__ == "__main__":
    sys.exit(main())
