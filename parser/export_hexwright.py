"""Export a wargame-map-parser terrain classification into a Hexwright-ready fragment.

Hexwright (github.com/lerugray/hexwright) is the hand-correction editor for the
hex/hexside data this parser produces. Its **Import WMP draft** button ingests a
JSON file of the shape ``{"terrain": {"CCRR": "<terrain>"}}`` and marks every hex
as an unconfirmed *draft* the operator then refines. WMP's
``ReferenceClassifier.classify_all()`` already emits ``{"CCRR": "<class>"}`` in the
same flat-top even-q CCRR addressing Hexwright uses, so the only bridge is a small
terrain-vocabulary map (WMP ``forest``/``lake`` -> Hexwright ``woods``/``water``).
Hexwright applies the same aliases on import, so this module is a convenience: it
lets a WMP run drop a clean, ready-to-import file in one step.

Pipeline: scan -> WMP classify (rough guess) -> Hexwright refine -> canonical export.

Usage
-----
As a library, right after a classification run::

    from parser import ReferenceClassifier, load_image, fit_from_anchors
    from parser.export_hexwright import write_hexwright
    terrain = clf.classify_all(arr, all_hexcodes)   # {"0803": "forest", ...}
    write_hexwright(terrain, "gota-terrain.hexwright.json")

As a CLI, to convert an already-saved WMP terrain dump::

    python -m parser.export_hexwright wmp-terrain.json -o gota-terrain.hexwright.json
"""
from __future__ import annotations

import argparse
import json

# WMP terrain class -> Hexwright terrain key. Unlisted classes pass through
# unchanged (Hexwright's palette + its own import aliases cover the rest).
TERRAIN_MAP = {
    "forest": "woods",
    "lake": "water",
}

# WMP classes that are in-hex POINT features in Hexwright, not hex-fill terrain.
# classify_all rarely emits these as a fill class; if it does, they pass through
# and can be re-typed by hand in Hexwright's inspector.
POINT_CLASSES = frozenset({"town", "city", "fortress"})


def to_hexwright(terrain: dict) -> dict:
    """Map a WMP terrain dict (``{CCRR: class}`` or ``{"terrain": {...}}``) to a
    Hexwright project fragment (``{"terrain": {CCRR: terrainKey}}``), applying the
    terrain-vocabulary bridge."""
    src = terrain.get("terrain", terrain) if isinstance(terrain, dict) else {}
    out = {}
    for code, cls in src.items():
        if not isinstance(cls, str):
            continue
        out[code] = TERRAIN_MAP.get(cls, cls)
    return {"terrain": out}


def write_hexwright(terrain: dict, path: str) -> int:
    """Write a Hexwright-ready terrain fragment to ``path``; return the hex count."""
    data = to_hexwright(terrain)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    return len(data["terrain"])


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a WMP terrain classification to a Hexwright-importable JSON."
    )
    ap.add_argument("input", help='WMP terrain JSON ({CCRR: class} or {"terrain": {...}})')
    ap.add_argument("-o", "--output", default=None,
                    help="output path (default: <input>.hexwright.json)")
    args = ap.parse_args()
    with open(args.input) as fh:
        terrain = json.load(fh)
    out = args.output or (args.input.rsplit(".", 1)[0] + ".hexwright.json")
    n = write_hexwright(terrain, out)
    print(f"Wrote {n} hexes -> {out}  (import in Hexwright via 'Import WMP draft')")


if __name__ == "__main__":
    _main()
