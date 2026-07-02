"""wargame-map-parser — extract per-hex terrain from a printed wargame map scan.

Pipeline:
  1. seams.fix_sheets      de-duplicate a multi-sheet board (shared overlap band)
  2. hexgrid.fit_from_anchors   calibrate CCRR<->pixel from a few read hex numbers
     (then hexgrid.verify_against_printed   confirm it isn't a uniform off-by-one)
  3. classify.ReferenceClassifier   nearest-exemplar terrain (colour+texture+morphology)
  4. overlay.draw_terrain / draw_centers   LOOK at the result before trusting it
  5. hexside_snap.HexsideSnapper   HMM/Viterbi map-matching of hand-traced linear
     features (rivers, ridges, impassible terrain) onto the hex-lattice hexside graph
     -- `from parser.hexside_snap import HexsideSnapper, SnapParams, snap_traces`

Method credit: Ray Weiss (reference-hex matching; bulbs-vs-lines morphology;
hexside-terrain needs an edge layer). Hexside-snap: spec by Fugu (fugu-ultra,
spec-only mode), validated on GotA 2026-07-02. See README.md and SKILL.md.

``hexside_snap`` requires scipy + scikit-image (see requirements.txt); the
rest of this package needs only numpy + Pillow, so hexside_snap is a
separate import (not re-exported at package level) to keep the base install
light.
"""
from .hexgrid import (HexGrid, fit_from_anchors, verify_against_printed,
                      flat_top_geometry_ratio, check_geometry_ratio,
                      parse_ccrr, to_ccrr)
from .classify import ReferenceClassifier, hex_features, load_image
from .seams import detect_overlap, stitch, fix_sheets
from .overlay import draw_terrain, draw_centers, TERRAIN_COLORS

__version__ = "0.1.0"
__all__ = [
    "HexGrid", "fit_from_anchors", "verify_against_printed",
    "flat_top_geometry_ratio", "check_geometry_ratio",
    "parse_ccrr", "to_ccrr",
    "ReferenceClassifier", "hex_features", "load_image",
    "detect_overlap", "stitch", "fix_sheets",
    "draw_terrain", "draw_centers", "TERRAIN_COLORS",
]
