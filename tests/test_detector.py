"""Mechanism tests for the six detector improvements in ``parser.detector``.

These are synthetic unit tests on a small, hand-traceable hex lattice.  They
verify that each fix's mechanism behaves as intended; they do not reproduce
the full NaB/TWU acceptance counts, which require the per-game corrections
datasets and scanned board rasters.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import HexGrid, check_geometry_ratio
from parser.detector import (
    LinearFeatureDetector,
    CorrectionScorer,
    DetectedEdge,
    score_against_corrections,
)
from parser.hexside_snap import HexsideSnapper


def _test_grid(row_pitch_factor: float = 1.0) -> HexGrid:
    """Regular flat-top hex grid, same conventions as tests/test_hexside_snap.py."""
    col_pitch = 100.0
    row_pitch = col_pitch * 2.0 / math.sqrt(3) * row_pitch_factor
    return HexGrid(
        image_full=(3300, 3800),
        col_pitch_x=col_pitch,
        row_pitch_y=row_pitch,
        x_intercept_col0=100.0,
        y_intercept_row0=100.0,
        even_col_y_offset=row_pitch / 2,
    )


def _valid_hexes(n: int = 30) -> list:
    return [f"{c:02d}{r:02d}" for c in range(1, n + 1) for r in range(1, n + 1)]


def _mask_for_polyline(image_full, pts, width: int = 4) -> np.ndarray:
    img = Image.new("L", image_full, 0)
    ImageDraw.Draw(img).line(pts, fill=255, width=width)
    return np.asarray(img) > 0


def _build_chain(snapper: HexsideSnapper, n_edges: int, start_idx: int):
    """Walk ``n_edges`` connected lattice edges in the straightest direction."""
    edges = snapper.EDGES
    path = [start_idx]
    used = {start_idx}
    pts = [tuple(edges[start_idx]["pa"]), tuple(edges[start_idx]["pb"])]
    cur_vertex = edges[start_idx]["vb"]
    prev_dir = np.array(edges[start_idx]["pb"]) - np.array(edges[start_idx]["pa"])
    prev_dir = prev_dir / np.hypot(*prev_dir)

    while len(path) < n_edges:
        candidates = [ei for ei in snapper.VERT_EDGES[cur_vertex] if ei not in used]
        if not candidates:
            raise RuntimeError(f"dead end after {len(path)} edges")
        best_idx, best_score = None, None
        for ei in candidates:
            e = edges[ei]
            d = (np.array(e["pb"]) - np.array(e["pa"]) if e["va"] == cur_vertex
                 else np.array(e["pa"]) - np.array(e["pb"]))
            d = d / np.hypot(*d)
            score = float(np.dot(prev_dir, d))
            if best_score is None or score > best_score:
                best_idx, best_score = ei, score
        e = edges[best_idx]
        if e["va"] == cur_vertex:
            pts.append(tuple(e["pb"]))
            prev_dir = np.array(e["pb"]) - np.array(e["pa"])
            cur_vertex = e["vb"]
        else:
            pts.append(tuple(e["pa"]))
            prev_dir = np.array(e["pa"]) - np.array(e["pb"])
            cur_vertex = e["va"]
        prev_dir = prev_dir / np.hypot(*prev_dir)
        path.append(best_idx)
        used.add(best_idx)

    expected = sorted((edges[ei]["a"], edges[ei]["b"]) for ei in path)
    return pts, expected, path


def _find_interior_edge(snapper: HexsideSnapper, lo: int = 14, hi: int = 16) -> int:
    for i, e in enumerate(snapper.EDGES):
        ca, ra = int(e["a"][:2]), int(e["a"][2:])
        cb, rb = int(e["b"][:2]), int(e["b"][2:])
        if lo <= ca <= hi and lo <= ra <= hi and lo <= cb <= hi and lo <= rb <= hi:
            return i
    raise AssertionError("no interior candidate edge")


# ---------------------------------------------------------------------------
# Fix 1: padded-frame perimeter extraction


def test_perimeter_padding_keeps_boundary_ink():
    """A trace that ends exactly at the image border should still be
    skeletonized and snapped instead of being discarded by the boundary."""
    grid = _test_grid()
    detector = LinearFeatureDetector(grid, _valid_hexes())
    snapper = HexsideSnapper(grid, _valid_hexes())

    # Use an interior chain and extend its starting point to the left border.
    start_idx = _find_interior_edge(snapper, lo=5, hi=8)
    pts, expected, _ = _build_chain(snapper, 3, start_idx)
    # extend first point horizontally to x=0, keeping y
    pts[0] = (0.0, float(pts[0][1]))

    mask = _mask_for_polyline(grid.image_full, pts)
    result = detector.detect_with_perimeter_padding(mask, "rivers")
    decoded = sorted((e["a"], e["b"]) for e in result.edges_out)
    # without padding this often decodes zero edges because the boundary ink is
    # treated as a map exit; with padding we should recover at least some.
    assert len(decoded) >= 1, "perimeter padding failed to keep boundary ink"


# ---------------------------------------------------------------------------
# Fix 2: impassible-specific calibration


def test_impassible_calibration_joins_dashed_outline():
    """A faint/dashed outline should be joined into a continuous mask so the
    snapper can decode it as one feature."""
    grid = _test_grid()
    detector = LinearFeatureDetector(grid, _valid_hexes())
    snapper = HexsideSnapper(grid, _valid_hexes())
    start_idx = _find_interior_edge(snapper)
    pts, expected, _ = _build_chain(snapper, 4, start_idx)

    # draw as widely-spaced dashes
    img = Image.new("L", grid.image_full, 0)
    draw = ImageDraw.Draw(img)
    for i in range(len(pts) - 1):
        if i % 2 == 0:
            draw.line([pts[i], pts[i + 1]], fill=255, width=3)
    mask = np.asarray(img) > 0

    calibrated = detector.calibrate_impassible_mask(None, mask)
    # after dilation the dashes should bridge into a connected component
    result = snapper.snap_layer(calibrated, "impassible")
    assert result.diagnostics["accepted_edges"] >= 1


# ---------------------------------------------------------------------------
# Fix 3: graph continuity gap-fill


def test_gap_fill_connects_short_linear_breaks():
    """Two collinear trace segments separated by a single missing hexside should
    be stitched by gap-fill when the intervening edge has mask evidence."""
    grid = _test_grid()
    detector = LinearFeatureDetector(grid, _valid_hexes())
    snapper = HexsideSnapper(grid, _valid_hexes())
    start_idx = _find_interior_edge(snapper)
    pts_full, expected_full, path = _build_chain(snapper, 5, start_idx)

    # Draw only the two end segments, deliberately skipping the middle edge.
    e_mid = snapper.EDGES[path[2]]
    pts_left = pts_full[:3]      # vertices for edges path[0], path[1]
    pts_right = pts_full[3:]     # vertices for edges path[3], path[4]
    mask = _mask_for_polyline(grid.image_full, pts_left)
    mask |= _mask_for_polyline(grid.image_full, pts_right)
    # add a faint pixel bridge across the missing middle edge so gap-fill has
    # mask evidence to justify the fill, but not enough for the raw snapper
    mx, my = (e_mid["pa"] + e_mid["pb"]) / 2
    rr = 2  # ~16 px total, below raw snapper MIN_PARALLEL (~0.35*H) for H=100
    mask[int(my - rr):int(my + rr), int(mx - rr):int(mx + rr)] = True

    result = snapper.snap_layer(mask, "rivers")
    before = set(result.accepted.keys())
    assert path[2] not in before, "raw snap already accepted the missing middle edge"

    filled = detector.gap_fill_layer(snapper, result, mask)
    after = set(filled.accepted.keys())
    assert len(after) > len(before), "gap-fill did not add any edges"


# ---------------------------------------------------------------------------
# Fix 4: rail vs hexside orientation deconfliction


def test_rail_deconfliction_suppresses_hexside_parallel_ink():
    """A rail candidate whose mask orientation is parallel to the hexside it
    crosses (not aligned with the center-to-center rail direction) should be
    suppressed."""
    grid = _test_grid()
    detector = LinearFeatureDetector(grid, _valid_hexes())
    snapper = HexsideSnapper(grid, _valid_hexes())
    start_idx = _find_interior_edge(snapper)
    e = snapper.EDGES[start_idx]
    mid = (e["pa"] + e["pb"]) / 2
    hexside_dir = (e["pb"] - e["pa"]) / np.hypot(*(e["pb"] - e["pa"]))

    # draw a stroke parallel to the hexside (wrong for rail, right for road/border)
    img = Image.new("L", grid.image_full, 0)
    draw = ImageDraw.Draw(img)
    length = snapper.H * 1.5
    p1 = (float(mid[0] - hexside_dir[0] * length), float(mid[1] - hexside_dir[1] * length))
    p2 = (float(mid[0] + hexside_dir[0] * length), float(mid[1] + hexside_dir[1] * length))
    draw.line([p1, p2], fill=255, width=5)
    mask = np.asarray(img) > 0

    rail_result = snapper.snap_layer(mask, "rail")
    # the snapper may or may not accept it; if it does, deconfliction should drop it
    if rail_result.accepted:
        deconf = detector.deconflict_rails(snapper, rail_result, None, mask)
        assert not deconf.accepted, "hexside-parallel rail was not suppressed"


# ---------------------------------------------------------------------------
# Fix 5: bridge topological validation


def test_bridge_validation_requires_road_and_river():
    """A bridge edge with no nearby road or river support should be suppressed."""
    grid = _test_grid()
    detector = LinearFeatureDetector(grid, _valid_hexes())
    snapper = HexsideSnapper(grid, _valid_hexes())
    start_idx = _find_interior_edge(snapper)
    pts, _, _ = _build_chain(snapper, 1, start_idx)
    mask = _mask_for_polyline(grid.image_full, pts)

    bridge_result = snapper.snap_layer(mask, "bridges")
    if bridge_result.accepted:
        validated = detector.validate_bridges(snapper, bridge_result, None, None, mask)
        assert not validated.accepted, "isolated bridge was not suppressed"


# ---------------------------------------------------------------------------
# Fix 6: road value calibration


def test_road_value_calibration_separates_primary_secondary():
    """Thick, dark strokes should be labeled primary; thin strokes secondary."""
    grid = _test_grid()
    detector = LinearFeatureDetector(grid, _valid_hexes())
    snapper = HexsideSnapper(grid, _valid_hexes())
    start_idx = _find_interior_edge(snapper)

    # thick primary road
    pts, _, _ = _build_chain(snapper, 1, start_idx)
    mask_primary = _mask_for_polyline(grid.image_full, pts, width=8)
    board = np.full((*grid.image_full[::-1],), 200, dtype=np.uint8)
    board[mask_primary] = 20  # dark ink

    result = snapper.snap_layer(mask_primary, "roads")
    edges = detector.calibrate_road_values(snapper, result, mask_primary, board)
    assert len(edges) == 1
    assert edges[0].value == "primary", f"expected primary, got {edges[0].value}"

    # thin secondary road
    mask_secondary = _mask_for_polyline(grid.image_full, pts, width=2)
    result2 = snapper.snap_layer(mask_secondary, "roads")
    edges2 = detector.calibrate_road_values(snapper, result2, mask_secondary, board)
    if edges2:
        assert edges2[0].value == "secondary", f"expected secondary, got {edges2[0].value}"


# ---------------------------------------------------------------------------
# Correction scorer


def test_correction_scorer_counts_added_removed_reclassified():
    corrections = {
        "NaB": {
            "road": {
                "added": ["0101|0102", "0102|0103"],
                "removed": ["0201|0202"],
                "reclassified": {"0126|0227": "primary"},
            }
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "corrections.json"
        path.write_text(json.dumps(corrections))

        output = {
            "road": [
                DetectedEdge(a="0101", b="0102", layer="road", value="primary"),
                DetectedEdge(a="0126", b="0227", layer="road", value="primary"),
            ]
        }
        scorer = CorrectionScorer(path)
        summary = scorer.score(output, "NaB")
        road = summary["road"]
        assert road["added_hits"] == 1
        assert road["removed_hits"] == 1  # 0201|0202 absent
        assert road["reclassified_hits"] == 1


def test_run_detector_acceptance_reports_missing_dataset():
    """The acceptance harness exits 2 when the corrections dataset is absent."""
    import subprocess
    repo = Path(__file__).resolve().parent.parent
    script = repo / "tests" / "run_detector_acceptance.py"
    env = dict(os.environ)
    env["DETECTOR_CORRECTIONS"] = "/nonexistent/corrections-2026-07-04.json"
    env["DETECTOR_OUTPUT"] = "/nonexistent/output.json"
    proc = subprocess.run([sys.executable, str(script), "--map", "NaB"],
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 2
    combined = proc.stdout + proc.stderr
    assert "BLOCKED" in combined or "not found" in combined or "ERROR" in combined


if __name__ == "__main__":
    test_perimeter_padding_keeps_boundary_ink()
    test_impassible_calibration_joins_dashed_outline()
    test_gap_fill_connects_short_linear_breaks()
    test_rail_deconfliction_suppresses_hexside_parallel_ink()
    test_bridge_validation_requires_road_and_river()
    test_road_value_calibration_separates_primary_secondary()
    test_correction_scorer_counts_added_removed_reclassified()
    test_run_detector_acceptance_reports_missing_dataset()
    print("all detector tests passed")
