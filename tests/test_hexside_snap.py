"""Tests for hexside-snap (HMM/Viterbi trace-to-hexside map-matching).

No real map raster is needed -- every mask here is a small synthetic
polyline drawn in-memory on a tiny synthetic HexGrid (a few hundred KB at
most), which doubles as the "smoke test on a small crop" this module needs:
it exercises the full clean -> skeletonize -> decompose -> Viterbi-decode ->
accept pipeline end to end, just on a lattice small enough to reason about
by hand instead of a real scanned board.
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw

from parser import HexGrid, check_geometry_ratio
from parser.hexside_snap import HexsideSnapper, SnapParams, snap_traces


# ---------------------------------------------------------------------------
# a small synthetic, geometrically-regular flat-top hex lattice big enough to
# build long test chains without running off the edge of the valid-hex block


def _test_grid(row_pitch_factor: float = 1.0) -> HexGrid:
    """A regular flat-top hex grid (row_pitch = sqrt(3)*H by default, so the
    single-band neighbor-distance gate in HexsideSnapper._build_candidate_graph
    finds every true neighbor). ``row_pitch_factor`` != 1.0 produces a
    slightly irregular grid for the irregular-grid test."""
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


def _find_interior_edge(snapper: HexsideSnapper, lo: int = 14, hi: int = 16) -> int:
    """Index of an EDGES entry whose both hexes fall in [lo,hi]x[lo,hi] --
    comfortably inside a 30x30 valid-hex block, away from any boundary."""
    for i, e in enumerate(snapper.EDGES):
        ca, ra = int(e["a"][:2]), int(e["a"][2:])
        cb, rb = int(e["b"][:2]), int(e["b"][2:])
        if lo <= ca <= hi and lo <= ra <= hi and lo <= cb <= hi and lo <= rb <= hi:
            return i
    raise AssertionError("no interior candidate edge found -- widen the valid-hex block")


def _build_chain(snapper: HexsideSnapper, n_edges: int, start_idx: int):
    """Walk ``n_edges`` connected lattice edges starting at ``start_idx``,
    always continuing in the straightest available direction (avoids
    doubling back into an already-used vertex within a small lattice).
    Returns ``(pixel_polyline, expected_sorted_edge_pairs)``: ``pixel_polyline``
    is the ground-truth vertex-to-vertex walk in pixel space -- exactly what
    a perfect hand trace of that hexside chain would look like -- and
    ``expected_sorted_edge_pairs`` is what a correct decode must recover.
    """
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
            raise RuntimeError(f"dead end after {len(path)} edges; widen the valid-hex block")
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
    return pts, expected


def _mask_for_polyline(image_full, pts, width: int = 4) -> np.ndarray:
    img = Image.new("L", image_full, 0)
    ImageDraw.Draw(img).line(pts, fill=255, width=width)
    return np.asarray(img) > 0


# ---------------------------------------------------------------------------
# tests


def test_single_hexside_decodes_to_known_edge():
    """A trace drawn exactly along one real hexside must decode to exactly
    that hexside -- the base case every more complex test builds on."""
    grid = _test_grid()
    snapper = HexsideSnapper(grid, _valid_hexes())
    start_idx = _find_interior_edge(snapper)
    pts, expected = _build_chain(snapper, 1, start_idx)
    assert len(expected) == 1

    mask = _mask_for_polyline(grid.image_full, pts)
    result = snapper.snap_layer(mask, "single")

    decoded = sorted((e["a"], e["b"]) for e in result.edges_out)
    assert decoded == expected
    assert result.diagnostics["accepted_edges"] == 1
    assert result.diagnostics["suppressed_candidates"] == 0


def test_zigzag_chain_decodes_to_known_hexside_sequence():
    """A drawn zigzag polyline on a known lattice must decode to the known
    hexside chain -- the core "does the HMM actually track the trace"
    check (task requirement #4)."""
    grid = _test_grid()
    snapper = HexsideSnapper(grid, _valid_hexes())
    start_idx = _find_interior_edge(snapper)
    pts, expected = _build_chain(snapper, 6, start_idx)
    assert len(expected) == 6

    mask = _mask_for_polyline(grid.image_full, pts)
    result = snapper.snap_layer(mask, "zigzag")

    decoded = sorted((e["a"], e["b"]) for e in result.edges_out)
    assert decoded == expected
    # every accepted edge should be a confident, along-tracking match, not a
    # short/ambiguous contact
    for rec in result.accepted.values():
        assert rec["theta_med"] <= snapper.ANG_ALONG


def test_long_chain_does_not_degenerate_to_cold_restart():
    """Regression guard for the cold-restart Viterbi bug documented in
    RESULTS.md implementation note 9 (out-v2/RESULTS.md, GotA 2026-07-02):
    the DP's "cold restart" fallback must compete only against sample
    positions with NO valid real transition -- if it competes on absolute
    accumulated cost instead, a long link degenerates to a 1-sample path
    (observed during the original dev pass: a 53-sample test link decoded
    to a single sample). This trace resamples to 60+ samples (16 edges *
    H=100px / 0.25H resample step); a working decode must recover a
    genuinely multi-edge, near-complete path -- not collapse to one state.
    """
    grid = _test_grid()
    snapper = HexsideSnapper(grid, _valid_hexes())
    start_idx = _find_interior_edge(snapper)
    n_edges = 16
    pts, expected = _build_chain(snapper, n_edges, start_idx)
    assert len(expected) == n_edges

    mask = _mask_for_polyline(grid.image_full, pts)
    result = snapper.snap_layer(mask, "long")

    # sanity: this trace really does resample to 50+ observations
    total_len = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                    for i in range(len(pts) - 1))
    n_samples_approx = total_len / snapper.RESAMPLE_STEP
    assert n_samples_approx >= 50, "test chain too short to exercise the regression"

    assert result.diagnostics["n_links_decoded"] == 1  # one connected skeleton link
    decoded = sorted((e["a"], e["b"]) for e in result.edges_out)
    # THE regression guard: a degenerate cold-restart decode caps out at ~1
    # edge (often zero, once the acceptance rule's Lparallel>=0.35H fails on
    # a single sample's worth of support). A correct decode recovers all 16.
    assert len(decoded) > 1, "decoded path degenerated -- possible cold-restart regression"
    assert decoded == expected


def test_slightly_irregular_grid_builds_and_decodes():
    """Task requirement: accept slightly-irregular grids where row pitch !=
    the regular-hex ratio. A ~2% row-pitch deviation should still build a
    candidate graph and decode a straightforward single-hexside trace
    correctly (see the module docstring's neighbor-detection caveat for
    grids that deviate further)."""
    grid = _test_grid(row_pitch_factor=1.02)
    check = check_geometry_ratio(grid, tolerance=0.03)
    assert check["deviation"] > 0.005  # confirms this grid really is off-ideal

    snapper = HexsideSnapper(grid, _valid_hexes())
    assert len(snapper.EDGES) > 0

    start_idx = _find_interior_edge(snapper)
    pts, expected = _build_chain(snapper, 1, start_idx)
    mask = _mask_for_polyline(grid.image_full, pts)
    result = snapper.snap_layer(mask, "irregular")
    decoded = sorted((e["a"], e["b"]) for e in result.edges_out)
    assert decoded == expected


def test_load_mask_accepts_array_and_alpha_png():
    arr = np.zeros((20, 20), dtype=bool)
    arr[5:15, 10] = True
    assert np.array_equal(HexsideSnapper.load_mask(arr), arr)

    rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    rgba[5:15, 10, 3] = 255  # alpha-encoded trace, the GotA convention
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.png")
        Image.fromarray(rgba, mode="RGBA").save(path)
        loaded = HexsideSnapper.load_mask(path)
    assert loaded.shape == (20, 20)
    assert loaded[5:15, 10].all()
    assert not loaded[:, :10].any()


def test_snap_traces_produces_hexwright_grouped_shape():
    """The output shape is the Hexwright-importable grouped shape
    ``{"<layer>": [{a,b}, ...]}`` (task requirement #2) -- checked here
    across two layers at once, matching how snap_v2.py's own main() merges
    multiple layers into one engine JSON."""
    grid = _test_grid()
    valid = _valid_hexes()
    snapper = HexsideSnapper(grid, valid)
    idx_a = _find_interior_edge(snapper, lo=10, hi=12)
    idx_b = _find_interior_edge(snapper, lo=20, hi=22)
    pts_a, expected_a = _build_chain(snapper, 1, idx_a)
    pts_b, expected_b = _build_chain(snapper, 1, idx_b)
    layers = {
        "rivers": _mask_for_polyline(grid.image_full, pts_a),
        "impassible": _mask_for_polyline(grid.image_full, pts_b),
    }

    hexwright_json, results = snap_traces(grid, valid, layers)

    assert set(hexwright_json.keys()) == {"rivers", "impassible"}
    assert hexwright_json["rivers"] == [{"a": expected_a[0][0], "b": expected_a[0][1]}]
    assert hexwright_json["impassible"] == [{"a": expected_b[0][0], "b": expected_b[0][1]}]
    assert all(isinstance(v, list) for v in hexwright_json.values())
    for edge in hexwright_json["rivers"] + hexwright_json["impassible"]:
        assert set(edge.keys()) == {"a", "b"}
    assert set(results.keys()) == {"rivers", "impassible"}


def test_overlay_render_writes_a_readable_image():
    """The overlay IS the acceptance check for this method (spec S2.7) --
    confirm render_overlay actually produces a valid image file."""
    grid = _test_grid()
    snapper = HexsideSnapper(grid, _valid_hexes())
    start_idx = _find_interior_edge(snapper)
    pts, _ = _build_chain(snapper, 3, start_idx)
    mask = _mask_for_polyline(grid.image_full, pts)
    result = snapper.snap_layer(mask, "overlay-test")

    with tempfile.TemporaryDirectory() as tmp:
        board_path = os.path.join(tmp, "board.jpg")
        Image.new("RGB", grid.image_full, (200, 195, 170)).save(board_path)
        out_path = os.path.join(tmp, "overlay.jpg")
        snapper.render_overlay(board_path, result.mask, result.accepted,
                               result.suppressed, out_path, scale=0.3)
        assert os.path.exists(out_path)
        with Image.open(out_path) as img:
            img.verify()


if __name__ == "__main__":
    test_single_hexside_decodes_to_known_edge()
    test_zigzag_chain_decodes_to_known_hexside_sequence()
    test_long_chain_does_not_degenerate_to_cold_restart()
    test_slightly_irregular_grid_builds_and_decodes()
    test_load_mask_accepts_array_and_alpha_png()
    test_snap_traces_produces_hexwright_grouped_shape()
    test_overlay_render_writes_a_readable_image()
    print("all tests passed")
