"""Automated linear-feature detector for hexside terrain, with the six
ranked improvements from ``docs/DETECTOR-IMPROVEMENT-SPEC-2026-07-04.md``.

This module sits above :mod:`parser.hexside_snap`.  It reuses the
HMM/Viterbi trace-to-hexside snapper for per-layer decoding, then applies
post-processing specific to each improvement:

1. padded-frame perimeter extraction (NaB edge-collar false negatives)
2. layer-specific impassible calibration (TWU impassible under-detection)
3. graph-continuity gap-fill (river/border/rail linear continuations)
4. rail vs. hexside orientation deconfliction (TWU rail false positives)
5. bridge topological validation (NaB bridge false positives)
6. road primary/secondary value calibration (NaB road reclassification)

Because the spec's acceptance tests are keyed to per-game correction
datasets (``corrections-2026-07-04.json``) that live in the game repos,
this module also ships :class:`CorrectionScorer` and
:func:`score_against_corrections` so those acceptance tests can be run
when the dataset is available.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, disk

from .hexgrid import HexGrid, parse_ccrr
from .hexside_snap import HexsideSnapper, SnapParams, LayerResult

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# data containers


@dataclass
class DetectedEdge:
    """One detected hexside edge, possibly with a layer value such as
    ``primary``/``secondary`` for roads."""
    a: str
    b: str
    layer: str
    value: Optional[str] = None

    def key(self, sep: str = "|") -> str:
        a, b = sorted((self.a, self.b))
        return f"{a}{sep}{b}"


@dataclass
class DetectorParams:
    """Tunable parameters for the six ranked fixes.  Every geometric field is
    expressed as a multiple of ``H`` (``grid.hex_size()``) where noted.
    """

    # Fix 1 -- padded-frame perimeter extraction
    pad_cols: int = 2          # extra hex columns/rows of padding to add
    pad_rows: int = 2
    boundary_support_min: float = 0.15  # * H -- min visible support for a boundary edge

    # Fix 2 -- impassible calibration
    impassible_close_r: float = 0.08    # * H -- closing disk radius for faint outlines
    impassible_min_component: float = 0.01  # * H^2 -- smaller than default snapper
    impassible_density_floor: float = 0.10  # fraction of edge pixels that must be ink
    impassible_join_hops: int = 2      # max graph hops for collinear fragment joining

    # Fix 3 -- graph continuity gap-fill
    gap_fill_max_hops: int = 2         # missing lattice edges to bridge
    gap_fill_heading_tol: float = 25.0  # degrees -- endpoint headings must be consistent
    gap_fill_support_min: float = 0.15  # * H -- mask evidence along the missing edge

    # Fix 4 -- rail orientation deconfliction
    rail_window_r: float = 0.80         # * H -- local orientation window around midpoint
    rail_orient_tol: float = 30.0       # degrees -- rail link vs. mask alignment
    hexside_parallel_tol: float = 20.0  # degrees -- suppress if mask parallels hexside

    # Fix 5 -- bridge validation
    bridge_search_r: float = 1.0        # * H -- how far to look for road/river supports
    bridge_symbol_min: float = 0.10     # * H^2 -- min mask area at bridge crossing

    # Fix 6 -- road value calibration
    road_width_primary_min: float = 3.0   # px -- stroke-width floor for primary
    road_contrast_primary_min: float = 40.0  # intensity delta from background
    road_width_secondary_max: float = 2.0   # px -- stroke-width ceiling for secondary


# ---------------------------------------------------------------------------
# helpers


def _edge_pair(a: str, b: str, sep: str = "|") -> str:
    a, b = sorted((a, b))
    return f"{a}{sep}{b}"


def _parse_edge_key(key: str) -> tuple[str, str]:
    for sep in ("|", "-"):
        if sep in key:
            a, b = key.split(sep, 1)
            return tuple(sorted((a.strip(), b.strip())))
    raise ValueError(f"cannot parse edge key: {key!r}")


def _hex_colrow(ccrr: str) -> tuple[int, int]:
    return parse_ccrr(ccrr)


def _edge_midpoint(snapper: HexsideSnapper, eidx: int) -> np.ndarray:
    e = snapper.EDGES[eidx]
    return (e["pa"] + e["pb"]) / 2.0


def _edge_direction(snapper: HexsideSnapper, eidx: int) -> np.ndarray:
    e = snapper.EDGES[eidx]
    d = e["pb"] - e["pa"]
    L = float(np.hypot(*d))
    return d / L if L > 1e-6 else np.array([1.0, 0.0])


def _edge_length(snapper: HexsideSnapper, eidx: int) -> float:
    e = snapper.EDGES[eidx]
    return float(np.hypot(*(e["pb"] - e["pa"])))


def _acute_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = abs(float(np.dot(u, v)))
    c = min(1.0, max(-1.0, c))
    return math.degrees(math.acos(c))


def _mask_support_along_edge(snapper: HexsideSnapper, mask: np.ndarray,
                              eidx: int, band_r: float) -> float:
    """Length of mask ink within ``band_r`` of the infinite supporting line of
    edge ``eidx``, clipped to the segment endpoints.  Returns px support."""
    e = snapper.EDGES[eidx]
    pa, pb = e["pa"], e["pb"]
    mid = (pa + pb) / 2.0
    u = _edge_direction(snapper, eidx)
    v = np.array([-u[1], u[0]])
    H = snapper.H
    band_px = max(1, int(math.ceil(band_r)))
    yy, xx = np.where(mask)
    if len(yy) == 0:
        return 0.0
    pts = np.column_stack([xx, yy]).astype(float)
    rel = pts - mid
    along = rel @ u
    across = rel @ v
    half = _edge_length(snapper, eidx) / 2.0
    in_seg = (np.abs(along) <= half + band_r) & (np.abs(across) <= band_r)
    return float(np.sum(in_seg))


def _local_mask_orientation(mask: np.ndarray, pt: np.ndarray,
                             window_r: float) -> Optional[np.ndarray]:
    """Principal orientation of mask pixels inside a window around ``pt``,
    returned as a unit vector.  Uses second-moment PCA on the ink pixels.
    ``None`` if too few pixels."""
    x, y = pt
    r = max(2, int(math.ceil(window_r)))
    h, w = mask.shape
    x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
    y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
    crop = mask[y0:y1, x0:x1]
    ys, xs = np.where(crop)
    if len(xs) < 4:
        return None
    xs = xs + x0
    ys = ys + y0
    pts = np.column_stack([xs, ys]).astype(float)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, np.argmax(vals)]
    return major / np.hypot(*major)


def _stroke_width_and_contrast(mask: np.ndarray, board: Optional[np.ndarray],
                                snapper: HexsideSnapper, eidx: int,
                                band_r: float) -> tuple[float, float]:
    """Estimate mean stroke width and contrast for the ink near edge ``eidx``.
    Width is the mean perpendicular extent of mask pixels in a band around the
    edge; contrast is the mean intensity difference between ink and a slightly
    wider background band.  If ``board`` is None, contrast is 0."""
    e = snapper.EDGES[eidx]
    pa, pb = e["pa"], e["pb"]
    mid = (pa + pb) / 2.0
    u = _edge_direction(snapper, eidx)
    v = np.array([-u[1], u[0]])
    half = _edge_length(snapper, eidx) / 2.0
    h, w = mask.shape

    # sample points in a bounding box around the edge segment
    r = max(1, int(math.ceil(band_r * 3)))
    cx, cy = int(mid[0]), int(mid[1])
    x0, x1 = max(0, cx - r - int(half)), min(w, cx + r + int(half) + 1)
    y0, y1 = max(0, cy - r - int(half)), min(h, cy + r + int(half) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    rel = pts - mid
    along = rel @ u
    across = rel @ v
    in_outer = (np.abs(along) <= half + band_r * 2) & (np.abs(across) <= band_r * 2)
    in_inner = (np.abs(along) <= half + band_r) & (np.abs(across) <= band_r)
    # restrict measurements to actual ink pixels inside the geometric bands
    mask_crop = mask[y0:y1, x0:x1].ravel()
    ink_inner = in_inner & mask_crop
    ink_outer = in_outer & mask_crop
    bg_outer = in_outer & ~mask_crop

    width = 0.0
    if ink_inner.sum() >= 2:
        width = float(np.ptp(across[ink_inner]))

    contrast = 0.0
    if board is not None and ink_inner.any():
        board_gray = np.asarray(board)
        if board_gray.ndim == 3:
            board_gray = board_gray.mean(axis=-1)
        board_crop = board_gray[y0:y1, x0:x1]
        ink_vals = board_crop.ravel()[ink_inner]
        if bg_outer.any():
            bg_vals = board_crop.ravel()[bg_outer]
        elif ink_outer.any():
            bg_vals = board_crop.ravel()[ink_outer]
        else:
            bg_vals = board_crop.ravel()[in_outer]
        if len(bg_vals):
            contrast = float(np.median(bg_vals) - np.median(ink_vals))
    return width, contrast


# ---------------------------------------------------------------------------
# core detector


class LinearFeatureDetector:
    """Orchestrate extraction and post-processing of linear hexside features.

    Usage::

        grid = HexGrid.from_json("hexgrid.json")
        valid = [...]
        detector = LinearFeatureDetector(grid, valid)
        out = detector.detect({"rivers": river_mask, "roads": road_mask, ...})
    """

    def __init__(self, grid: HexGrid, valid_hexes: Iterable[str],
                 params: Optional[DetectorParams] = None,
                 base_snap_params: Optional[SnapParams] = None,
                 layer_snapper_class: Optional[dict[str, type]] = None,
                 layer_snap_fn: Optional[dict[str, callable]] = None):
        self.grid = grid
        self.valid_hexes = sorted(set(valid_hexes))
        self.params = params or DetectorParams()
        self.base_snap_params = base_snap_params or SnapParams()
        self.H = grid.hex_size()
        self.layer_snapper_class = layer_snapper_class or {}
        self.layer_snap_fn = layer_snap_fn or {}

    def _snapper(self, layer_name: str = "layer",
                 snap_params: Optional[SnapParams] = None) -> HexsideSnapper:
        cls = self.layer_snapper_class.get(layer_name, HexsideSnapper)
        return cls(self.grid, self.valid_hexes,
                   params=snap_params or self.base_snap_params)

    def _default_snap_fn(self, snapper: HexsideSnapper, mask: np.ndarray,
                         layer_name: str) -> LayerResult:
        return snapper.snap_layer(mask, layer_name=layer_name)

    # -- Fix 1: padded-frame perimeter extraction ---------------------------

    def pad_mask_for_perimeter(self, mask: np.ndarray,
                                crop_bounds: Optional[tuple] = None) -> np.ndarray:
        """Pad a boolean mask with a false-valued border so that perimeter ink
        touching the original crop boundary is preserved through skeleton
        extraction.  ``crop_bounds`` is ``(x0, y0, x1, y1)`` in original image
        coordinates; if None, the mask's full extent is the crop."""
        pad_c = int(round(self.params.pad_cols * self.H))
        pad_r = int(round(self.params.pad_rows * self.H))
        h, w = mask.shape
        padded = np.zeros((h + 2 * pad_r, w + 2 * pad_c), dtype=bool)
        padded[pad_r:pad_r + h, pad_c:pad_c + w] = mask
        return padded

    def score_boundary_edges(self, snapper: HexsideSnapper,
                              per_edge_support: dict,
                              original_shape: tuple,
                              pad: tuple[int, int]) -> dict[int, float]:
        """Return a visible-support normalization factor for edges whose
        geometric segment intersects the original crop boundary.  Edges fully
        inside the original frame receive factor 1.0."""
        orig_h, orig_w = original_shape
        pad_r, pad_c = pad
        factors = {}
        for eidx, sup in per_edge_support.items():
            e = snapper.EDGES[eidx]
            pa = e["pa"] - np.array([pad_c, pad_r])
            pb = e["pb"] - np.array([pad_c, pad_r])
            xs = [pa[0], pb[0]]
            ys = [pa[1], pb[1]]
            inside = all(0 <= x < orig_w and 0 <= y < orig_h
                         for x, y in zip(xs, ys))
            if inside:
                factors[eidx] = 1.0
                continue
            # fraction of segment length inside the original frame
            clipped = _clip_segment_to_rect(pa, pb, (0, 0, orig_w, orig_h))
            if clipped is None:
                factors[eidx] = 0.0
                continue
            (x0, y0), (x1, y1) = clipped
            visible_len = math.hypot(x1 - x0, y1 - y0)
            full_len = _edge_length(snapper, eidx)
            factors[eidx] = visible_len / full_len if full_len > 1e-6 else 0.0
        return factors

    def detect_with_perimeter_padding(self, mask: np.ndarray, layer: str,
                                       snapper: Optional[HexsideSnapper] = None,
                                       snap_params: Optional[SnapParams] = None
                                       ) -> LayerResult:
        """Fix 1 entry point: pad the mask, snap, then normalize boundary-edge
        support by the visible fraction of the edge inside the original crop.
        """
        padded = self.pad_mask_for_perimeter(mask)
        if snapper is None:
            snapper = self._snapper(layer, snap_params)
        snap_fn = self.layer_snap_fn.get(layer, self._default_snap_fn)
        result = snap_fn(snapper, padded, layer)

        # translate accepted edge geometry back to original frame and adjust
        # support for boundary edges
        pad_c = int(round(self.params.pad_cols * self.H))
        pad_r = int(round(self.params.pad_rows * self.H))
        factors = self.score_boundary_edges(
            snapper, result.accepted, mask.shape, (pad_r, pad_c))

        accepted = {}
        for eidx, rec in result.accepted.items():
            fac = factors.get(eidx, 1.0)
            rec = dict(rec)
            rec["visible_fraction"] = fac
            if fac > 0 and rec["Lparallel"] / max(fac, 0.01) >= snapper.MIN_PARALLEL:
                accepted[eidx] = rec
        result.accepted = accepted
        result.edges_out = sorted(
            [{"a": snapper.EDGES[e]["a"], "b": snapper.EDGES[e]["b"]} for e in accepted],
            key=lambda o: (o["a"], o["b"]),
        )
        return result

    # -- Fix 2: impassible-specific calibration ------------------------------

    def calibrate_impassible_mask(self, board: Optional[np.ndarray],
                                   base_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Fix 2 entry point: build an impassible-feature mask tuned for faint,
        dashed continuous outlines.  If ``base_mask`` is supplied, it is used
        as a starting point and cleaned with impassible-tuned morphology; if
        not, a conservative edge/contrast detector on ``board`` is used.
        """
        if base_mask is not None:
            m = base_mask.astype(bool)
        elif board is not None:
            m = self._impassible_ink_mask(board)
        else:
            raise ValueError("calibrate_impassible_mask needs board or base_mask")

        p = self.params
        close_r = max(1, int(round(p.impassible_close_r * self.H)))
        m = ndi.binary_dilation(m, structure=disk(close_r))
        m = ndi.binary_dilation(m, structure=disk(max(1, close_r // 2)))  # join dashes
        # small components are kept here because faint outlines are thin
        m = ndi.binary_fill_holes(m)
        return m

    def _impassible_ink_mask(self, board: np.ndarray) -> np.ndarray:
        """Simple, conservative dark-outline detector on a grayscale or RGB
        board.  Real impassible outlines are usually dark/black strokes on a
        lighter background, so a local contrast floor works across palettes."""
        gray = np.asarray(board)
        if gray.ndim == 3:
            gray = gray.mean(axis=-1)
        # dark ink: intensity below a locally-adaptive floor
        bg = ndi.maximum_filter(gray, size=int(2 * self.H))
        dark = gray < (bg - self.params.road_contrast_primary_min)
        # also keep high-gradient boundaries
        gx, gy = np.gradient(gray)
        grad = np.hypot(gx, gy)
        edge = grad > np.percentile(grad, 90)
        return dark | edge

    # -- Fix 3: graph continuity gap-fill --------------------------------------

    def gap_fill_layer(self, snapper: HexsideSnapper, result: LayerResult,
                       mask: np.ndarray) -> LayerResult:
        """Fix 3 entry point: after snapping a layer, build a graph of accepted
        edges and fill short gaps (1--2 missing lattice edges) between
        same-layer degree-1 endpoints whose headings are consistent and where
        the intervening edge(s) carry mask evidence."""
        accepted = set(result.accepted.keys())
        if not accepted:
            return result

        # graph: vertex id -> set of edge indices
        vert_edges: dict[int, set[int]] = defaultdict(set)
        for eidx in accepted:
            e = snapper.EDGES[eidx]
            vert_edges[e["va"]].add(eidx)
            vert_edges[e["vb"]].add(eidx)

        # endpoints: vertices incident to exactly one accepted edge
        endpoints = [v for v, eds in vert_edges.items() if len(eds) == 1]
        if len(endpoints) < 2:
            return result

        added: set[int] = set()
        for i, va in enumerate(endpoints):
            ea = next(iter(vert_edges[va]))
            dir_a = _endpoint_heading(snapper, ea, va)
            for vb in endpoints[i + 1:]:
                eb = next(iter(vert_edges[vb]))
                dir_b = _endpoint_heading(snapper, eb, vb)
                # endpoints must point toward each other with similar direction
                if _acute_angle_deg(dir_a, -dir_b) > self.params.gap_fill_heading_tol:
                    continue
                path = _short_graph_path(snapper, va, vb,
                                          self.params.gap_fill_max_hops,
                                          avoid=accepted)
                if path is None:
                    continue
                # require mask evidence on every missing edge
                ok = True
                support = 0.0
                for eidx in path:
                    if eidx in accepted:
                        continue
                    s = _mask_support_along_edge(
                        snapper, mask, eidx,
                        self.params.gap_fill_support_min * self.H)
                    if s < self.params.gap_fill_support_min * self.H:
                        ok = False
                        break
                    support += s
                    added.add(eidx)
                if ok:
                    for eidx in path:
                        if eidx not in result.accepted:
                            result.accepted[eidx] = {
                                "Lparallel": support,
                                "Lcross": 0.0,
                                "theta_med": 0.0,
                                "n_samples": 0,
                                "gap_fill": True,
                            }

        result.edges_out = sorted(
            [{"a": snapper.EDGES[e]["a"], "b": snapper.EDGES[e]["b"]} for e in result.accepted],
            key=lambda o: (o["a"], o["b"]),
        )
        result.diagnostics["gap_filled_edges"] = len(added)
        return result

    # -- Fix 4: rail vs hexside orientation deconfliction --------------------

    def deconflict_rails(self, snapper: HexsideSnapper, rail_result: LayerResult,
                         road_result: Optional[LayerResult],
                         rail_mask: np.ndarray) -> LayerResult:
        """Fix 4 entry point: suppress rail candidates whose local mask
        orientation is parallel to the crossed hexside rather than aligned
        with the center-to-center rail link, unless the rail connects two
        already accepted rail components with independent rail evidence."""
        accepted = dict(rail_result.accepted)
        to_drop: set[int] = set()

        # precompute connected rail components
        rail_components = _edge_components(snapper, set(accepted.keys()))
        comp_id = {}
        for cid, eds in enumerate(rail_components):
            for eidx in eds:
                comp_id[eidx] = cid

        for eidx in list(accepted.keys()):
            e = snapper.EDGES[eidx]
            mid = _edge_midpoint(snapper, eidx)
            orient = _local_mask_orientation(rail_mask, mid,
                                             self.params.rail_window_r * self.H)
            if orient is None:
                continue
            hexside_dir = _edge_direction(snapper, eidx)
            # rail link direction is center-to-center, perpendicular to hexside
            rail_dir = np.array([-hexside_dir[1], hexside_dir[0]])
            parallel_to_hexside = _acute_angle_deg(orient, hexside_dir) < self.params.hexside_parallel_tol
            aligned_to_rail = _acute_angle_deg(orient, rail_dir) < self.params.rail_orient_tol
            if aligned_to_rail:
                continue
            if parallel_to_hexside:
                # do not drop if it connects two independent rail components
                cid = comp_id.get(eidx)
                neighbors = set(e for e in accepted if e != eidx and comp_id.get(e) != cid)
                if not neighbors:
                    to_drop.add(eidx)

        for eidx in to_drop:
            rail_result.suppressed[eidx] = rail_result.accepted.pop(eidx)
            rail_result.suppressed[eidx]["rail_orient_conflict"] = True

        # remove edges also claimed by a road layer (orientation-independent overlap)
        if road_result is not None:
            road_edges = set(road_result.accepted.keys())
            for eidx in list(rail_result.accepted.keys()):
                if eidx in road_edges:
                    rail_result.suppressed[eidx] = rail_result.accepted.pop(eidx)
                    rail_result.suppressed[eidx]["road_overlap"] = True

        rail_result.edges_out = sorted(
            [{"a": snapper.EDGES[e]["a"], "b": snapper.EDGES[e]["b"]} for e in rail_result.accepted],
            key=lambda o: (o["a"], o["b"]),
        )
        rail_result.diagnostics["rail_deconfliction_drops"] = len(to_drop)
        return rail_result

    # -- Fix 5: bridge topological validation --------------------------------

    def validate_bridges(self, bridge_snapper: HexsideSnapper,
                          bridge_result: LayerResult,
                          road_snapper: Optional[HexsideSnapper] = None,
                          road_result: Optional[LayerResult] = None,
                          river_snapper: Optional[HexsideSnapper] = None,
                          river_result: Optional[LayerResult] = None,
                          bridge_mask: Optional[np.ndarray] = None) -> LayerResult:
        """Fix 5 entry point: accept a bridge only if a road edge and a river
        edge are present or newly inferred at the same/adjacent crossing, and
        bridge-symbol evidence exists.  Isolated bridge candidates without both
        supports are suppressed."""
        accepted = dict(bridge_result.accepted)
        if not accepted:
            return bridge_result

        road_edges = set(road_result.accepted.keys()) if road_result else set()
        river_edges = set(river_result.accepted.keys()) if river_result else set()
        search_px = self.params.bridge_search_r * self.H
        bridge_H = getattr(bridge_snapper, "H", self.H) if bridge_snapper else self.H

        to_drop: set[int] = set()
        for eidx in list(accepted.keys()):
            e = bridge_snapper.EDGES[eidx] if bridge_snapper else None
            mid = _edge_midpoint(bridge_snapper, eidx) if bridge_snapper else (
                (e["pa"] + e["pb"]) / 2.0 if e else np.array([0.0, 0.0]))

            has_road = (_has_nearby_edge(road_snapper, mid, road_edges, search_px)
                        if road_snapper else False)
            has_river = (_has_nearby_edge(river_snapper, mid, river_edges, search_px)
                         if river_snapper else False)
            symbol_ok = True
            if bridge_mask is not None and bridge_snapper is not None:
                area = _mask_support_along_edge(bridge_snapper, bridge_mask, eidx,
                                               self.params.bridge_search_r * bridge_H)
                symbol_ok = area >= self.params.bridge_symbol_min * bridge_H * bridge_H

            if not (has_road and has_river and symbol_ok):
                to_drop.add(eidx)

        for eidx in to_drop:
            bridge_result.suppressed[eidx] = bridge_result.accepted.pop(eidx)
            bridge_result.suppressed[eidx]["bridge_missing_support"] = True

        bridge_result.edges_out = sorted(
            [{"a": bridge_snapper.EDGES[e]["a"], "b": bridge_snapper.EDGES[e]["b"]}
             for e in bridge_result.accepted],
            key=lambda o: (o["a"], o["b"]),
        )
        bridge_result.diagnostics["bridge_validation_drops"] = len(to_drop)
        return bridge_result

    # -- Fix 6: road value calibration ---------------------------------------

    def calibrate_road_values(self, snapper: HexsideSnapper, road_result: LayerResult,
                              road_mask: np.ndarray,
                              board: Optional[np.ndarray] = None) -> list[DetectedEdge]:
        """Fix 6 entry point: classify each accepted road edge as primary or
        secondary using local stroke width and contrast over the final road
        mask, not pre-repair raw density.  Returns a list of :class:`DetectedEdge`
        with the ``value`` field populated."""
        out = []
        for eidx in sorted(road_result.accepted.keys()):
            e = snapper.EDGES[eidx]
            width, contrast = _stroke_width_and_contrast(
                road_mask, board, snapper, eidx,
                band_r=max(1.0, self.params.road_width_primary_min))
            if (width >= self.params.road_width_primary_min and
                    contrast >= self.params.road_contrast_primary_min):
                value = "primary"
            elif width <= self.params.road_width_secondary_max:
                value = "secondary"
            else:
                # ambiguous: default to secondary unless very dark/wide
                value = "secondary" if contrast < self.params.road_contrast_primary_min else "primary"
            out.append(DetectedEdge(a=e["a"], b=e["b"], layer="road", value=value))
        return out

    # -- full pipeline -------------------------------------------------------

    def detect(self, board: Optional[np.ndarray],
               masks: dict[str, np.ndarray],
               layer_snap_params: Optional[dict[str, SnapParams]] = None) -> dict[str, list[DetectedEdge]]:
        """Run the full detector-improvement pipeline on ``masks``.

        ``masks`` maps layer name to boolean ndarray.  Supported layer names:
        ``rivers``, ``roads``, ``rail``, ``rails``, ``border``, ``borders``,
        ``impassible``, ``bridges``.  Returns a dict mapping layer name to a
        list of :class:`DetectedEdge`.
        """
        layer_snap_params = layer_snap_params or {}
        raw: dict[str, LayerResult] = {}
        snappers: dict[str, HexsideSnapper] = {}

        # Initial snap, with layer-specific tweaks where applicable.
        for name, mask in masks.items():
            sp = layer_snap_params.get(name)
            snapper = self._snapper(name, sp)
            snappers[name] = snapper
            if name in ("rivers", "roads", "rail", "rails", "border", "borders", "impassible", "bridges"):
                # perimeter padding for all edge layers (Fix 1)
                result = self.detect_with_perimeter_padding(mask, name, snapper=snapper, snap_params=sp)
            else:
                result = snapper.snap_layer(mask, layer_name=name)
            raw[name] = result

        # Fix 2: impassible calibration
        if "impassible" in masks:
            imp_mask = self.calibrate_impassible_mask(board, masks["impassible"])
            imp_snapper = self._snapper("impassible", layer_snap_params.get("impassible"))
            snappers["impassible"] = imp_snapper
            raw["impassible"] = self.detect_with_perimeter_padding(
                imp_mask, "impassible", snapper=imp_snapper,
                snap_params=layer_snap_params.get("impassible"))

        # Fix 3: gap-fill for linear layers
        linear_layers = ["rivers", "border", "borders", "rail", "rails"]
        for name in linear_layers:
            if name in raw and name in masks:
                raw[name] = self.gap_fill_layer(snappers[name], raw[name], masks[name])

        # Fix 4: rail deconfliction
        rail_name = "rail" if "rail" in raw else "rails"
        if rail_name in raw:
            road_res = raw.get("roads")
            raw[rail_name] = self.deconflict_rails(
                snappers[rail_name], raw[rail_name], road_res,
                masks.get(rail_name, masks["impassible"]))

        # Fix 5: bridge validation
        if "bridges" in raw:
            raw["bridges"] = self.validate_bridges(
                snappers.get("bridges"), raw["bridges"],
                snappers.get("roads"), raw.get("roads"),
                snappers.get("rivers"), raw.get("rivers"),
                masks.get("bridges"))

        # Fix 6: road value calibration
        out: dict[str, list[DetectedEdge]] = {}
        for name, result in raw.items():
            if name == "roads" and name in masks:
                out[name] = self.calibrate_road_values(snappers[name], result, masks[name], board)
            else:
                out[name] = [
                    DetectedEdge(a=e["a"], b=e["b"], layer=name)
                    for e in result.edges_out
                ]
        return out


# ---------------------------------------------------------------------------
# acceptance-test scoring


class CorrectionScorer:
    """Score detector output against an operator correction dataset.

    Expected correction JSON shape::

        {
          "NaB": {
            "river": {"added": ["0101|0102", ...], "removed": [...],
                      "reclassified": {"0126|0227": "primary"}},
            ...
          },
          "TWU": {...}
        }

    Keys may use ``|`` or ``-`` as the edge separator.  ``added``/``reclassified``
    keys must be present with the operator value; ``removed`` keys must be absent.
    """

    def __init__(self, corrections_path: str | Path):
        self.path = Path(corrections_path)
        if not self.path.exists():
            raise FileNotFoundError(f"corrections dataset not found: {self.path}")
        self.data = json.loads(self.path.read_text())

    def score(self, output: dict[str, list[DetectedEdge]], map_name: str) -> dict:
        """Return per-layer and aggregate scores for ``map_name``."""
        corrections = self.data.get(map_name, {})
        summary: dict[str, dict] = {}
        for layer, corr in corrections.items():
            edges = output.get(layer, [])
            edge_values = {e.key(): (e.value or "present") for e in edges}
            added = corr.get("added", [])
            removed = corr.get("removed", [])
            reclassified = corr.get("reclassified", {})

            added_hits = sum(1 for k in added if _key_present(k, edge_values))
            removed_hits = sum(1 for k in removed if not _key_present(k, edge_values))
            reclass_hits = sum(
                1 for k, v in reclassified.items()
                if edge_values.get(_normalize_key(k)) == v
            )
            summary[layer] = {
                "added_total": len(added),
                "added_hits": added_hits,
                "removed_total": len(removed),
                "removed_hits": removed_hits,
                "reclassified_total": len(reclassified),
                "reclassified_hits": reclass_hits,
                "output_edges": len(edges),
            }
        return summary


def _normalize_key(key: str) -> str:
    a, b = _parse_edge_key(key)
    return f"{a}|{b}"


def _key_present(key: str, edge_values: dict) -> bool:
    try:
        return _normalize_key(key) in edge_values
    except ValueError:
        return False


def score_against_corrections(output: dict[str, list[DetectedEdge]],
                              corrections_path: str | Path,
                              map_name: str) -> dict:
    """Convenience wrapper around :class:`CorrectionScorer`."""
    scorer = CorrectionScorer(corrections_path)
    return scorer.score(output, map_name)


# ---------------------------------------------------------------------------
# internal geometry helpers


def _clip_segment_to_rect(p1: np.ndarray, p2: np.ndarray,
                          rect: tuple[float, float, float, float]
                          ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Liang-Barsky clip of segment p1-p2 to (xmin, ymin, xmax, ymax).
    Returns the clipped endpoints or None if fully outside."""
    xmin, ymin, xmax, ymax = rect
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x2 - x1, y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
    if u1 > u2:
        return None
    return (np.array([x1 + u1 * dx, y1 + u1 * dy]),
            np.array([x1 + u2 * dx, y2 + u2 * dy]))


def _endpoint_heading(snapper: HexsideSnapper, eidx: int, vertex: int) -> np.ndarray:
    """Unit vector pointing from ``vertex`` along edge ``eidx`` into the edge."""
    e = snapper.EDGES[eidx]
    if e["va"] == vertex:
        d = e["pb"] - e["pa"]
    else:
        d = e["pa"] - e["pb"]
    L = float(np.hypot(*d))
    return d / L if L > 1e-6 else np.array([1.0, 0.0])


def _short_graph_path(snapper: HexsideSnapper, v_from: int, v_to: int,
                      max_hops: int, avoid: set[int]) -> Optional[list[int]]:
    """Shortest path from ``v_from`` to ``v_to`` through the lattice using at
    most ``max_hops`` edges, avoiding edges in ``avoid``.  Returns the list of
    edge indices (including existing edges) or None."""
    if v_from == v_to:
        return []
    # BFS with path tracking
    frontier = {v_from: []}
    for _ in range(max_hops + 1):
        nxt = {}
        for v, path in frontier.items():
            for eidx in snapper.VERT_EDGES[v]:
                if eidx in avoid:
                    continue
                e = snapper.EDGES[eidx]
                ov = e["vb"] if e["va"] == v else e["va"]
                new_path = path + [eidx]
                if ov == v_to:
                    return new_path
                if ov not in nxt or len(new_path) < len(nxt[ov]):
                    nxt[ov] = new_path
        frontier = nxt
        if not frontier:
            break
    return None


def _edge_components(snapper: HexsideSnapper, edge_set: set[int]) -> list[set[int]]:
    """Connected components of ``edge_set`` in the lattice-vertex graph."""
    if not edge_set:
        return []
    adj: dict[int, set[int]] = defaultdict(set)
    for eidx in edge_set:
        e = snapper.EDGES[eidx]
        adj[eidx].update(
            e2 for e2 in snapper.VERT_EDGES[e["va"]]
            if e2 in edge_set and e2 != eidx
        )
        adj[eidx].update(
            e2 for e2 in snapper.VERT_EDGES[e["vb"]]
            if e2 in edge_set and e2 != eidx
        )
    seen = set()
    comps = []
    for eidx in edge_set:
        if eidx in seen:
            continue
        stack = [eidx]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.add(cur)
            stack.extend(adj[cur] - seen)
        comps.append(comp)
    return comps


def _has_nearby_edge(snapper: HexsideSnapper, pt: np.ndarray,
                     edge_set: set[int], radius: float) -> bool:
    """True if any edge in ``edge_set`` has a midpoint within ``radius`` of ``pt``."""
    if not edge_set:
        return False
    mids = np.array([_edge_midpoint(snapper, eidx) for eidx in edge_set])
    if len(mids) == 0:
        return False
    dists = np.hypot(*(mids - pt).T)
    return bool(np.any(dists <= radius))
