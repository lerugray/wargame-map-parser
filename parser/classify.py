"""Reference-hex terrain classification.

The core idea (credit: Ray Weiss): a wargame map's terrain types are defined
*relative to each other* on that map's own palette. So classify each hex by
**nearest match to a few labeled reference (exemplar) hexes** — a known clear
hex, a known forest hex, a known lake/sea hex, a known swamp hex — rather than
hand-tuned absolute colour thresholds. Reference-matching self-calibrates to
the scan; absolute thresholds break the moment a map's palette differs from the
numbers you baked in.

Feature vector per hex (sampled from a centred interior box, away from the
printed hex number):

    [mean_R, mean_G, mean_B, gray_std, elongation, mark_density]

- **mean RGB** — hue. Water/sea is a distinct blue-grey (B > R, G); forest is
  green (G > R, low B); clear is bright cream. A *confident* exemplar per class
  is the whole game — one bad exemplar (e.g. a "sea" sample that's actually
  land) poisons every match.
- **gray_std** — texture amplitude. Solid fills (lake, sea) have LOW variance;
  printed terrain symbols (forest, swamp) have HIGH variance. This separates a
  solid blue lake from a stippled blue-grey swamp of nearly the same hue.
- **elongation / mark_density** — *morphology* of the printed marks
  (credit: Ray): **forest symbols are circular "bulbs"; swamp symbols are
  "lines"** (dashes/tussocks). Colour can't tell them apart on a cream palette
  (and dark town icons masquerade as forest), but blob shape can: forest blobs
  are compact (elongation ~1), swamp blobs are elongated (high elongation).

Hard limit — HEXSIDE terrain. Full-hex classification (any method) cannot
capture terrain drawn on hex EDGES: lakes-on-hexsides, rivers, escarpments. On
the canonical example (TWU East Prussia) the real Masurian lakes run along hex
edges, so many hexes are half-lake/half-land and no full-hex label is right.
Detect/keep those as a confined region and model them in a separate EDGE layer.

---

GotA (Guns of the Americas, 2026-06-30) — REFINEMENTS OVER PURE NEAREST-EXEMPLAR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GotA's ~4,000-hex continental map revealed three cases where
``ReferenceClassifier`` alone is insufficient:

1. **Same-base-color terrains** (GotA clear/desert/rough = three tan shades).
   Mean-color nearest-exemplar conflates them; shade ordering + local context
   needed.

2. **Symbol-only terrains** (GotA swamp = short horizontal broken dashes on
   cream). Color alone finds nothing (cream == clear). Morphology finds them
   once trained on operator-confirmed exemplars (10/12 hit rate post-supervised
   training; near-zero pre-supervision).

3. **Over-broad water** (~480 spurious "water" hexes with relaxed centroid
   gate). Desaturated tan land drifts toward the water centroid. Fix: strict
   explicit blue-hue gate (``B > R + margin AND B > G + margin``) BEFORE
   nearest-centroid matching.

RECOMMENDED APPROACH (layered, not replaced — ``ReferenceClassifier`` stays):
  1. Strict blue-hue gate → definitive water (skip nearest-centroid for these).
  2. Morphology-first for symbol terrains (swamp/forest): if elongation > thresh
     → swamp; if high circular-blob density → forest. Override base color.
  3. ``ReferenceClassifier`` nearest-centroid for everything else.

OPERATOR-EXEMPLAR GOTCHAS:
  - **Mask the printed hex number** (CCRR label) before sampling — dark ink
    biases mean color and fakes a mark in morphology.
  - **Center-only sampling** (``r <= ~half hex``) — excludes edge rivers,
    coastlines, hexside bleed from neighboring terrain.
  - Exemplars from operator SCREENSHOTS of the real hex (not guessed from a
    rescaled overview) are required for hard terrains. See ``docs/CONVENTIONS.md``.

TODO / ROADMAP — ``HybridClassifier`` (see stub at bottom of this module).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .hexgrid import HexGrid, parse_ccrr


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------
def _interior_patch(arr: np.ndarray, cx: float, cy: float, r: int) -> np.ndarray:
    """RGB patch of radius r around (cx, cy), clipped to image bounds."""
    h, w = arr.shape[:2]
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r))
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r))
    return arr[y0:y1, x0:x1]


def _connected_components(mask: np.ndarray):
    """4-connected components of a boolean mask (no scipy dependency).

    Yields arrays of (row, col) pixel coordinates, one per component.
    """
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            yield np.array(pts)


def _morphology(patch: np.ndarray) -> tuple[float, float]:
    """Return (mean_elongation, mark_density) of the printed marks in a patch.

    Marks = pixels notably darker than the patch's bright background. Forest
    bulbs are compact (elongation ~1); swamp dashes are elongated (>~2).
    """
    if patch.size == 0:
        return 0.0, 0.0
    gray = patch[..., :3].mean(axis=2)
    bg = np.percentile(gray, 80)                 # bright background level
    mask = gray < (bg - 18)                       # darker marks
    density = float(mask.mean())
    elongs, weights = [], []
    for pts in _connected_components(mask):
        if len(pts) < 6:                          # ignore specks
            continue
        ys, xs = pts[:, 0].astype(float), pts[:, 1].astype(float)
        cov = np.cov(np.stack([xs, ys]))
        if cov.shape != (2, 2):
            continue
        ev = np.linalg.eigvalsh(cov)
        ev = np.clip(ev, 1e-6, None)
        elongs.append(float(np.sqrt(ev[1] / ev[0])))  # major/minor axis ratio
        weights.append(len(pts))
    if not elongs:
        return 1.0, density
    return float(np.average(elongs, weights=weights)), density


def hex_features(arr: np.ndarray, grid: HexGrid, col: int, row: int,
                 sample_radius: float | None = None) -> np.ndarray:
    """6-D feature vector for one hex, sampled from the full-res scan."""
    cx, cy = grid.center(col, row)
    r = int(sample_radius if sample_radius is not None else 0.42 * grid.hex_size())
    patch = _interior_patch(arr, cx, cy, r)
    if patch.size == 0:
        return np.zeros(6)
    rgb = patch[..., :3].reshape(-1, 3).astype(float)
    mean = rgb.mean(axis=0)
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    std = float(gray.std())
    elong, density = _morphology(patch)
    return np.array([mean[0], mean[1], mean[2], std, elong, density])


# ---------------------------------------------------------------------------
# reference-hex classifier
# ---------------------------------------------------------------------------
@dataclass
class ReferenceClassifier:
    """Nearest-(z-scored)-centroid classifier over labeled exemplar hexes."""

    grid: HexGrid
    _mu: np.ndarray = None
    _sd: np.ndarray = None
    _centroids: dict = None
    feature_names = ("R", "G", "B", "std", "elongation", "density")

    def fit(self, arr: np.ndarray, exemplars: dict[str, list[str]]) -> "ReferenceClassifier":
        """exemplars: {terrain_class: [hexcodes ...]} of CONFIDENT samples."""
        feats: dict[str, np.ndarray] = {}
        pooled = []
        for cls, hexes in exemplars.items():
            fs = np.array([hex_features(arr, self.grid, *parse_ccrr(h)) for h in hexes])
            feats[cls] = fs
            pooled.append(fs)
        pooled = np.vstack(pooled)
        self._mu = pooled.mean(axis=0)
        self._sd = pooled.std(axis=0)
        self._sd[self._sd == 0] = 1.0
        self._centroids = {cls: ((fs - self._mu) / self._sd).mean(axis=0)
                           for cls, fs in feats.items()}
        return self

    def _z(self, f: np.ndarray) -> np.ndarray:
        return (f - self._mu) / self._sd

    def classify_hex(self, arr: np.ndarray, hexcode: str) -> tuple[str, dict[str, float]]:
        """Return (best_class, {class: distance}) for one hex."""
        f = self._z(hex_features(arr, self.grid, *parse_ccrr(hexcode)))
        dists = {cls: float(np.linalg.norm(f - c)) for cls, c in self._centroids.items()}
        return min(dists, key=dists.get), dists

    def classify_all(self, arr: np.ndarray, hexes: list[str]) -> dict[str, str]:
        return {h: self.classify_hex(arr, h)[0] for h in hexes}


def load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------
# TODO: HybridClassifier — color gate + morphology override + centroid fallback
# ---------------------------------------------------------------------------
# Lesson from GotA (2026-06-30): pure ReferenceClassifier (nearest z-scored
# centroid) is insufficient for maps with same-palette terrains and/or
# symbol-only terrain types (swamp). The correct approach layers three tiers:
#
#   Tier 1 — STRICT BLUE-HUE GATE: B > R + margin AND B > G + margin → water.
#             Prevents ~480 spurious water hexes (GotA: centroid-only over-grab).
#
#   Tier 2 — MORPHOLOGY OVERRIDE: run _morphology() on the interior patch.
#             If elongation > swamp_thresh → swamp.
#             If mark_density > forest_thresh AND elongation < compact_thresh → forest.
#             Symbol terrains OVERRIDE base color.
#
#   Tier 3 — NEAREST-CENTROID FALLBACK: ReferenceClassifier.classify_hex() for
#             everything that doesn't hit a tier-1 or tier-2 gate.
#
# Implementation sketch (not yet wired — add tests before promoting to production):
#
# class HybridClassifier:
#     """Layered classifier: strict water gate → morphology override → centroid fallback.
#
#     Parameters
#     ----------
#     ref_clf : ReferenceClassifier
#         A fitted ReferenceClassifier used as the final fallback.
#     water_blue_margin : int
#         Minimum pixel-value margin by which B must exceed both R and G for a hex
#         to be classified as water without consulting the centroid. Default 15.
#     swamp_elongation_thresh : float
#         ``elongation`` above this value triggers a swamp override. Default 2.5.
#     forest_density_thresh : float
#         ``mark_density`` above this value (with low elongation) triggers a forest
#         override. Default 0.08.
#     forest_compact_thresh : float
#         ``elongation`` must be BELOW this for the forest override to fire. Default 1.8.
#     """
#     def __init__(self, ref_clf: ReferenceClassifier,
#                  water_blue_margin: int = 15,
#                  swamp_elongation_thresh: float = 2.5,
#                  forest_density_thresh: float = 0.08,
#                  forest_compact_thresh: float = 1.8):
#         self.ref_clf = ref_clf
#         self.water_blue_margin = water_blue_margin
#         self.swamp_elongation_thresh = swamp_elongation_thresh
#         self.forest_density_thresh = forest_density_thresh
#         self.forest_compact_thresh = forest_compact_thresh
#
#     def classify_hex(self, arr: np.ndarray, hexcode: str) -> str:
#         col, row = parse_ccrr(hexcode)
#         feat = hex_features(arr, self.ref_clf.grid, col, row)
#         r, g, b = feat[0], feat[1], feat[2]
#         elong, density = feat[4], feat[5]
#         # Tier 1: strict blue-hue gate
#         if b > r + self.water_blue_margin and b > g + self.water_blue_margin:
#             return "water"
#         # Tier 2: morphology override
#         if elong > self.swamp_elongation_thresh:
#             return "swamp"
#         if density > self.forest_density_thresh and elong < self.forest_compact_thresh:
#             return "forest"
#         # Tier 3: nearest-centroid fallback
#         return self.ref_clf.classify_hex(arr, hexcode)[0]
#
#     def classify_all(self, arr: np.ndarray, hexes: list[str]) -> dict[str, str]:
#         return {h: self.classify_hex(arr, h) for h in hexes}
#
# To promote:
#   1. Add unit tests covering the tier-1/2/3 code paths.
#   2. Calibrate water_blue_margin on a real scan (sample 10 confirmed water hexes
#      and 10 confirmed non-water tan hexes; choose margin at the gap midpoint).
#   3. Calibrate swamp/forest thresholds from operator-confirmed exemplars.
#   4. Remove the comment block above and make this a live class.

