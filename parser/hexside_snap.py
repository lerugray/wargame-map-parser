"""Hexside-snap: HMM/Viterbi map-matching of hand-traced linear features onto
a hex-lattice hexside graph.

Method (credit: Fugu-ultra, spec-only mode, commissioned by Ray Weiss;
spec at the origin project, ``fugu-spec-v2-2026-07-02.md``; implemented and
operator-validated on *Guns of the Americas* 2026-07-02 -- 814 river +
64 impassible hexsides accepted, confirmed against the rendered overlay).
Ported into this repo as a first-class, parameterized module -- see
``docs/CONVENTIONS.md`` "Hexside-snap" section for when to reach for it.

Hand-traced linear features (rivers, ridges, impassible-terrain boundaries)
meander 25-50px off the clean geometric hexsides they represent -- proven on
GotA, don't re-litigate. Every distance-threshold / proximity-buffer method
hits a hard ~46% coverage ceiling against a meandering trace, because
"nearest hexside within Npx" throws away exactly the information that
resolves ambiguity: which way the trace is *headed*.

The fix: treat the traced skeleton as a noisy GPS trace and the hex-lattice
hexside graph as the road network. Decode the most likely CONNECTED walk
through the graph via per-link Viterbi, using:

- an **emission cost** that rewards low perpendicular distance to a
  hexside's supporting line AND parallel tangent (not just proximity) --
  this is what a distance-only method is blind to;
- a **transition cost** that only allows moving between graph-adjacent
  hexsides (same edge, through a shared lattice vertex, or a short bridge
  across a real gap in the trace);
- a post-decode **along-vs-crossing support rule** that throws out edges
  the trace only grazed or crossed, so the HMM can't claim a hexside just
  because the trace passed near or across it.

Every geometric parameter is a multiple of ``H``, the hexside length
(``HexGrid.hex_size()`` -- the circumradius of a flat-top hex, which equals
its side length). :class:`SnapParams` pins every constant the spec leaves
qualitative to a concrete value, matching the operator-validated GotA run
verbatim (``w_end=1.2``, junction-zone angle weight ``0.3``, bridge penalty
``3.0``/hidden edge, backtrack-in-junction penalty ``5.0``, reversal-in-
junction penalty ``8.0``).

A real bug was caught during the original dev pass and is preserved here as
a load-bearing fix, not an implementation detail: the Viterbi DP's
"cold restart" option (start a fresh hypothesis at a sample with no valid
transition from the previous sample) must compete only against sample
positions with NO valid real transition -- never against the *accumulated*
cost of a real multi-step path. A restart's cost is always just one
emission term, so if it competed on absolute accumulated cost it would
always look cheaper than any multi-step path and the DP would degenerate
to restarting at every sample (observed during dev: a 53-sample test link
decoded to a 1-sample path). See ``viterbi_link`` below and
``tests/test_hexside_snap.py::test_long_chain_does_not_degenerate_to_cold_restart``,
which regression-guards this specifically.

Usage
-----
As a library::

    from parser import HexGrid
    from parser.hexside_snap import HexsideSnapper, SnapParams

    grid = HexGrid.from_json("hexgrid.json")
    valid_hexes = [...]  # every eligible ("land") hex code
    snapper = HexsideSnapper(grid, valid_hexes)          # spec defaults
    results = snapper.snap_layers({
        "rivers": "traces/rivers-trace.png",              # path, or a bool ndarray
        "impassible": "traces/impassible-trace.png",
    })
    hexwright_json = HexsideSnapper.to_hexwright_json(results)
    # hexwright_json == {"rivers": [{"a":"CCRR","b":"CCRR"}, ...], "impassible": [...]}

    snapper.render_overlay("board.jpg", results["rivers"].mask,
                           results["rivers"].accepted, results["rivers"].suppressed,
                           "overlay-rivers.jpg", scale=0.5)

As a CLI::

    python -m parser.hexside_snap \\
        --grid hexgrid.json --terrain terrain.json \\
        --trace rivers=traces/rivers-trace.png \\
        --trace impassible=traces/impassible-trace.png \\
        --out hexsides-snap.json \\
        --board board.jpg --overlay overlays/ --overlay-scale 0.5

Output is the Hexwright-importable grouped shape ``{"<layer>": [{a,b}, ...]}``
-- the same shape ``store.importHexsides()`` already migrates from (verified
against ``hexwright/src/store.js`` on the origin project).

The 10px-proximity metric is BANNED as an acceptance criterion for this
method -- it is structurally wrong for a meandering trace (see
``docs/CONVENTIONS.md``). Acceptance is coverage / connectivity / Fréchet-
distance metrics (this module reports them) plus the operator's eyes on the
``--overlay`` render. Never call a hexside-snap run "done" from a count.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from skimage.morphology import (
    skeletonize, remove_small_objects, remove_small_holes, binary_closing, disk,
)

from .hexgrid import HexGrid, parse_ccrr

Image.MAX_IMAGE_PIXELS = None

OFFS8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
SIN30 = math.sin(math.radians(30))


# ---------------------------------------------------------------------------
# geometry / numeric helpers with no HexsideSnapper state


def huber(x: float) -> float:
    """Robust loss: quadratic for |x|<=1, linear after (spec S1.5)."""
    ax = abs(x)
    return 0.5 * ax * ax if ax <= 1 else ax - 0.5


def poly_arclen(chain: list[tuple[float, float]]) -> float:
    if len(chain) < 2:
        return 0.0
    pts = np.array(chain, dtype=float)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def angle_to_edge(tangent: np.ndarray, u_hat: np.ndarray) -> float:
    """Acute angle in degrees between an (unoriented) tangent and an edge
    direction -- the theta term in the emission cost (spec S1.5)."""
    c = abs(float(np.dot(tangent, u_hat)))
    c = min(1.0, max(-1.0, c))
    return math.degrees(math.acos(c))


def samples_arc_gap(samples: np.ndarray, i0: int, i1: int) -> float:
    return float(np.hypot(*(samples[i1] - samples[i0])))


def discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    """Discrete Fréchet distance between two polylines -- used to score how
    tightly a decoded lattice-vertex walk hugs the hand-traced skeleton
    (spec S2.1). Densify inputs at ``0.25H`` before calling for a meaningful
    number; this function itself just computes the distance."""
    n, m = len(P), len(Q)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float("inf")

    if n > 400:
        idx = np.linspace(0, n - 1, 400).astype(int)
        P = P[idx]
        n = len(P)
    if m > 400:
        idx = np.linspace(0, m - 1, 400).astype(int)
        Q = Q[idx]
        m = len(Q)
    ca = np.full((n, m), -1.0)

    def d(i, j):
        return float(np.hypot(*(P[i] - Q[j])))

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        if i == 0 and j == 0:
            ca[i, j] = d(0, 0)
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), d(i, 0))
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), d(0, j))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), d(i, j))
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    import sys as _sys
    old = _sys.getrecursionlimit()
    _sys.setrecursionlimit(10000)
    try:
        return c(n - 1, m - 1)
    finally:
        _sys.setrecursionlimit(old)


def build_skeleton_graph(skel: np.ndarray):
    """8-connected pixel graph of a boolean skeleton image -> (pixel set,
    {pixel: degree}). Degree-1 = endpoint, degree-2 = ordinary chain pixel,
    degree!=2 = junction candidate (spec S1.3.7)."""
    ys, xs = np.where(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    deg_img = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    degree = {p: int(deg_img[p[1], p[0]]) for p in pts}
    return pts, degree


def cluster_nodes(node_pixels: list[tuple[int, int]], radius: float):
    """Union-find cluster node pixels within Euclidean ``radius`` into
    skeleton-junction clusters -> (pixel->cluster_id, cluster_id->centroid)."""
    if not node_pixels:
        return {}, {}
    pts = np.array(node_pixels, dtype=float)
    tree = cKDTree(pts)
    close = tree.query_pairs(r=radius)
    par = list(range(len(pts)))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def u(x, y):
        rx, ry = f(x), f(y)
        if rx != ry:
            par[rx] = ry

    for i, j in close:
        u(i, j)
    groups: dict = defaultdict(list)
    for i in range(len(pts)):
        groups[f(i)].append(i)
    pix_to_cluster = {}
    cluster_centroid = {}
    for cid, (root, members) in enumerate(groups.items()):
        centroid = pts[members].mean(axis=0)
        cluster_centroid[cid] = centroid
        for m in members:
            pix_to_cluster[node_pixels[m]] = cid
    return pix_to_cluster, cluster_centroid


class St:
    """One HMM state: a physical hexside walked in one of its two
    orientations. ``tail``/``head`` are lattice-vertex ids; ``pt``/``ph`` are
    their pixel positions; ``u_hat`` is the unit tail->head direction;
    ``L`` is the segment length (~``H``)."""
    __slots__ = ("eidx", "orient", "tail", "head", "pt", "ph", "u_hat", "L")


# ---------------------------------------------------------------------------
# parameters


@dataclass
class SnapParams:
    """Geometric + cost parameters for hexside-snap.

    Every field expressed "as multiple of H" is later scaled by the
    calibrated hexside length (``H = grid.hex_size()``) -- so the same
    ``SnapParams`` instance is reusable across maps of any scale.

    **Defaults are the ``fugu-spec-v2-2026-07-02.md`` values, operator-
    validated on GotA 2026-07-02** (814 river + 64 impassible hexsides
    accepted, confirmed against the rendered overlay). Don't change a
    default without re-validating against a known-good overlay render --
    see ``docs/CONVENTIONS.md``.

    A few fields (``d_vanchor``, ``d_eanchor``, ``gap_split``,
    ``endpoint_snap_r``, ``q_endpoint_hyp``) are declared because the spec
    names them, but are **not referenced** by this implementation -- the
    same simplifications the original GotA run documented (its own
    "vertex-anchor cost" and "post-hoc endpoint union" are folded implicitly
    into the transition cost and the graph-connectivity constraint instead
    of being separate terms). Preserved here for API fidelity with the spec
    and in case a future extension wires them in; changing them currently
    has no effect on output.
    """

    # candidate hexside graph
    nbr_lo: float = 1.35             # * H -- neighbor-center distance gate, low
    nbr_hi: float = 1.85             # * H -- neighbor-center distance gate, high
    vertex_merge_tol: float = 0.03   # * H -- merge coincident geometric endpoints

    # mask preprocessing
    close_radius: float = 0.04       # * H -- binary-closing disk radius
    min_component_area: float = 0.02  # * H^2 -- drop mask components smaller than this
    min_hole_area: float = 0.02      # * H^2 -- fill mask holes smaller than this

    # skeleton -> links
    junction_contract_r: float = 0.20  # * H -- contract junction/knot pixel clusters
    spur_min: float = 0.30           # * H -- prune non-exit terminal spurs shorter than this
    exit_r: float = 0.75             # * H -- map/playable-boundary exit radius

    # resampling
    resample_step: float = 0.25      # * H
    tangent_chord: float = 0.50      # * H -- centered chord for tangent smoothing
    junction_zone_r: float = 0.30    # * H -- junction angle-downweight zone

    # candidate search
    r_search: float = 1.10           # * H -- candidate hexside search radius
    slack: float = 0.35              # * H -- allowed projection past segment endpoint
    max_cand_phys: int = 16          # max retained physical candidates per sample

    # emission cost
    d_emit: float = 0.45             # * H -- perpendicular-distance emission scale
    d_end: float = 0.30              # * H -- endpoint-overrun emission scale
    w_end: float = 1.2               # endpoint-overrun weight (spec: "slightly higher than distance")
    w_ang_normal: float = 1.0        # angle-term weight, normal samples
    w_ang_junction: float = 0.3      # angle-term weight, junction-zone samples (spec's own number)

    # transition cost
    d_trans: float = 0.35            # * H -- transition length scale
    backtrack_max: float = 0.15      # * H -- allowed same-edge backtrack before penalty
    backtrack_junction_penalty: float = 5.0
    reversal_junction_penalty: float = 8.0
    bridge_gap_max: float = 0.75     # * H -- bridge limit across short skeleton gaps
    bridge_max_hops: int = 2         # max hidden hexsides in a bridge
    bridge_penalty_per_edge: float = 3.0

    # acceptance rule (spec S1.8)
    ang_along: float = 35.0          # degrees -- along-edge tangent threshold
    ang_cross: float = 60.0          # degrees -- crossing tangent threshold
    min_parallel: float = 0.35       # * H -- minimum accepted parallel support per edge

    # declared per the spec's naming; not referenced by this implementation
    # (see class docstring)
    d_vanchor: float = 0.45          # * H
    d_eanchor: float = 0.45          # * H
    gap_split: float = 0.75          # * H
    endpoint_snap_r: float = 0.60    # * H
    q_endpoint_hyp: float = 0.60     # * H


# ---------------------------------------------------------------------------
# result container


@dataclass
class LayerResult:
    """Decode result for one trace-mask layer.

    ``edges_out`` is the Hexwright-importable fragment for this layer
    (sorted ``[{"a": CCRR, "b": CCRR}, ...]``). ``accepted``/``suppressed``
    map internal edge index -> support record (``Lparallel``, ``Lcross``,
    ``theta_med``, ``n_samples``) for diagnostics and overlay rendering.
    ``mask`` is the cleaned (post morphology) boolean mask, kept around for
    ``render_overlay``.
    """

    layer: str
    mask: np.ndarray
    accepted: dict
    suppressed: dict
    diagnostics: dict
    edges_out: list


# ---------------------------------------------------------------------------
# the snapper


class HexsideSnapper:
    """Builds the candidate hexside lattice graph once for a :class:`HexGrid`
    and a set of eligible ("land") hex codes, then decodes any number of
    trace-mask layers against it via HMM/Viterbi map-matching. See the
    module docstring for the method and :class:`SnapParams` for every
    tunable constant.
    """

    def __init__(self, grid: HexGrid, valid_hexes: Iterable[str],
                params: SnapParams | None = None):
        self.grid = grid
        self.params = params = params or SnapParams()
        self.H = H = grid.hex_size()

        self.VALID = sorted(set(valid_hexes))
        if len(self.VALID) < 2:
            raise ValueError("need >=2 valid hex codes to build a candidate hexside graph")
        self.centers = {c: grid.center(*parse_ccrr(c)) for c in self.VALID}
        self.cxy = np.array([self.centers[c] for c in self.VALID])
        self.ctree = cKDTree(self.cxy)

        # cost weights -- unitless, taken directly from SnapParams
        self.W_END = params.w_end
        self.W_ANG_NORMAL = params.w_ang_normal
        self.W_ANG_JUNCTION = params.w_ang_junction
        self.BRIDGE_PENALTY_PER_EDGE = params.bridge_penalty_per_edge
        self.BACKTRACK_JUNCTION_PENALTY = params.backtrack_junction_penalty
        self.REVERSAL_JUNCTION_PENALTY = params.reversal_junction_penalty
        self.MAX_CAND_PHYS = params.max_cand_phys
        self.BRIDGE_MAX_HOPS = params.bridge_max_hops
        self.ANG_ALONG = params.ang_along
        self.ANG_CROSS = params.ang_cross

        # geometric constants -- resolved to px by scaling with H
        self.R_SEARCH = params.r_search * H
        self.SLACK = params.slack * H
        self.D_EMIT = params.d_emit * H
        self.D_END = params.d_end * H
        self.D_TRANS = params.d_trans * H
        self.D_VANCHOR = params.d_vanchor * H       # not referenced -- see SnapParams docstring
        self.D_EANCHOR = params.d_eanchor * H       # not referenced -- see SnapParams docstring
        self.BACKTRACK_MAX = params.backtrack_max * H
        self.GAP_SPLIT = params.gap_split * H       # not referenced -- see SnapParams docstring
        self.BRIDGE_GAP_MAX = params.bridge_gap_max * H
        self.MIN_PARALLEL = params.min_parallel * H
        self.RESAMPLE_STEP = params.resample_step * H
        self.TANGENT_CHORD = params.tangent_chord * H
        self.JUNCTION_ZONE_R = params.junction_zone_r * H
        self.SPUR_MIN = params.spur_min * H
        self.EXIT_R = params.exit_r * H
        self.JUNCTION_CONTRACT_R = params.junction_contract_r * H
        self.CLOSE_R = max(1, int(round(params.close_radius * H)))
        self.MIN_COMPONENT_AREA = params.min_component_area * H * H
        self.MIN_HOLE_AREA = params.min_hole_area * H * H
        self.ENDPOINT_SNAP_R = params.endpoint_snap_r * H   # not referenced
        self.Q_ENDPOINT_HYP = params.q_endpoint_hyp * H     # not referenced

        self._build_candidate_graph()

    # -- candidate graph construction ------------------------------------

    def edge_geom(self, a: str, b: str):
        """Geometric endpoints of the hexside segment between adjacent hex
        centers ``a``/``b``: the perpendicular bisector of the center-to-
        center vector, length ``H``, centered on the midpoint."""
        ax, ay = self.centers[a]
        bx, by = self.centers[b]
        mx, my = (ax + bx) / 2, (ay + by) / 2
        dx, dy = bx - ax, by - ay
        d = math.hypot(dx, dy)
        ex, ey = -dy / d, dx / d
        p1 = (mx - ex * self.H / 2, my - ey * self.H / 2)
        p2 = (mx + ex * self.H / 2, my + ey * self.H / 2)
        return p1, p2

    def _build_candidate_graph(self):
        """Candidate land-land hexsides (spec S1.1). Neighbor detection is
        the 6 geometrically nearest hex centers, accepted only within
        ``[nbr_lo*H, nbr_hi*H]`` -- this sidesteps hand-deriving a parity-
        sensitive even-q neighbor table by relying on the fact that all six
        true neighbor directions are equidistant on a REGULAR flat-top hex
        lattice (``sqrt(3)*H``). For a grid whose ``row_pitch`` deviates
        from the ideal ``2/sqrt(3) * col_pitch`` ratio (see
        ``hexgrid.check_geometry_ratio``), that equidistance degrades --
        widen ``SnapParams.nbr_lo``/``nbr_hi`` to compensate for a mildly
        irregular grid; for a badly irregular grid, re-anchor the grid fit
        first."""
        H = self.H
        p = self.params
        nbr_lo, nbr_hi = p.nbr_lo * H, p.nbr_hi * H

        n_neighbors = min(7, len(self.VALID))
        d7, i7 = self.ctree.query(self.cxy, k=n_neighbors)
        pair_set = set()
        for i in range(len(self.VALID)):
            for jj in range(1, n_neighbors):
                j = i7[i, jj]
                dd = d7[i, jj]
                if nbr_lo <= dd <= nbr_hi:
                    a, b = self.VALID[i], self.VALID[j]
                    pair_set.add((a, b) if a < b else (b, a))
        self.PAIRS = sorted(pair_set)

        # merge coincident geometric endpoints into lattice vertices
        all_pts = []
        for (a, b) in self.PAIRS:
            p1, p2 = self.edge_geom(a, b)
            all_pts.append(p1)
            all_pts.append(p2)
        all_pts_arr = np.array(all_pts) if all_pts else np.zeros((0, 2))
        vert_pos: dict = {}
        point_to_vertex: dict = {}
        if len(all_pts_arr):
            ptree = cKDTree(all_pts_arr)
            eps = p.vertex_merge_tol * H
            close_pairs = ptree.query_pairs(r=eps)
            parent = list(range(len(all_pts_arr)))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry

            for i, j in close_pairs:
                union(i, j)
            clusters: dict = defaultdict(list)
            for i in range(len(all_pts_arr)):
                clusters[find(i)].append(i)
            for vid, (root, members) in enumerate(clusters.items()):
                pos = all_pts_arr[members].mean(axis=0)
                vert_pos[vid] = pos
                for m in members:
                    point_to_vertex[m] = vid
        self.vert_pos = vert_pos

        edges = []
        for k, (a, b) in enumerate(self.PAIRS):
            va = point_to_vertex[2 * k]
            vb = point_to_vertex[2 * k + 1]
            pa = vert_pos[va]
            pb = vert_pos[vb]
            edges.append({"idx": k, "a": a, "b": b, "va": va, "vb": vb,
                         "pa": np.array(pa), "pb": np.array(pb)})
        self.EDGES = edges

        vert_edges: dict = defaultdict(list)
        for e in edges:
            vert_edges[e["va"]].append(e["idx"])
            vert_edges[e["vb"]].append(e["idx"])
        self.VERT_EDGES = vert_edges

        self.MIDS = (np.array([((e["pa"][0] + e["pb"][0]) / 2, (e["pa"][1] + e["pb"][1]) / 2)
                               for e in edges]) if edges else np.zeros((0, 2)))
        self.MID_TREE = cKDTree(self.MIDS) if len(self.MIDS) else None

        if vert_pos:
            self.NEAREST_VERT_TREE = cKDTree(np.array(list(vert_pos.values())))
            self.NEAREST_VERT_IDS = list(vert_pos.keys())
        else:
            self.NEAREST_VERT_TREE = None
            self.NEAREST_VERT_IDS = []

    def state_geom(self, eidx: int, orient: int):
        e = self.EDGES[eidx]
        if orient == 0:
            tail, head, pt, ph = e["va"], e["vb"], e["pa"], e["pb"]
        else:
            tail, head, pt, ph = e["vb"], e["va"], e["pb"], e["pa"]
        d = ph - pt
        L = float(np.hypot(*d))
        u_hat = d / L
        return tail, head, pt, ph, u_hat, L

    def make_state(self, eidx: int, orient: int) -> St:
        s = St()
        s.eidx = eidx
        s.orient = orient
        s.tail, s.head, s.pt, s.ph, s.u_hat, s.L = self.state_geom(eidx, orient)
        return s

    def nearest_vertex(self, pt):
        """Nearest lattice vertex id + distance to ``pt``. Not used
        internally by the decode pipeline (see :class:`SnapParams`
        docstring); exposed for callers doing their own junction work."""
        if self.NEAREST_VERT_TREE is None:
            return None, float("inf")
        dv, iv = self.NEAREST_VERT_TREE.query(pt)
        return self.NEAREST_VERT_IDS[iv], dv

    def is_boundary_exit(self, pt) -> bool:
        """A map/playable-boundary exit (spec S1.3): within ``EXIT_R`` of the
        raster boundary (exact), or farther from the nearest lattice vertex
        than any real interior point would be (approximated as
        ``> 1.3H`` -- a proxy for "off the edge of the numbered map, into
        ocean/blank space")."""
        IW, IH = self.grid.image_full
        x, y = pt
        if x <= self.EXIT_R or y <= self.EXIT_R or x >= IW - self.EXIT_R or y >= IH - self.EXIT_R:
            return True
        if self.NEAREST_VERT_TREE is None:
            return True
        dv, _ = self.NEAREST_VERT_TREE.query(pt)
        return dv > 1.3 * self.H

    # -- mask preprocessing ------------------------------------------------

    @staticmethod
    def load_mask(mask) -> np.ndarray:
        """Accept a boolean/uint8 ndarray directly, or a path to a trace
        image. Paths with an alpha channel use ``alpha > 0`` as the trace
        (the GotA convention: RGBA trace PNGs where alpha encodes the 1-bit
        hand-traced line). Grayscale/1-bit paths use any nonzero pixel;
        plain RGB paths without alpha fall back to any non-black pixel."""
        if isinstance(mask, np.ndarray):
            return mask.astype(bool)
        img = Image.open(mask)
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return arr[:, :, 3] > 0
        if arr.ndim == 2:
            return arr > 0
        return arr[..., :3].sum(axis=-1) > 0

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        m = binary_closing(mask, disk(self.CLOSE_R))
        m = remove_small_objects(m, min_size=int(self.MIN_COMPONENT_AREA))
        m = remove_small_holes(m, area_threshold=int(self.MIN_HOLE_AREA))
        return m

    # -- skeleton -> links ---------------------------------------------------

    def decompose_links(self, pts, degree):
        """Decompose the skeleton pixel graph into maximal links: endpoint-
        to-junction, junction-to-junction, endpoint-to-endpoint, plus
        deterministic break-to-break links for closed loops with no junction
        pixels at all (spec S1.3.11)."""
        node_pixels = [p for p in pts if degree.get(p, 0) != 2]
        pix_to_cluster, cluster_centroid = cluster_nodes(node_pixels, self.JUNCTION_CONTRACT_R)
        node_set = set(node_pixels)

        def nbrs(p):
            x, y = p
            out = []
            for dx, dy in OFFS8:
                q = (x + dx, y + dy)
                if q in pts:
                    out.append(q)
            return out

        links = []
        used_steps = set()
        visited_ordinary = set()
        for p in node_pixels:
            for n in nbrs(p):
                if (p, n) in used_steps:
                    continue
                if n in node_set:
                    if (n, p) in used_steps:
                        continue
                    used_steps.add((p, n))
                    used_steps.add((n, p))
                    links.append({"chain": [p, n], "from_cluster": pix_to_cluster[p],
                                 "to_cluster": pix_to_cluster[n]})
                    continue
                if n in visited_ordinary:
                    continue
                chain = [p, n]
                visited_ordinary.add(n)
                prev, cur = p, n
                steps = 0
                while degree.get(cur, 0) == 2 and steps < 2_000_000:
                    cn = [q for q in nbrs(cur) if q != prev]
                    if not cn:
                        break
                    nxt = cn[0]
                    chain.append(nxt)
                    if degree.get(nxt, 0) == 2:
                        visited_ordinary.add(nxt)
                    prev, cur = cur, nxt
                    steps += 1
                used_steps.add((p, n))
                to_cluster = pix_to_cluster.get(cur)
                links.append({"chain": chain, "from_cluster": pix_to_cluster[p],
                             "to_cluster": to_cluster if to_cluster is not None else -1,
                             "open_end": to_cluster is None})

        # closed loops: components with no node (junction/endpoint) pixels at all
        visited_all = set()
        for link in links:
            visited_all.update(link["chain"])
        remaining = pts - visited_all
        while remaining:
            start = min(remaining)  # deterministic
            chain = [start]
            prev, cur = None, start
            seen = {start}
            while True:
                nxt = None
                for q in nbrs(cur):
                    if q == prev:
                        continue
                    if q not in seen:
                        nxt = q
                        break
                if nxt is None:
                    if prev is not None and start in nbrs(cur) and len(chain) > 2:
                        chain.append(start)
                    break
                chain.append(nxt)
                seen.add(nxt)
                prev, cur = cur, nxt
                if len(chain) > 5_000_000:
                    break
            cid_lo = ("loop", start)
            cluster_centroid[cid_lo] = np.array(start, dtype=float)
            links.append({"chain": chain, "from_cluster": cid_lo, "to_cluster": cid_lo,
                         "closed_loop": True})
            remaining -= set(chain)

        return links, cluster_centroid

    def prune_short_spurs(self, links, cluster_centroid):
        """Drop spur links shorter than ``SPUR_MIN`` unless the free end is a
        map/playable-boundary exit."""
        kept = []
        for link in links:
            chain = link["chain"]
            is_spur = link.get("to_cluster", -1) == -1
            if not is_spur:
                kept.append(link)
                continue
            length = poly_arclen(chain)
            endpt = np.array(chain[-1], dtype=float)
            if length >= self.SPUR_MIN or self.is_boundary_exit(endpt):
                kept.append(link)
        return kept

    def resample_link(self, chain, junction_zone_pts):
        pts = np.array(chain, dtype=float)
        seglen = np.hypot(*np.diff(pts, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        total = cum[-1]
        if total < 1e-6:
            return None
        n = max(2, int(total // self.RESAMPLE_STEP) + 1)
        s_vals = np.linspace(0, total, n)
        xs = np.interp(s_vals, cum, pts[:, 0])
        ys = np.interp(s_vals, cum, pts[:, 1])
        samples = np.column_stack([xs, ys])

        tangents = np.zeros_like(samples)
        for i in range(n):
            s0 = max(0, s_vals[i] - self.TANGENT_CHORD / 2)
            s1 = min(total, s_vals[i] + self.TANGENT_CHORD / 2)
            x0 = np.interp(s0, cum, pts[:, 0]); y0 = np.interp(s0, cum, pts[:, 1])
            x1 = np.interp(s1, cum, pts[:, 0]); y1 = np.interp(s1, cum, pts[:, 1])
            dx, dy = x1 - x0, y1 - y0
            d = math.hypot(dx, dy)
            tangents[i] = (dx / d, dy / d) if d > 1e-6 else (1.0, 0.0)

        delta_s = np.diff(s_vals, prepend=s_vals[0])
        delta_s[0] = 0.0

        jz, jtree = junction_zone_pts
        junction_flag = np.zeros(n, dtype=bool)
        if jtree is not None and len(jz):
            dj, _ = jtree.query(samples)
            junction_flag = dj <= self.JUNCTION_ZONE_R

        return samples, tangents, delta_s, junction_flag, s_vals

    # -- candidate states / costs -------------------------------------------

    def get_candidates(self, x):
        if self.MID_TREE is None:
            return []
        idxs = self.MID_TREE.query_ball_point(x, r=2.0 * self.H)
        scored = []
        for ei in idxs:
            e = self.EDGES[ei]
            pa, pb = e["pa"], e["pb"]
            d = pb - pa
            L = float(np.hypot(*d))
            if L < 1e-6:
                continue
            u_hat = d / L
            w = x - pa
            u = float(np.dot(w, u_hat))
            perp = w - u * u_hat
            dperp = float(np.hypot(*perp))
            if dperp > self.R_SEARCH:
                continue
            dend = max(0.0, -u, u - L)
            if dend > self.SLACK:
                continue
            scored.append((ei, dperp, u, dend, L, u_hat))
        scored.sort(key=lambda t: t[1])
        return scored[:self.MAX_CAND_PHYS]

    def emission(self, dperp, dend, theta_deg, junction_zone):
        dist_term = huber(dperp / self.D_EMIT)
        end_term = huber(dend / self.D_END)
        ang_term = huber(math.sin(math.radians(theta_deg)) / SIN30)
        w_ang = self.W_ANG_JUNCTION if junction_zone else self.W_ANG_NORMAL
        return dist_term + self.W_END * end_term + w_ang * ang_term

    def bfs_bridge(self, v_from, v_to, max_hops):
        """Shortest graph-walk distance (in hexside lengths) from ``v_from``
        to ``v_to``, up to ``max_hops`` hidden hexsides. ``None`` if
        unreachable within the hop budget."""
        if v_from == v_to:
            return 0.0
        frontier = {v_from: 0.0}
        for _ in range(max_hops):
            nxt = {}
            for v, d in frontier.items():
                for ei in self.VERT_EDGES[v]:
                    e = self.EDGES[ei]
                    ov = e["vb"] if e["va"] == v else e["va"]
                    nd = d + self.H
                    if ov == v_to:
                        return nd
                    if ov not in nxt or nd < nxt[ov]:
                        nxt[ov] = nd
            frontier = nxt
            if not frontier:
                break
        return None

    def same_edge_cost(self, pu, u_new, ds, junction_zone):
        """Transition type 1: same oriented edge, mostly forward progress."""
        progress = u_new - pu
        if progress < -self.BACKTRACK_MAX:
            if not junction_zone:
                return None
            return self.BACKTRACK_JUNCTION_PENALTY + huber((abs(progress) - ds) / self.D_TRANS)
        return huber((abs(progress) - ds) / self.D_TRANS)

    def transition_cost(self, pst: St, pu, st: St, u_new, ds, junction_zone):
        """Allowed transition types (spec S1.6): (1) same oriented edge,
        (2) adjacent oriented edge sharing a lattice vertex, (3) a short
        bridge across a real trace gap. Everything else returns ``None``
        (forbidden)."""
        if pst.eidx == st.eidx and pst.orient == st.orient:
            return self.same_edge_cost(pu, u_new, ds, junction_zone)
        if pst.eidx == st.eidx and pst.orient != st.orient:
            # immediate reversal on the same physical edge
            if junction_zone:
                return self.REVERSAL_JUNCTION_PENALTY
            return None
        if pst.head == st.tail:
            # adjacent oriented edge, walks through one real shared vertex
            gw = (pst.L - pu) + u_new
            return huber((gw - ds) / self.D_TRANS)
        # bridge across a short gap, only when the skeleton arc gap itself is short
        if ds > self.BRIDGE_GAP_MAX:
            return None
        dist = self.bfs_bridge(pst.head, st.tail, self.BRIDGE_MAX_HOPS)
        if dist is None:
            return None
        gw = (pst.L - pu) + dist + u_new
        tcost = huber((gw - ds) / self.D_TRANS)
        n_hidden = max(1, int(round(dist / self.H)))
        return tcost + self.BRIDGE_PENALTY_PER_EDGE * n_hidden

    def viterbi_link(self, samples, tangents, delta_s, junction_flag):
        """Per-link Viterbi decode.

        Every sample also carries an implicit "cold restart" option (cost =
        emission only, no backpointer) that competes against real
        transitions from the previous decoded sample. **This restart must
        only win when no real transition from any previous state is valid**
        -- i.e. it is a fallback, not a first-class competitor on absolute
        accumulated cost. A restart's cost is always just one emission term;
        letting it compete unconditionally means it always beats any
        multi-step accumulated path and the DP degenerates to restarting at
        every sample (the bug this exact code once had -- see the module
        docstring and the regression test named there). This is also where
        the spec's "split the skeleton link on a >0.75H no-candidate gap"
        ends up happening, implicitly, inside the same DP pass rather than
        as a separate pre-pass.
        """
        n = len(samples)
        cand_per_sample = []
        for i in range(n):
            scored = self.get_candidates(samples[i])
            states = []
            for (ei, dperp, u, dend, L, u_hat) in scored:
                theta = angle_to_edge(tangents[i], u_hat)
                cost = self.emission(dperp, dend, theta, junction_flag[i])
                for orient in (0, 1):
                    st = self.make_state(ei, orient)
                    states.append((st, cost, u if orient == 0 else L - u, dperp, theta))
            cand_per_sample.append(states)

        if all(len(c) == 0 for c in cand_per_sample):
            return None

        dp = [{} for _ in range(n)]      # dp[i][(eidx,orient)] = (cost, backptr_key, u_along, state)
        keyfn = lambda st: (st.eidx, st.orient)

        for i in range(n):
            states = cand_per_sample[i]
            if not states:
                continue
            if i == 0:
                for (st, ecost, u_along, dperp, theta) in states:
                    k = keyfn(st)
                    if k not in dp[0] or ecost < dp[0][k][0]:
                        dp[0][k] = (ecost, None, u_along, st)
                continue
            back = i - 1
            while back >= 0 and not dp[back]:
                back -= 1
            prev = dp[back] if back >= 0 else {}
            ds_eff = samples_arc_gap(samples, back, i) if back >= 0 else delta_s[i]
            for (st, ecost, u_along, dperp, theta) in states:
                # cold-restart is a FALLBACK ONLY -- see docstring above
                best_cost, best_back = None, None
                for pk, (pcost, pback, pu, pst) in prev.items():
                    tcost = self.transition_cost(pst, pu, st, u_along, ds_eff, junction_flag[i])
                    if tcost is None:
                        continue
                    total = pcost + tcost + ecost
                    if best_cost is None or total < best_cost:
                        best_cost, best_back = total, pk
                if best_cost is None:
                    # no valid real transition exists (genuine gap/topology break)
                    best_cost, best_back = ecost, None
                k = keyfn(st)
                if k not in dp[i] or best_cost < dp[i][k][0]:
                    dp[i][k] = (best_cost, best_back, u_along, st)

        last = n - 1
        while last >= 0 and not dp[last]:
            last -= 1
        if last < 0:
            return None
        bestk = min(dp[last], key=lambda k: dp[last][k][0])
        path = []
        i = last
        k = bestk
        while i >= 0:
            if k not in dp[i]:
                i -= 1
                continue
            cost, back, u_along, st = dp[i][k]
            path.append((i, st, u_along))
            if back is None:
                break
            k = back
            i -= 1
            while i >= 0 and k not in dp[i]:
                i -= 1
            if i < 0:
                break
        path.reverse()
        return path

    def decode_link(self, chain, junction_zone_pts):
        r = self.resample_link(chain, junction_zone_pts)
        if r is None:
            return None
        samples, tangents, delta_s, junction_flag, s_vals = r
        path = self.viterbi_link(samples, tangents, delta_s, junction_flag)
        return path, samples, tangents, junction_flag

    # -- per-layer pipeline --------------------------------------------------

    def snap_layer(self, mask, layer_name: str = "layer") -> LayerResult:
        """Decode one trace-mask layer end to end: clean -> skeletonize ->
        decompose into links -> Viterbi-decode each link -> accumulate
        per-edge support -> apply the along-vs-crossing acceptance rule
        (spec S1.8). ``mask`` is a path or boolean ndarray (see
        :meth:`load_mask`)."""
        mask = self.load_mask(mask)
        raw_px = int(mask.sum())
        mask = self.clean_mask(mask)
        cleaned_px = int(mask.sum())
        skel = skeletonize(mask)
        skeleton_px = int(skel.sum())

        pts, degree = build_skeleton_graph(skel)

        links, cluster_centroid = self.decompose_links(pts, degree)
        links = self.prune_short_spurs(links, cluster_centroid)

        jz_pts = np.array(list(cluster_centroid.values())) if cluster_centroid else np.zeros((0, 2))
        jz_tree = cKDTree(jz_pts) if len(jz_pts) else None
        jz_bundle = (jz_pts, jz_tree)

        per_edge_support: dict = defaultdict(list)
        link_frechet = []

        for link in links:
            chain = link["chain"]
            if poly_arclen(chain) < 1e-3:
                continue
            res = self.decode_link(chain, jz_bundle)
            if res is None:
                continue
            path, samples, tangents, junction_flag = res
            if not path:
                continue
            prev_key = None
            decoded_seq = []
            for (i, st, u_along) in path:
                key = (st.eidx, st.orient)
                if key != prev_key:
                    decoded_seq.append(st)
                    prev_key = key
                theta = angle_to_edge(tangents[i], st.u_hat)
                w = samples[i] - st.pt
                u = float(np.dot(w, st.u_hat))
                perp = w - u * st.u_hat
                dperp = float(np.hypot(*perp))
                near_vertex = (min(u, st.L - u) <= 0.25 * self.H)
                step_len = self.RESAMPLE_STEP
                per_edge_support[st.eidx].append(
                    (dperp, theta, step_len, bool(junction_flag[i]), near_vertex))
            if decoded_seq:
                vwalk = [decoded_seq[0].pt] + [s.ph for s in decoded_seq]
                if len(vwalk) >= 2:
                    fdist = discrete_frechet(np.array(chain, dtype=float), np.array(vwalk))
                    link_frechet.append(fdist / self.H)

        # along-vs-crossing acceptance rule (spec S1.8)
        accepted = {}
        suppressed = {}
        for eidx, samples_list in per_edge_support.items():
            Lp = sum(s[2] for s in samples_list if s[1] <= self.ANG_ALONG)
            Lc = sum(s[2] for s in samples_list if s[1] >= self.ANG_CROSS and not s[4])
            thetas = [s[1] for s in samples_list]
            thetamed = float(np.median(thetas)) if thetas else 999.0
            ok = (Lp >= self.MIN_PARALLEL) and (thetamed <= self.ANG_ALONG) and (Lp >= 2 * Lc)
            rec = {"Lparallel": Lp, "Lcross": Lc, "theta_med": thetamed,
                  "n_samples": len(samples_list)}
            if ok:
                accepted[eidx] = rec
            else:
                suppressed[eidx] = rec

        frechet_arr = np.array(link_frechet) if link_frechet else np.array([0.0])
        diagnostics = {
            "layer": layer_name,
            "mask_px_raw": raw_px,
            "mask_px_cleaned": cleaned_px,
            "skeleton_px": skeleton_px,
            "n_links": len(links),
            "n_links_decoded": len(link_frechet),
            "accepted_edges": len(accepted),
            "suppressed_candidates": len(suppressed),
            "frechet_dF_over_H": {
                "median": float(np.median(frechet_arr)),
                "p90": float(np.percentile(frechet_arr, 90)),
                "max": float(np.max(frechet_arr)),
            },
            "accepted_detail": {f"{self.EDGES[e]['a']}-{self.EDGES[e]['b']}": v
                               for e, v in accepted.items()},
            "suppressed_detail": {f"{self.EDGES[e]['a']}-{self.EDGES[e]['b']}": v
                                 for e, v in list(suppressed.items())[:200]},
        }

        edges_out = sorted(
            [{"a": self.EDGES[e]["a"], "b": self.EDGES[e]["b"]} for e in accepted],
            key=lambda o: (o["a"], o["b"]),
        )

        return LayerResult(layer=layer_name, mask=mask, accepted=accepted,
                           suppressed=suppressed, diagnostics=diagnostics,
                           edges_out=edges_out)

    def snap_layers(self, layers: dict) -> dict[str, LayerResult]:
        """Decode every layer in ``layers`` (``{name: path_or_mask_array}``),
        reusing this snapper's candidate graph."""
        return {name: self.snap_layer(mask, layer_name=name) for name, mask in layers.items()}

    @staticmethod
    def to_hexwright_json(results: dict[str, LayerResult]) -> dict:
        """``{layer: [{"a":CCRR,"b":CCRR}, ...]}`` -- the Hexwright-
        importable grouped hexside shape (matches
        ``store.importHexsides()`` on the origin project)."""
        return {name: res.edges_out for name, res in results.items()}

    # -- verification overlay ------------------------------------------------

    def render_overlay(self, board_img_path, mask: np.ndarray, accepted: dict,
                       suppressed: dict, out_path, scale: float = 1.0, crop=None) -> str:
        """The operator-verification overlay: magenta = hand trace, green =
        high-confidence accepted hexside, amber = accepted-but-lower-
        confidence, red = suppressed crossing/ambiguous candidate. **This
        render IS the acceptance check** for hexside-snap (spec S2.7) --
        counts never certify a run; the operator's eyes on this render do."""
        IW, IH = self.grid.image_full
        if crop:
            x0, y0, x1, y1 = crop
        else:
            x0, y0, x1, y1 = 0, 0, IW, IH
        w, h = x1 - x0, y1 - y0
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))

        base = Image.open(board_img_path).convert("RGB")
        base_crop = base.crop((x0, y0, x1, y1)).resize((sw, sh), Image.BILINEAR)
        img = base_crop.convert("RGBA")

        mcrop = mask[y0:y1, x0:x1]
        mimg = Image.fromarray((mcrop * 255).astype(np.uint8)).resize((sw, sh), Image.BILINEAR)
        marr = np.zeros((sh, sw, 4), dtype=np.uint8)
        marr[..., 0] = 255
        marr[..., 1] = 0
        marr[..., 2] = 200
        marr[..., 3] = (np.asarray(mimg) > 40).astype(np.uint8) * 140
        img = Image.alpha_composite(img, Image.fromarray(marr))

        draw = ImageDraw.Draw(img)

        def to_px(pt):
            return ((pt[0] - x0) * scale, (pt[1] - y0) * scale)

        for eidx in suppressed:
            e = self.EDGES[eidx]
            p1, p2 = to_px(e["pa"]), to_px(e["pb"])
            in1 = 0 <= p1[0] <= sw and 0 <= p1[1] <= sh
            in2 = 0 <= p2[0] <= sw and 0 <= p2[1] <= sh
            if in1 or in2:
                draw.line([p1, p2], fill=(220, 30, 30, 200), width=2)

        for eidx, rec in accepted.items():
            e = self.EDGES[eidx]
            p1, p2 = to_px(e["pa"]), to_px(e["pb"])
            # NOTE: a single hexside only ever accumulates ~0.75-1.0H of
            # parallel support (3-4 resampled points at 0.25H spacing along
            # an H-long edge), so "confident" is scaled to that, not to a
            # multi-edge run.
            confident = rec["Lparallel"] >= 0.9 * self.H and rec["theta_med"] <= 15
            color = (40, 200, 60, 230) if confident else (230, 170, 30, 230)
            draw.line([p1, p2], fill=color, width=3)

        img.convert("RGB").save(out_path, quality=90)
        return str(out_path)


# ---------------------------------------------------------------------------
# one-shot convenience wrapper + CLI


def snap_traces(grid: HexGrid, valid_hexes: Iterable[str], layers: dict,
                params: SnapParams | None = None, board_img=None,
                overlay_dir=None, overlay_scale: float = 1.0):
    """One-shot: build the candidate graph, decode every layer, optionally
    write verification overlays. Returns ``(hexwright_json, results)`` where
    ``hexwright_json`` is the Hexwright-importable ``{layer: [{a,b},...]}``
    dict and ``results`` is ``{layer: LayerResult}`` for diagnostics."""
    snapper = HexsideSnapper(grid, valid_hexes, params=params)
    results = snapper.snap_layers(layers)
    if overlay_dir is not None:
        if board_img is None:
            raise ValueError("board_img is required when overlay_dir is set")
        overlay_dir = Path(overlay_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for name, res in results.items():
            snapper.render_overlay(board_img, res.mask, res.accepted, res.suppressed,
                                   overlay_dir / f"overlay-{name}-full.jpg",
                                   scale=overlay_scale)
    return HexsideSnapper.to_hexwright_json(results), results


def _parse_trace_args(items) -> dict:
    layers = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--trace expects LAYER=PATH, got: {item!r}")
        name, path = item.split("=", 1)
        if name in layers:
            raise SystemExit(f"--trace layer {name!r} given more than once")
        layers[name] = path
    return layers


def _load_valid_hexes(terrain_path, hexes_path) -> list:
    if terrain_path:
        data = json.loads(Path(terrain_path).read_text())
        src = data.get("terrain", data) if isinstance(data, dict) else data
        return list(src.keys())
    if hexes_path:
        return json.loads(Path(hexes_path).read_text())
    raise SystemExit("one of --terrain or --hexes-json is required")


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Snap hand-traced linear features (rivers, ridges, impassible "
                    "terrain...) onto a hex-lattice hexside graph via HMM/Viterbi "
                    "map-matching. See docs/CONVENTIONS.md 'Hexside-snap'.")
    ap.add_argument("--grid", required=True, help="HexGrid JSON (HexGrid.to_json() output)")
    ap.add_argument("--terrain",
                    help="terrain JSON ({'terrain':{CCRR:cls}} or {CCRR:cls}) -- "
                         "keys become the valid/eligible hex set")
    ap.add_argument("--hexes-json",
                    help="alternative to --terrain: a JSON list of valid CCRR hex codes")
    ap.add_argument("--trace", action="append", metavar="LAYER=PATH", required=True,
                    help="a trace-mask layer, repeatable, e.g. --trace rivers=rivers-trace.png")
    ap.add_argument("--out", required=True, help="output path for the Hexwright-importable JSON")
    ap.add_argument("--diagnostics", help="optional path to write full per-layer diagnostics JSON")
    ap.add_argument("--board", help="board raster path -- required if --overlay is set")
    ap.add_argument("--overlay", metavar="DIR",
                    help="write verification overlays (magenta/green/amber/red) to this directory")
    ap.add_argument("--overlay-scale", type=float, default=1.0)
    args = ap.parse_args()

    grid = HexGrid.from_json(args.grid)
    valid_hexes = _load_valid_hexes(args.terrain, args.hexes_json)
    layers = _parse_trace_args(args.trace)

    hexwright_json, results = snap_traces(
        grid, valid_hexes, layers,
        board_img=args.board, overlay_dir=args.overlay, overlay_scale=args.overlay_scale,
    )

    Path(args.out).write_text(json.dumps(hexwright_json, indent=2))
    n = sum(len(v) for v in hexwright_json.values())
    print(f"Wrote {n} hexsides across {len(hexwright_json)} layer(s) -> {args.out}")

    if args.diagnostics:
        diag = {name: res.diagnostics for name, res in results.items()}
        Path(args.diagnostics).write_text(json.dumps(diag, indent=2))
        print(f"Wrote diagnostics -> {args.diagnostics}")

    if args.overlay:
        print(f"Wrote overlays -> {args.overlay}  "
             f"(magenta=trace, green=confident, amber=accepted-low-confidence, "
             f"red=suppressed -- THIS is the acceptance check, review before trusting)")


if __name__ == "__main__":
    _main()
