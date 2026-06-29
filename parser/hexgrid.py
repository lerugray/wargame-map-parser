"""Hex-grid calibration: map a printed-board hex coordinate (CCRR) to a pixel
center, and back.

A wargame map is a grid of hexes with printed coordinate numbers. To extract
per-hex terrain we need a function ``(col, row) -> (x, y)`` that lands on each
hex's center in the scan. For a clean scan that is an affine model:

    x = x_intercept + col * col_pitch_x
    y = y_intercept + row * row_pitch_y + (even_col_y_offset if col is even else 0)

(flat-top hexes, "even-q" offset — even columns shifted down half a row). The
model is derived once from a handful of known hex->pixel anchors (read a few
printed numbers off the scan) via least squares, then reused for every hex.

Why affine is usually enough: on a flat scan the only systematic error is a
seam where two printed map sheets were joined (see ``seams.py``) — fix the
image and the model collapses to a single line.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Iterable


def parse_ccrr(hexcode: str) -> tuple[int, int]:
    """'3115' -> (col=31, row=15). Assumes 2-digit col + 2-digit row."""
    hexcode = str(hexcode)
    return int(hexcode[:2]), int(hexcode[2:])


def to_ccrr(col: int, row: int) -> str:
    return f"{col:02d}{row:02d}"


@dataclass
class HexGrid:
    """Affine CCRR<->pixel calibration for a flat-top, even-q offset hex map."""

    image_full: tuple[int, int]          # (width, height) of the full-res scan
    col_pitch_x: float                   # horizontal px between adjacent columns
    row_pitch_y: float                   # vertical px between adjacent rows
    x_intercept_col0: float              # x of the (hypothetical) column 0 center
    y_intercept_row0: float              # y of the (hypothetical) row 0 center
    even_col_y_offset: float = 0.0       # even columns shifted down by this many px
    web_scale: float = 1.0               # board-web.jpg = web_scale * board-full
    orientation: str = "flat-top"
    offset_scheme: str = "even-q"        # even columns shifted down

    def center(self, col: int, row: int) -> tuple[float, float]:
        """Full-image pixel center of hex (col, row)."""
        x = self.x_intercept_col0 + col * self.col_pitch_x
        y = self.y_intercept_row0 + row * self.row_pitch_y
        if col % 2 == 0:
            y += self.even_col_y_offset
        return x, y

    def center_web(self, col: int, row: int) -> tuple[float, float]:
        x, y = self.center(col, row)
        return x * self.web_scale, y * self.web_scale

    def center_of(self, hexcode: str, web: bool = False) -> tuple[float, float]:
        col, row = parse_ccrr(hexcode)
        return self.center_web(col, row) if web else self.center(col, row)

    def hex_size(self) -> float:
        """Center-to-vertex radius (full-res). For flat-top hexes the column
        pitch is 3/4 of the hex width (= 3/4 * 2 * size)."""
        return self.col_pitch_x / 0.75 / 2.0

    def polygon(self, col: int, row: int, web: bool = False, size: float | None = None):
        """Vertices of the flat-top hex outline, for drawing overlays."""
        cx, cy = (self.center_web(col, row) if web else self.center(col, row))
        s = (size if size is not None else self.hex_size()) * (self.web_scale if web else 1.0)
        return [(cx + s * math.cos(math.radians(a)), cy + s * math.sin(math.radians(a)))
                for a in (0, 60, 120, 180, 240, 300)]

    # ----- (de)serialization -------------------------------------------------
    def to_json(self, path: str | None = None) -> dict:
        d = asdict(self)
        d["image_full"] = list(self.image_full)
        d["formula"] = ("x = {xi} + col*{cp}; y = {yi} + row*{rp} + "
                        "((col%2==0)?{eo}:0)").format(
            xi=self.x_intercept_col0, cp=self.col_pitch_x,
            yi=self.y_intercept_row0, rp=self.row_pitch_y, eo=self.even_col_y_offset)
        if path:
            with open(path, "w") as f:
                json.dump(d, f, indent=2)
        return d

    @classmethod
    def from_json(cls, path_or_dict) -> "HexGrid":
        d = path_or_dict
        if isinstance(d, str):
            with open(d) as f:
                d = json.load(f)
        # tolerate the richer schema some projects carry (x_model.left/right etc.)
        xm = d.get("x_model", {})
        ym = d.get("y_model", {})
        return cls(
            image_full=tuple(d["image_full"]),
            col_pitch_x=d.get("col_pitch_x") or xm.get("col_pitch_x"),
            row_pitch_y=d.get("row_pitch_y") or ym.get("row_pitch_y"),
            x_intercept_col0=d.get("x_intercept_col0")
                or xm.get("x_intercept_col0")
                or (xm.get("left") or {}).get("x_intercept_col0"),
            y_intercept_row0=d.get("y_intercept_row0") or ym.get("y_intercept_row0"),
            even_col_y_offset=d.get("even_col_y_offset")
                or ym.get("even_col_down_offset", 0.0),
            web_scale=d.get("web_scale", 1.0),
            orientation=d.get("orientation", "flat-top"),
            offset_scheme=d.get("offset_scheme", "even-q"),
        )


def fit_from_anchors(anchors: Iterable[dict], image_full: tuple[int, int],
                     web_scale: float = 1.0) -> HexGrid:
    """Least-squares fit of the affine model from known hex->pixel anchors.

    Each anchor is ``{"col": int, "row": int, "x": px, "y": px}`` — read a few
    printed hex numbers off the scan and click their centers. You need at least
    two distinct columns and two distinct rows; a handful spread across the
    board (and across any seam) is better. Returns a calibrated HexGrid.

    ⚠ OFF-BY-ONE TRAP (the TWU −1-row bug): the anchor (col,row) MUST be the number
    PRINTED in that hex — NOT eyeballed from geometry. A *uniform* mislabel (e.g. every
    anchor read one row too low) yields a PERFECT least-squares fit (≈0 residual, lands on
    real hexes) yet every computed center is one hex off the printed numbering — invisible
    to the fit and to a "does it land on a hex?" glance. ALWAYS cross-check the result with
    ``verify_against_printed()`` (or ``overlay.draw_centers`` read against the printed
    numbers) at hexes spread top/middle/bottom before trusting the parse.
    """
    import numpy as np

    a = list(anchors)
    cols = np.array([p["col"] for p in a], float)
    rows = np.array([p["row"] for p in a], float)
    xs = np.array([p["x"] for p in a], float)
    ys = np.array([p["y"] for p in a], float)

    # x = x0 + col*cp   ->   [1, col] . [x0, cp]
    Ax = np.column_stack([np.ones_like(cols), cols])
    x0, cp = np.linalg.lstsq(Ax, xs, rcond=None)[0]

    # y = y0 + row*rp + even*offset   ->   [1, row, (col even)] . [y0, rp, offset]
    even = (cols % 2 == 0).astype(float)
    Ay = np.column_stack([np.ones_like(rows), rows, even])
    y0, rp, eo = np.linalg.lstsq(Ay, ys, rcond=None)[0]

    return HexGrid(
        image_full=tuple(image_full), col_pitch_x=float(cp), row_pitch_y=float(rp),
        x_intercept_col0=float(x0), y_intercept_row0=float(y0),
        even_col_y_offset=float(eo), web_scale=web_scale,
    )


def verify_against_printed(grid: "HexGrid", truth_anchors: Iterable[dict],
                           tol_frac: float = 0.4) -> list[dict]:
    """Catch a SYSTEMATIC calibration offset (e.g. an off-by-one-row anchor mislabel)
    that the least-squares fit in ``fit_from_anchors`` CANNOT see.

    A uniform shift in the anchor CCRR labels (say every anchor read one row too low)
    produces a perfect fit — low residual, lands on real hexes — yet every computed center
    is one hex off the PRINTED number. The only way to detect it is to compare
    ``grid.center(col,row)`` against pixels read INDEPENDENTLY off the printed numbers
    (ideally hexes NOT used as fit anchors), spread across the board.

    truth_anchors: iterable of ``{"col","row","x","y"}`` where col/row are exactly what is
      PRINTED in that hex and x,y are its pixel center (full-image space).
    Returns a list of mismatches ``[{"ccrr","expected_px","got_px","dist_px"}]`` whose
    distance exceeds ``tol_frac * row_pitch_y``. **Empty list = calibration matches the
    printed numbering. Non-empty = the grid is off (very likely a whole-row/col shift) —
    do NOT trust the parse until it is empty.** This is the gate that would have caught the
    TWU −1-row bug, which fit cleanly but rendered/sampled every hex one row off the print.
    """
    tol = tol_frac * grid.row_pitch_y
    out = []
    for a in truth_anchors:
        col, row = int(a["col"]), int(a["row"])
        ex, ey = grid.center(col, row)
        gx, gy = float(a["x"]), float(a["y"])
        dist = ((ex - gx) ** 2 + (ey - gy) ** 2) ** 0.5
        if dist > tol:
            out.append({"ccrr": to_ccrr(col, row),
                        "expected_px": (round(ex, 1), round(ey, 1)),
                        "got_px": (round(gx, 1), round(gy, 1)),
                        "dist_px": round(dist, 1)})
    return out
