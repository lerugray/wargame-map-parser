"""Detect and fix duplicated-band seams from multi-sheet wargame maps.

A boxed wargame map is often printed across several sheets that **share an
overlap strip** — the same band of hexes printed on both sheets' inner margins
so players can align them on a table. If a digitizer concatenates the scans
edge-to-edge, that shared band appears TWICE, side by side: the same hex
numbers repeat near the join, and any hex-grid calibration needs an ugly
two-segment "jog" to cope.

This module finds the duplicated band width and rebuilds the board with each
column appearing exactly once — after which the calibration is a single clean
affine line.

Canonical case (TWU East Prussia): left sheet 3338w + right sheet 3339w
concatenated to 6677w duplicated column 31 (hexes 31xx); dropping the 159px
duplicate band gave a 6518w board and collapsed a two-segment x-model into one.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _gray(img: np.ndarray) -> np.ndarray:
    return img[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])


def detect_overlap(left: np.ndarray, right: np.ndarray,
                   strip_w: int = 24, search: int = 400,
                   band: tuple[float, float] = (0.25, 0.75)) -> dict:
    """Find how many px of the RIGHT sheet duplicate the LEFT sheet's content.

    Takes a vertical strip from the left sheet's right edge and slides it across
    the right sheet's left region, scoring by mean absolute difference over a
    central vertical band (avoids irregular top/bottom map edges). The best
    offset is the width of the duplicated band to drop from the right sheet.

    Returns ``{"overlap_px", "score", "scores"}``. A *low* score at a clear
    minimum = a real duplicate; a flat/high profile = probably no overlap.
    """
    lg, rg = _gray(left), _gray(right)
    h = min(lg.shape[0], rg.shape[0])
    y0, y1 = int(band[0] * h), int(band[1] * h)
    templ = lg[y0:y1, -strip_w:]
    scores = []
    for d in range(0, min(search, rg.shape[1] - strip_w)):
        cand = rg[y0:y1, d:d + strip_w]
        scores.append(float(np.abs(templ - cand).mean()))
    scores = np.array(scores)
    overlap = int(scores.argmin())
    return {"overlap_px": overlap, "score": float(scores[overlap]),
            "scores": scores}


def stitch(left: np.ndarray, right: np.ndarray, overlap_px: int) -> np.ndarray:
    """Concatenate the two sheets, dropping the right sheet's duplicate band."""
    h = min(left.shape[0], right.shape[0])
    return np.hstack([left[:h], right[:h, overlap_px:]])


def fix_sheets(left_path: str, right_path: str, out_path: str,
               strip_w: int = 24, search: int = 400) -> dict:
    """End-to-end: load two sheet scans, detect the duplicate band, write the
    de-duplicated board. Returns the detection result + final size.
    """
    left = np.asarray(Image.open(left_path).convert("RGB"))
    right = np.asarray(Image.open(right_path).convert("RGB"))
    det = detect_overlap(left, right, strip_w=strip_w, search=search)
    board = stitch(left, right, det["overlap_px"])
    Image.fromarray(board).save(out_path, quality=92)
    det["out_size"] = (int(board.shape[1]), int(board.shape[0]))
    det.pop("scores", None)
    return det
