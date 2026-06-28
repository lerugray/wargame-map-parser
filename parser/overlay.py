"""Visual verification overlays.

Terrain extraction must be *looked at*, not trusted from counts — a wrong
exemplar or a mis-calibrated grid produces confident nonsense. These helpers
render the two checks that matter:

- ``draw_centers``  — computed hex centers + labels, to confirm calibration
  lands on the printed hexes (especially across a fixed seam).
- ``draw_terrain`` — per-hex terrain as translucent coloured hexes, to confirm
  the classification matches what the eye sees.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .hexgrid import HexGrid, parse_ccrr

# distinct, legible terrain colours (RGBA)
TERRAIN_COLORS = {
    "clear":  (200, 200, 200, 70),
    "forest": (30, 150, 30, 110),
    "swamp":  (0, 190, 160, 110),
    "lake":   (0, 70, 220, 130),
    "water":  (0, 70, 220, 130),
    "town":   (255, 140, 0, 120),
    "city":   (255, 140, 0, 120),
    "fortress": (200, 0, 0, 120),
}


def _font(size: int):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_terrain(board_img: str, grid: HexGrid, hex_terrain: dict[str, str],
                 out_path: str, web: bool = True, label: bool = False) -> str:
    """Render each hex's terrain as a translucent coloured hex over the board."""
    base = Image.open(board_img).convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    fnt = _font(10)
    for hexcode, terr in hex_terrain.items():
        col, row = parse_ccrr(hexcode)
        color = TERRAIN_COLORS.get(terr, (120, 120, 120, 60))
        poly = grid.polygon(col, row, web=web)
        d.polygon(poly, fill=color, outline=color[:3] + (255,))
        if label:
            cx, cy = grid.center_web(col, row) if web else grid.center(col, row)
            d.text((cx - 12, cy - 5), terr[:2], fill=color[:3] + (255,), font=fnt)
    out = Image.alpha_composite(base, ov).convert("RGB")
    out.save(out_path)
    return out_path


def draw_centers(board_img: str, grid: HexGrid, hexes: list[str],
                 out_path: str, web: bool = True) -> str:
    """Dot + CCRR label at each computed center — the calibration sanity check."""
    base = Image.open(board_img).convert("RGB")
    d = ImageDraw.Draw(base)
    fnt = _font(11)
    for hexcode in hexes:
        col, row = parse_ccrr(hexcode)
        cx, cy = grid.center_web(col, row) if web else grid.center(col, row)
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=(255, 0, 0), width=2)
        d.text((cx + 4, cy - 6), hexcode, fill=(200, 0, 0), font=fnt)
    base.save(out_path)
    return out_path
