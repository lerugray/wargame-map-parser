# wargame-map-parser

Extract per-hex terrain from a scanned wargame map — turn a board image into a
`hex → terrain` table you can drive a digital game with.

Built while digitizing the map for *The World Undone: East Prussia 1914*. The
method generalizes to any flat-top hex wargame map with printed coordinate
numbers.

## Why this is its own thing

The naive approach — pick absolute colour thresholds ("blue enough = lake") —
**breaks on every new map**, because each map's palette is different and its
terrain types are only defined *relative to each other*. This tool instead:

1. **De-duplicates multi-sheet boards.** Boxed maps are printed across sheets
   that share an overlap strip (the same hexes on both inner margins). Scanning
   them edge-to-edge prints that band twice — the same hex numbers repeat near
   the join. `seams` finds the duplicated band and rebuilds the board so every
   column appears once, collapsing the calibration from an ugly two-segment
   "jog" into one clean line.
2. **Calibrates the hex grid** from a handful of read-off hex numbers
   (`hexgrid.fit_from_anchors`) — an affine `(col,row) → pixel` model.
3. **Classifies terrain by reference hexes, not thresholds**
   (`classify.ReferenceClassifier`). Label one confident exemplar of each type
   (a known clear hex, a known forest hex, a known lake/sea hex…); every other
   hex is assigned its nearest exemplar in a self-calibrating feature space.
4. **Makes you look at the result** (`overlay`) — confident nonsense is the
   failure mode, so the answer is always verified visually, never trusted from
   counts.

## The feature space (what actually separates terrains)

| Feature | Separates |
|---|---|
| **mean RGB (hue)** | water/sea (blue) · forest (green) · clear (cream) |
| **colour variance** | *solid* fills (lake, sea — low variance) vs *printed symbols* (forest, swamp — high variance) of the same hue |
| **morphology** (blob shape) | **forest = circular "bulbs"** vs **swamp = "lines"** (dashes/tussocks) — colour can't tell these apart on a cream palette; blob elongation can |

That last row is the trick that rescues the case colour fails: forest tree
symbols are compact blobs, swamp marks are elongated. (Credit: Ray Weiss, who
also contributed the reference-hex idea itself.)

## Hard limit: hexside terrain

Full-hex classification **cannot** capture terrain drawn on hex *edges* —
lakes-on-hexsides, rivers, escarpments. On real maps the most important water
(e.g. the Masurian Lakes) often runs along hexsides, so those hexes are
half-water/half-land and *no full-hex label is right*. Detect and confine those
to a region, and model them in a **separate edge layer** — don't force them into
a full-hex type. The tool flags this rather than guessing.

## Install

```bash
pip install -r requirements.txt      # numpy + Pillow only
```

## Quick start

```python
from parser import (fix_sheets, fit_from_anchors, ReferenceClassifier,
                    load_image, draw_terrain)

# 1. de-duplicate a two-sheet board (writes board-full.jpg)
info = fix_sheets("EP_left.jpg", "EP_right.jpg", "board-full.jpg")
print(info)   # {'overlap_px': ..., 'out_size': (W, H), ...}  -- verify vs calibration

# 2. calibrate from a few read-off hex centers (>=2 cols, >=2 rows, span the board)
grid = fit_from_anchors([
    {"col": 1,  "row": 8,  "x": 174,  "y": 1484},
    {"col": 29, "row": 8,  "x": 3088, "y": 1484},
    {"col": 47, "row": 24, "x": 4961, "y": 3407},
    # ... include >=2 EVEN-column anchors with their (down-shifted) centers
], image_full=(6518, 5139), web_scale=0.5)

# 3. classify by reference hexes
arr = load_image("board-full.jpg")
clf = ReferenceClassifier(grid).fit(arr, {
    "clear":  ["0510", "2010"],
    "forest": ["2926", "3431"],
    "swamp":  ["1514", "1614"],
    "water":  ["0140", "0240"],     # a CONFIDENT blue sample -- a bad exemplar poisons everything
})
terrain = clf.classify_all(arr, ["3115", "4022", "2010"])

# 4. LOOK at it
draw_terrain("board-web.jpg", grid, terrain, "check.png")
```

See [`examples/twu/`](examples/twu/) for the full East Prussia case study
(seam fix + calibration + lake/swamp cleanup), and [`SKILL.md`](SKILL.md) for
the agent-readable method.

## Caveats (read before trusting output)

- **`detect_overlap` returns an estimate.** Confirm it against the calibration:
  the right overlap is the one that makes the two-segment x-jog disappear.
- **The fit is only as good as the anchors.** Include ≥2 even-column anchors
  with their actual (down-shifted) centers, or the even-column offset comes out
  wrong.
- **Reference-matching is reliable for hue/texture (is-this-water), less so for
  sorting non-water noise into forest/swamp** — dark town icons can match the
  forest centroid. Default ambiguous noise to clear and verify with the overlay.
- **Always run the overlay.** Counts lie; pictures don't.

## Licence

MIT.
