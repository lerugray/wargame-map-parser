---
name: parse-wargame-map
description: Extract per-hex terrain from a scanned wargame map (flat-top hex board with printed CCRR coordinates) into a hex→terrain table. Use when digitizing a board wargame's map — de-duplicating multi-sheet scans, calibrating the hex grid, and classifying terrain by reference-hex matching. Backed by the wargame-map-parser repo (numpy + Pillow).
---

# Parsing a wargame map → per-hex terrain

Reference-hex method: classify each hex by **nearest labeled exemplar**, not
absolute colour thresholds (which break per map). Pipeline lives in `parser/`.

## Procedure

1. **De-duplicate the board if it's multi-sheet.** Boxed maps print across
   sheets sharing an overlap strip; edge-to-edge scans duplicate it (same hex
   numbers repeat near the join).
   - `seams.detect_overlap(left, right)` → estimated duplicate-band width.
   - `seams.fix_sheets(left_path, right_path, out_path)` → rebuilds the board.
   - **Verify**: the right overlap is the one that makes the hex-grid x-jog
     disappear (calibration becomes a single affine line). The detector's number
     is an estimate; the calibration is the cross-check.

2. **Calibrate the hex grid.** Read 5–8 printed hex numbers off the scan and
   note their pixel centers (≥2 distinct columns, ≥2 rows, spread across the
   board AND across any former seam; **include ≥2 even-column anchors with their
   actual down-shifted centers**). `hexgrid.fit_from_anchors(anchors, image_full)`
   → affine `(col,row)→pixel`. Save with `grid.to_json()`.
   **Each anchor's (col,row) MUST be the number PRINTED in that hex — read it, don't
   eyeball it from the grid geometry.** If you mislabel every anchor the same way (e.g.
   one row low), the fit is still perfect but the whole grid is off by that shift; step 6
   is where you catch it.

3. **Pick CONFIDENT exemplars per terrain type** — one+ obviously-correct hex of
   each (clear, forest, swamp, lake/water, …). One bad exemplar (a "sea" sample
   that's actually land) poisons every match — sanity-check each centroid's
   colour before trusting it (water must read blue: B > R,G).

4. **Classify.** `clf = ReferenceClassifier(grid).fit(arr, exemplars)`;
   `clf.classify_all(arr, hexes)`. Feature space = mean RGB (hue) + colour
   variance (solid vs textured) + **morphology** (forest = circular *bulbs*,
   swamp = *lines* — `elongation`). Reliable for is-this-water; for sorting
   non-water noise into forest/swamp it's weaker (dark town icons match forest)
   — default ambiguous noise to clear.

5. **Hexside terrain → a separate edge layer.** Full-hex classification cannot
   represent lakes-on-hexsides / rivers / escarpments. Where the real water runs
   along hex edges (half-water hexes), confine it to a region and model it as
   edges (like a rivers layer), NOT a full-hex type. Flag this; don't guess.

6. **ALWAYS verify the calibration against the PRINTED NUMBERS — not just "lands on
   a hex".** Run `hexgrid.verify_against_printed(grid, truth)` with a few hexes read
   straight off the printed numbers (ideally NOT the fit anchors), top/middle/bottom.
   It must return `[]`. Then `overlay.draw_centers` and confirm each computed **CCRR
   label equals the number PRINTED in that hex**. A *uniform* row/col offset lands on
   real hexes AND fits with ~0 least-squares residual — so "it lands on a hex" and a
   clean fit BOTH pass while the whole map is one hex off (the TWU −1-row bug: every
   hex rendered/sampled one row below its printed number, undetected for 3 sessions).
   The label-vs-printed-number check is the only thing that catches it. Then
   `overlay.draw_terrain` (classification matches the eye?). Counts lie; render and look.

## Red flags
- Classifying with hardcoded colour cutoffs → switch to exemplars.
- A terrain count that triples an existing high-confidence class → mis-matched
  exemplar or town-icons-as-forest; look at the overlay.
- Keeping 100+ "lake" hexes sprawled across the board → over-broad water; real
  lakes are confined. Re-exemplar with a true sea sample.
