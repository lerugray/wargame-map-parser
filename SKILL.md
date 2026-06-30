---
name: parse-wargame-map
description: Extract per-hex terrain from a scanned wargame map (flat-top hex board with printed CCRR coordinates) into a hex→terrain table. Use when digitizing a board wargame's map — de-duplicating multi-sheet scans, calibrating the hex grid, and classifying terrain by reference-hex matching. Backed by the wargame-map-parser repo (numpy + Pillow).
---

# Parsing a wargame map → per-hex terrain

Reference-hex method: classify each hex by **nearest labeled exemplar**, not
absolute colour thresholds (which break per map). Pipeline lives in `parser/`.

**Method note (GotA 2026-06-30):** pure nearest-exemplar on mean color alone
proved insufficient on GotA's ~4,000-hex continental map. The current best
practice is a **hybrid color + morphology** approach with supervised
operator-confirmed exemplars. Full detail in `docs/CONVENTIONS.md`.

## Procedure

0. **Identify the THREE layers before touching any code.**
   Every hex wargame map has three graphically distinct layers — classify them
   separately, not all at once from the fill interior:
   - **Hex fill** — terrain printed INSIDE the hex body (clear, forest, desert,
     swamp, water). Sampled from the hex center region.
   - **Hexside edge** — features on hex EDGES (rivers, rail, mountain ridges,
     coastline breaks). A fill sampler will MISS these. GotA mountains: hexside,
     not fill — a fill classifier found 2 in 1,539 hexes.
   - **Point features** — city circles, VP/BP numbers, port anchors, capitol
     stars. Extract separately; never fold into a fill class.

   Check the reference map physically: if a symbol lives on the line *between*
   two hexes, it is a hexside feature.

1. **De-duplicate the board if it's multi-sheet.** Boxed maps print across
   sheets sharing an overlap strip; edge-to-edge scans duplicate it (same hex
   numbers repeat near the join).
   - `seams.detect_overlap(left, right)` → estimated duplicate-band width.
   - `seams.fix_sheets(left_path, right_path, out_path)` → rebuilds the board.
   - **Verify**: the right overlap is the one that makes the hex-grid x-jog
     disappear (calibration becomes a single affine line). The detector's number
     is an estimate; the calibration is the cross-check.

2. **Calibrate the hex grid.**
   - **Anchors:** operator reads 3+ printed hex CCRRs directly off the scan and
     clicks their pixel centers. Span NW, NE, SE of the board (don't cluster).
     Include ≥2 even-column anchors with their actual down-shifted centers.
     **Each anchor's (col,row) MUST be the number PRINTED in that hex** — read
     it; don't eyeball from geometry.
   - `hexgrid.fit_from_anchors(anchors, image_full)` → calibrated HexGrid.
   - **Geometry-ratio check (NEW):** for flat-top hexes `row_pitch / col_pitch`
     must equal `2 / √3 ≈ 1.1547`. Call `hexgrid.check_geometry_ratio(grid)`.
     A ratio outside ≈1.15–1.16 indicates a bad fit (GotA: bad fit = 1.23,
     correct fit = 1.154). Fix by re-anchoring from NW/NE/SE.
   - **Pitch cross-check:** FFT/autocorrelation of horizontal image slices
     confirms pitch independently. Flat-top diagonals produce a strong
     half-pitch harmonic — fundamental = 2 × that peak.
   - Save with `grid.to_json()`.

3. **Pick OPERATOR-CONFIRMED exemplars per terrain type** — not eyeball guesses
   from a rescaled overview.
   - For hard terrains (symbol-based, or same-palette-different-shade): the
     operator **screenshots the actual hex** from the scan viewer, reads its
     CCRR, and you extract the pixel patch at those exact coordinates. That is
     the supervised exemplar.
   - One bad exemplar (a "sea" sample that's actually land) poisons every match.
     Sanity-check each: water centroid must be `B > R,G`.
   - **Gotchas for sampling:**
     - **Mask the hex number.** The CCRR label is dark ink in the center zone;
       it biases color mean and fakes a "mark" in morphology. Exclude ~10%
       center radius or blank the number zone before sampling.
     - **Center-only.** Use `r ≤ half hex radius` — excludes edge rivers,
       coastlines, and neighboring hexside bleed.
     - **Strict blue-hue gate for water:** `B > R + margin AND B > G + margin`
       explicitly, not nearest-centroid. Nearest-centroid over-grabbed ~480
       non-water hexes on GotA.

4. **Classify using HYBRID color + morphology, layered.**
   Current `ReferenceClassifier` uses mean RGB + gray_std + morphology features.
   The correct application order for best accuracy:
   1. **Strict water gate first** (definitive blue → water; skip remaining).
   2. **Morphology override** — elongated marks → swamp; circular compact blobs
      → forest. Symbol terrains override base color.
   3. **Nearest-centroid fallback** for unresolved hexes (no strong morphology
      signal).

   Feature space = mean RGB (hue) + colour variance (solid vs textured) +
   **morphology** (forest = circular *bulbs*, swamp = *lines* — `elongation`).

   See `docs/CONVENTIONS.md §3` for the full hybrid approach and a roadmap
   for a `HybridClassifier` class (color gate + morphology overlay).

5. **Hexside terrain → a SEPARATE edge layer.**
   Full-hex classification cannot represent rivers / rail / mountain ridges /
   escarpments. Where a feature runs along hex edges (half-feature hexes),
   confine it to a region and model it as edges — NOT a full-hex type.
   GotA mountains are ENTIRELY hexside; zero fill pixels contain them.
   Flag this; don't guess.

6. **ALWAYS verify the calibration against the PRINTED NUMBERS — not just
   "lands on a hex".**
   - `hexgrid.verify_against_printed(grid, truth)` with ≥2 hexes NOT used as
     fit anchors, top/middle/bottom. Must return `[]`.
   - `overlay.draw_centers` — confirm each computed CCRR label equals the
     number PRINTED in that hex.
   - A *uniform* row/col offset lands on real hexes AND fits with ≈0 residual —
     so "it lands on a hex" and a clean fit BOTH pass while the whole map is
     one hex off (the TWU −1-row bug: every hex one row below its printed
     number, undetected 3 sessions).
   - **Origin-drift diagnosis:** if far corners drift but one anchor looks
     "dead-on," suspect the origin, not the pitch. Re-detect that anchor's
     pixel center objectively — the eyeball lied.

7. **Self-validate classification before reporting results.**
   - Reproduce operator-confirmed exemplars: ≥10/12 (or comparable threshold)
     must be classified correctly.
   - Geographic sanity: swamp on coasts/lowlands; desert in arid zones; rough
     near mountains; forest in wooded regions.
   - `overlay.draw_terrain` — counts lie; **the operator must review the overlay
     on their own screen**. Your render never certifies a result.

## Operator gate — the non-negotiable

The operator's visual review is the GATE for every classification result.
Never call a digitization done from a count or your own overlay. If the
operator says something looks wrong, it IS wrong. Restart from the mis-labeled
hexes. "I can't reproduce it" is not a valid response.

Machine vision (including LLM vision APIs) is NOT ground truth for number
reading or pixel alignment. Use `verify_against_printed()` + direct numpy
sampling.

## Orchestration note

Map rasters are large (GotA ≈90 MP). Never load the raster into the
orchestrator context. Run FFT, full-board classify, and overlay render in
background subagents that return short structured results (counts, mismatch
lists, sample patches), not raw arrays.

## Red flags
- Classifying with hardcoded colour cutoffs → switch to exemplars.
- A terrain count that triples an existing high-confidence class → mis-matched
  exemplar or town-icons-as-forest; look at the overlay.
- Keeping 100+ "lake" hexes sprawled across the board → over-broad water; re-
  exemplar with a strict-blue sample and add the explicit hue gate.
- Mountain/ridge count near zero on a map with obvious peaks → mountains are a
  hexside feature, not fill; switch to the edge layer.
- `check_geometry_ratio()` returns `ok: False` → bad fit; re-anchor from NW/NE/SE.
- `verify_against_printed()` returns non-empty → systematic offset; fix before
  classifying anything.
