# Wargame Map Parser — Conventions and Best Practices

Lessons distilled from two real digitizations:

- **TWU East Prussia** (*The World Undone*): two-sheet scanned hex board, ~800 hexes.
  The core tool was built from this map; it established the reference-hex nearest-exemplar
  approach and the seam-fix pipeline.
- **GotA** (*Guns of the Americas: The American Front 1914–1919*, 2026-06-30): continental
  hex map, ~4,000 hexes, multiple terrain types, mountains as hexside features.
  This digitization stress-tested every assumption from TWU and produced the refinements below.

These rules refine and extend the original single-method nearest-exemplar approach documented
in the README. Where they conflict with the README, the rules here are newer.

---

## 1. A map has THREE layers — digitize them separately

Every hex wargame map carries three graphically distinct layers. Trying to extract all
information from full-hex interior sampling alone is the single most common failure mode.

| Layer | What it captures | How to extract |
|---|---|---|
| **Hex fill** (interior) | Terrain printed INSIDE the hex body — clear, forest, desert, rough, swamp, water | Sample hex interior (center region only, `r ≤ ~half hex`); classify by color + morphology |
| **Hexside edge** | Features drawn ON hex EDGES — rivers, rail lines, **mountain ridges**, coastline transitions | Sample along the hex edge lines; requires a separate edge-layer model |
| **Point feature** | Symbols AT or NEAR the hex center — city circles, capitol stars, port anchors, VP/BP numbers | Detect by centroid template or intensity threshold; never fold into a fill class |

### The critical mistake this prevents

GotA mountains are a hexside feature — the mountain-ridge symbol is drawn on the hex *edge*,
not the interior fill. A fill classifier searching for mountain terrain found 2 in 1,539 hexes
because it was looking in the wrong layer.

**Rule:** before sampling anything, check the reference map physically. If a terrain symbol
lives on the line *between* two hexes, it is a hexside feature. Extract it from an edge layer,
not a fill sampler.

### Implication for point features

Mask the printed hex number (the CCRR label printed inside the hex) before sampling fill.
A visible number biases the color mean and reads as a spurious dark symbol in the morphology
pass. Exclude the ~10% center-radius zone around the number (or track the number-zone
centroid and blank it before sampling).

---

## 2. Grid pinning — fitting `fit_from_anchors` reliably

### Anchor selection

Pin from 3+ operator-read anchors spanning the full board: ideally NW corner, NE corner,
and SE corner. **The operator's eyes reading a clear printed number are ground truth** for the
anchor CCRR; tiny-number OCR on scans is unreliable and should not be trusted for anchors.

Minimum requirements (unchanged from the README):
- ≥2 distinct columns; ≥2 distinct rows
- Spread across the board AND across any seam
- Include ≥2 even-column anchors with their actual down-shifted pixel centers

### Pitch confirmation via FFT / autocorrelation

The hex column and row pitch can be confirmed objectively by autocorrelation of horizontal and
vertical intensity slices across the board. For flat-top hexes, the diagonal edges produce a
strong **half-pitch harmonic** — the autocorrelation often peaks at `pitch/2` (the half-pitch),
not at the full pitch. The fundamental column pitch is `2 × half_pitch`.

This provides an anchor-independent pitch estimate to cross-check the `fit_from_anchors`
result.

### Geometry-ratio sanity check ← **NEW**

For a geometrically correct flat-top hex grid:

```
row_pitch / col_pitch = 2 / √3 ≈ 1.1547
```

After fitting, call `hexgrid.check_geometry_ratio(grid)` to validate. A ratio deviating from
1.1547 by more than ~0.03 is a strong indicator of a bad fit — wrong anchors, an unflattened
scan, or origin-drift (see below).

**GotA example:** an initial fit spanning only part of the map gave ratio ≈ 1.23 — clearly
wrong. Refitting from three operator-read NW/NE/SE anchors gave ratio ≈ 1.1540 (within
tolerance). The geometry ratio is a fast sanity check that catches wrong fits before you
invest time classifying.

### Validate at un-fitted hexes

After fitting, **predict 2+ hexes you did NOT use as anchors** and confirm that the predicted
CCRR matches what is printed in those hexes. Use `hexgrid.verify_against_printed()`.

A "the grid lands on hexes" visual check is not sufficient. A uniform CCRR-label offset (e.g.,
every anchor read one row too low) produces a PERFECT least-squares fit and lands on real hex
centers — yet the whole map is shifted one row off the printed numbering. Un-fitted spot-checks
are the only thing that catches this (the TWU −1-row bug went undetected for three sessions
until this check was applied).

### Origin-drift diagnosis

If far corners drift but one anchor appears "dead-on," suspect the **origin parameters**
(`x_intercept_col0`, `y_intercept_row0`), not the pitch. The eyeball-read anchor lied —
re-detect that anchor's pixel center objectively (find the darkest number-cluster pixels in a
small search window) rather than trusting a click.

---

## 3. Classification is HYBRID — color AND morphology, layered

This is the central lesson from GotA. Pure nearest-exemplar on mean color alone (the original
approach) proved insufficient when:

- Multiple terrains share the same base color (GotA clear / desert / rough = three tan shades
  distinguished only by how dark they print).
- A terrain is visually distinguished by a **printed symbol**, not by color (swamp = short
  horizontal broken dashes on a cream background; the cream base is identical to clear).

### Two-tier approach

**Tier 1 — COLOR gate:** classify terrains that are visually distinct by hue. Water (distinct
blue) is the primary color-only terrain.

- Water requires a **strict blue-hue gate**: check `B > R + margin` AND `B > G + margin`
  explicitly. RGB-nearest-centroid drags desaturated tan land into the water class and
  over-grabs hundreds of non-water hexes (observed: ~480 spurious "water" hexes on GotA with
  a relaxed gate). The strict gate eliminates this.

**Tier 2 — MORPHOLOGY override:** classify terrains distinguished by printed symbol (swamp,
forest, and any symbol-based terrain). Symbol terrains OVERRIDE base color:
- cream fill + detected swamp-dash morphology → **swamp**
- cream fill + detected forest-bulb morphology → **forest**
- cream fill alone, no strong morphology → **clear** (default)

Layer order:
1. Apply the strict water gate (definitive blue → water; skip remaining tiers).
2. Run morphology detection on the interior patch (elongated marks → swamp candidate;
   circular compact blobs → forest candidate).
3. If morphology score is below threshold or ambiguous → fall back to nearest-centroid color.

### Supervised exemplars — operator-confirmed, not guessed

For hard terrains (symbol-based, shade-only distinguishable), get the operator to:
1. Screenshot the actual hex directly from the scan viewer.
2. Read the CCRR from the printed number.
3. Extract the real pixel patch from those coordinates.

These are "operator-confirmed exemplars." Use them as the fit inputs instead of eyeball-picked
CCRRs from a rescaled overview image.

**GotA swamp result:** pure nearest-centroid classified swamp entirely wrong (near-zero hit
rate). Once driven by operator-screenshot exemplars, accuracy reached 10/12 on the confirmed
test set.

### Critical gotchas for fill sampling

1. **Mask the hex number.** The CCRR label printed inside the hex is dark ink in the
   center-ish region. It biases mean color and reads as a spurious dark mark in the morphology
   pass. Either exclude the ~10% center-radius zone or track and blank the number centroid.

2. **Center-only sampling.** Use `r ≤ ~half hex radius`. This excludes border effects: edge
   rivers, coastline color transitions, and neighboring hexside features that bleed across the
   edge. The `hex_features()` function uses `0.42 * hex_size` by default; this is appropriate
   for most maps.

3. **Strict blue-hue gate for water.** Spell it out explicitly rather than relying on nearest-
   centroid. A condition like `B > R + 15 and B > G + 5` (tunable) is far more reliable than
   Euclidean distance to a "water" centroid.

### Self-validate before promoting

Before locking a classifier pass:
- **Reproduce operator exemplars:** ≥10/12 (or similar) confirmed hexes must be classified
  correctly.
- **Geographic sanity:** swamp on coastal lowlands or flood plains; desert in arid zones;
  rough near mountains; forest in wooded zones. A "correct" count that maps swamp to the
  Sonoran Desert is wrong regardless of what the numbers say.

---

## 4. Process discipline

### The operator's eyes are the gate

Never call a digitization "done" based on a count or an overlay rendered by the same process
that produced the classification. The operator's visual review is the gate.

If the operator says something looks wrong on their screen, it IS wrong. Restart from the
mis-labeled hexes, re-exemplar, re-validate. "I can't reproduce it on my render" is not a
valid response (see rule: verify-reported-visual-defects-objectively).

### Vision is not ground truth for numbers or pixels

Machine vision — including LLM vision APIs — is unreliable for:
- Reading tiny printed CCRR numbers → use operator-read anchors.
- Confirming a pixel value → use direct numpy sampling.
- Confirming a hex is "aligned" or "correctly labeled" → use `verify_against_printed()` +
  numeric comparison.

An LLM vision pass that returns "looks aligned" on a broken render is not a check. Pixel
arithmetic is a check.

### Orchestrate via background subagents

A high-res wargame map raster is large (GotA was ~90 MP, TWU was ~33 MP). Never load the
full raster into the orchestrator context window. All heavy operations (FFT pitch detection,
full-board classify, overlay render) should run in background subagents that report back
**short structured results** — counts, mismatch lists, sample patches — not raw arrays.

### Keep the operator's review folder organized

Maintain a clean output structure:
- **Top level:** the latest overlay PNG, the latest terrain CSV/JSON, the hexgrid JSON.
- **`_archive/` or `_debug/`:** intermediate renders, failed experiments, dated debug outputs.

The operator should open the folder and immediately see the current result without hunting
through version-suffixed files.

---

## 5. Hexside-snap — HMM/Viterbi map-matching for hand-traced linear features

**When to use it:** you have a hand-traced linear feature over a scanned hex map —
rivers, ridges/mountains, impassible-terrain boundaries, rail lines, coastline
breaks — and need to assign it onto the hex-lattice **hexside** graph (which two
adjacent hexes share the edge the feature runs along). This is the hexside-edge
layer named in §1 above; `parser.hexside_snap` is the tool that fills it in from a
hand trace instead of requiring the operator to paint every hexside by hand.

**Method:** `parser.hexside_snap.HexsideSnapper` (spec by Fugu, fugu-ultra
spec-only mode, commissioned by Ray Weiss; validated on *Guns of the Americas*
2026-07-02 — 814 river + 64 impassible hexsides accepted, confirmed against the
rendered overlay). Treats the hand-traced skeleton as a noisy GPS trace and the
hex-lattice hexside graph as the road network, and decodes the most likely
CONNECTED walk through the graph via per-link Viterbi — an emission cost that
rewards low perpendicular distance to a hexside's supporting line AND parallel
tangent (not just proximity), a transition cost that only allows moving between
graph-adjacent hexsides, and a post-decode along-vs-crossing support rule that
throws out edges the trace only grazed or crossed.

### The 10px-proximity metric is BANNED for this method

Distance-threshold / fixed-pixel-buffer methods (accept a hexside if the trace
passes within Npx of it) hit a hard **~46% coverage ceiling** on a meandering
hand trace, because "nearest within Npx" throws away exactly the information
that resolves ambiguity: which direction the trace is *headed*. Do not use a
10px (or any fixed-pixel) proximity gate as a success criterion, a ranking
metric, or an acceptance test for hexside-snap output. It is known to reject
correct meandering assignments and was the reason the original coverage ceiling
existed.

### Acceptance is coverage / connectivity / Fréchet distance, plus the operator's eyes

Objective metrics rank candidates and flag review priority — they are not
pass/fail gates:
- **Coverage proxy** — the fraction of resampled skeleton arc assigned to an
  edge that survived the along-vs-crossing acceptance rule.
- **Discrete Fréchet distance** (normalized by `H`) between the hand-traced
  skeleton and the decoded lattice-vertex walk — median/p90/max, per
  `HexsideSnapper.snap_layer`'s `diagnostics["frechet_dF_over_H"]`.
- **Connectivity** — connected-component count, degree histogram, duplicate/
  broken chains.

**The operator's visual review of the `--overlay` render is the only pass/fail
gate** (same discipline as §4 above): magenta = hand trace, green = high-
confidence accepted hexside, amber = accepted-but-lower-confidence, red =
suppressed crossing/ambiguous candidate. Never call a hexside-snap run "done"
from a count or your own render — the operator confirms on their own screen.

### Constants are pinned, not guessed

Every geometric parameter is expressed as a multiple of `H` (the hexside
length, `HexGrid.hex_size()`). `SnapParams`'s defaults are the operator-
validated GotA values — don't change one without re-validating against a known-
good overlay. A real Viterbi bug is preserved as a documented fix, not an
implementation footnote: the DP's "cold restart" option must only compete when
no real transition from any previous state is valid, never against the
*accumulated* cost of a real multi-step path (a restart's cost is always just
one emission term, so unconditional competition degenerates the whole decode to
1-sample paths — this happened during dev on a 53-sample test link). See the
module docstring in `parser/hexside_snap.py` and
`tests/test_hexside_snap.py::test_long_chain_does_not_degenerate_to_cold_restart`.

### Irregular grids

The candidate-graph neighbor detection uses a single Euclidean distance band
(`SnapParams.nbr_lo`/`nbr_hi`, default `[1.35H, 1.85H]`) rather than a parity-
aware even-q neighbor table — correct because on a REGULAR flat-top hex lattice
all six neighbor directions are equidistant (`sqrt(3)*H`). A grid whose
`row_pitch` deviates from the ideal `2/sqrt(3) * col_pitch` ratio (see
`hexgrid.check_geometry_ratio`) breaks that equidistance proportionally to the
deviation; widen `nbr_lo`/`nbr_hi` to compensate for a mildly irregular grid, or
re-anchor the grid fit first if the deviation is large.

### Quick start

```python
from parser import HexGrid
from parser.hexside_snap import HexsideSnapper, snap_traces

grid = HexGrid.from_json("hexgrid.json")
valid_hexes = [...]  # every eligible ("land") hex code, e.g. terrain.json's keys

hexwright_json, results = snap_traces(
    grid, valid_hexes,
    layers={"rivers": "traces/rivers-trace.png", "impassible": "traces/impassible-trace.png"},
    board_img="board.jpg", overlay_dir="overlays/", overlay_scale=0.5,
)
# hexwright_json == {"rivers": [{"a":"CCRR","b":"CCRR"}, ...], "impassible": [...]}
# -- import into Hexwright, or write straight to data/hexsides.json after review.
```

Or via the CLI: `python -m parser.hexside_snap --grid hexgrid.json --terrain
terrain.json --trace rivers=rivers-trace.png --out hexsides-snap.json --board
board.jpg --overlay overlays/`.

---

## Checklist — before declaring a digitization complete

- [ ] Three layers accounted for: fill classified, hexside features modeled as an edge layer,
      point features extracted separately.
- [ ] Grid geometry ratio checked: `row_pitch / col_pitch ≈ 1.1547 ± 0.03`
      (`hexgrid.check_geometry_ratio(grid)` returns `ok: True`).
- [ ] Calibration validated at un-fitted hexes: `verify_against_printed()` returns `[]`.
- [ ] Hex numbers masked (or excluded from sampling radius) for fill classification.
- [ ] Center-only sampling enforced (`r ≤ half hex`).
- [ ] Water gate is strict-blue (not nearest-centroid-color).
- [ ] Operator-confirmed exemplars for every hard terrain type (≥10/12 accuracy on confirmed
      set).
- [ ] Geographic sanity check passed (terrain distribution makes physical sense for the map).
- [ ] Operator has reviewed the final overlay on their own screen and confirmed it correct.
- [ ] If hexside features (rivers/ridges/impassible) were traced: hexside-snap's
      `--overlay` render reviewed by the operator, NOT accepted from a coverage
      count (§5).

### Hexside-snap: pinned behaviors (do not "fix" without re-validation)

Two real quirks found by the 2026-07-02 fugu review are deliberately KEPT,
because they were present in the GotA-validated run and changing them changes
which edges get accepted/suppressed (the operator-approved decode contract):

1. `prune_short_spurs`: endpoint spurs always carry a real `to_cluster`
   (never -1), so the length-based spur pruning path never fires.
2. `snap_layer` passes `step_len=RESAMPLE_STEP` rather than each segment's
   actual `delta_s` into the Lparallel/Lcross accumulation.

If either is ever corrected, treat it as a NEW algorithm version: re-run the
GotA layers, re-render overlays, and get operator eyes on the diff before
adopting. Diagnostics-only and CLI-surface fixes from the same review WERE
applied (Fréchet empty-polyline → inf, overlay viewport y-test, duplicate
--trace now raises).
