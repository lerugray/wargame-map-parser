**Basis:** `added` = machine false negative; `removed` = machine false positive. I clustered by midpoint bands and contiguous hex/link runs.

## 1. Spatial error structure

### NaB: 323 corrections
Counts: river 120 added; road 151 added + 1 reclassified; bridge 29 added + 22 removed.

**River, 120/120 added:** one map-edge collar failure.
- Left edge: 44 additions at x≈34/56, y≈502–2589.
- Bottom edge: 58 additions at y≈2589–2664, x≈408–3312.
- Right edge: 17 additions at x≈3334–3378.
- Top edge: 1 addition at y≈39.
No river removals; no label/legend/sheet-seam pattern.

**Road, 152 corrections:** same edge-collar failure plus one class error.
- 145/151 road additions are on or immediately continuing from the perimeter: left x≈34/56, bottom y≈2602–2664, right x≈3334–3378, and short top/near-top fragments.
- The single reclassification is near the left-side road network at x≈100, y≈1365: value threshold/class calibration, not geometry.
- Residual road additions are isolated interior/near-interior fragments, not a second systematic cluster.

**Bridge, 51 corrections:**
- 21/29 additions are perimeter/co-located with newly added edge road/river crossings.
- 22/22 removals are bridge false positives. They are mostly scattered isolated points, with a small bottom-row group at y≈2589. They do not form a continuous bridge path.

**NaB conclusion:** dominant pattern is crop/perimeter suppression of valid mask evidence. Bridge overcalls are a separate topological-validation failure.

### TWU: 551 corrections
Counts: rivers 43 added/1 removed; rails 79 added/120 removed; border 54 added/1 removed; impassible 253 added/0 removed.

**Impassible, 253/253 added:** layer-wide under-detection.
- Long continuous outline/coast/perimeter components, not random misses.
- West vertical run alone: 40 additions at x≈20.9, y≈412–1584.
- Remaining additions trace northern/coastal and southern/eastern impassible outlines. This is a calibration/density/template failure.

**Rivers, 44 corrections:**
- 43 additions form four southern connected runs:
  - 7 edges, x≈385–489, y≈1704–1824.
  - 11 edges, x≈776–958, y≈1704–1899.
  - 13 edges, x≈1036–1114, y≈1689–2004.
  - 12 edges, x≈1218–1270, y≈1704–2004.
- The one removal is at x≈1114, y≈1674, immediately north of the added southern run: wrong-edge/parallel-layer artifact.

**Border, 55 corrections:**
- 54 additions form three southern/eastern runs:
  - Main southern border: 37 edges from x≈802 to x≈1790, y≈1674–2004.
  - Southeast pocket: 15 edges around x≈2467–2571, y≈1704–2004.
  - Far-east pair: 2 edges around x≈3091–3117, y≈1193–1208.
- The one removal is the same wrong location as the river removal: x≈1114, y≈1674.

**Rails, 199 corrections:**
- Additions: 79 missed true rail links; 54/79 are in the southern third, y≥1500, forming continuation/gap-fill through the real southern/eastern rail net.
- Removals: 120 false rails. These are not uniform grid noise and not text-underlines; they are route-shaped clusters.
- Strongest false-positive corridor: 20 removed rail links exactly on y≈1673.8, x≈1166–2415, a straight southern run coincident with added border/impassible/river-edge evidence.
- Geometry: TWU rails are center-to-center links. Border/river/impassible are hexside edges. A rail link and the crossed hexside can share the same midpoint but have perpendicular orientation. The machine accepted hexside-oriented ink as rail-oriented ink.
- Smaller removal clusters occur near true rail additions, usually as 1–5 neighboring links around the correct route: wrong-neighbor/wrong-orientation snaps, not scattered artifacts.

No convincing label-box, legend, or sheet-seam signature in either map.

## 2. Structural error classes and shares

### NaB, 323 total
1. **Map-edge/crop false negatives:** 286/323 = **88.5%**  
   Includes 120 river additions, 145 road additions, 21 bridge additions.
2. **Bridge isolated false positives:** 22/323 = **6.8%**  
   Removed bridge keys lacking road+river crossing support.
3. **Residual interior miss/class errors:** 15/323 = **4.6%**  
   Remaining bridge/road additions plus 1 road reclassification.

### TWU, 551 total
1. **Impassible calibration/density false negatives:** 253/551 = **45.9%**.
2. **Linear-continuation false negatives:** 176/551 = **31.9%**  
   Rivers 43 + border 54 + rails 79 additions.
3. **Rail/hexside layer-orientation confusion false positives:** 120/551 = **21.8%**.
4. **Non-rail wrong-edge false positives:** 2/551 = **0.4%**  
   The river+border removals at `2129-2228`.

## 3. Detector-improvement specification, ranked

Acceptance tests compare new output to these correction keys: added/reclassified keys should be present with operator value; removed keys should be absent.

### 1. NaB padded-frame perimeter extraction
- **Targets:** NaB map-edge/crop class, 286 corrections.
- **Mechanism:** Before mask tracing/snapping, pad raster beyond the crop. Do not discard candidate edges whose support polygon intersects the original crop boundary. Score partial masks in the outer 2 hex columns/rows using visible-support normalization.
- **Plug-in:** mask cleanup + edge scoring.
- **Acceptance:** output contains ≥110/120 NaB river additions, ≥135/145 NaB edge-collar road additions, and ≥18/21 NaB edge-collar bridge additions; output does not retain >4/22 NaB bridge removal keys.

### 2. TWU impassible-specific calibration
- **Targets:** TWU impassible under-detection, 253 corrections.
- **Mechanism:** Use layer-specific HSV/contrast/density thresholds for impassible, separate from rivers/borders. Permit faint/dashed continuous outline evidence; join collinear fragments before snapping.
- **Plug-in:** calibration thresholds + mask cleanup.
- **Acceptance:** output contains ≥230/253 TWU impassible addition keys.

### 3. Graph continuity gap-fill for linear features
- **Targets:** TWU river/border/rail additions, 176 corrections; residual NaB path gaps.
- **Mechanism:** After initial snapping, build per-layer graphs. For same-layer degree-1 endpoints separated by 1–2 missing lattice edges/links with consistent heading, lower the intervening-edge score threshold and fill only if mask evidence is present.
- **Plug-in:** path-continuity post-processing.
- **Acceptance:** output contains ≥40/43 TWU river additions, ≥50/54 TWU border additions, and ≥60/79 TWU rail additions.

### 4. TWU rail vs hexside orientation deconfliction
- **Targets:** TWU rail false positives, 120 removals.
- **Mechanism:** At each rail candidate midpoint, estimate local mask principal orientation. Suppress rail if ink is parallel to the crossed hexside rather than aligned with the center-to-center rail link, unless it connects two already accepted rail components with independent rail evidence.
- **Plug-in:** rail edge scoring + layer assignment.
- **Acceptance:** output excludes ≥90/120 TWU rail removal keys, including ≥19/20 removals on the exact y≈1673.8 corridor; output still contains ≥65/79 TWU rail addition keys.

### 5. Bridge topological validation
- **Targets:** NaB bridge false positives/remnant misses, 51 bridge corrections.
- **Mechanism:** Accept a bridge only if a road edge and river edge are present or newly inferred at the same/adjacent crossing and bridge-symbol evidence exists. Suppress isolated bridge candidates without both supports.
- **Plug-in:** bridge post-validation.
- **Acceptance:** output excludes ≥18/22 NaB bridge removal keys and contains ≥24/29 NaB bridge addition keys.

### 6. Road value calibration after continuity repair
- **Targets:** NaB road reclassification.
- **Mechanism:** Assign primary/secondary after path repair using local stroke width/contrast over the final road mask, not pre-repair raw density alone.
- **Plug-in:** edge classification.
- **Acceptance:** output value for `1,26|2,27` is primary; no accepted NaB road correction key with operator value secondary is emitted as primary unless its correction value is primary.