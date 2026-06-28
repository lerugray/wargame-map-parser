# Case study: *The World Undone — East Prussia 1914*

The map this tool was built on. (Board scans aren't included — they're the
publisher's art — but here's the full worked process and the numbers.)

## 1. Duplicated-column seam

The digital board was built by concatenating two sheet scans edge-to-edge:

```
left sheet  3338w  +  right sheet 3339w  =  6677w
```

`3338 + 3339 = 6677` exactly — a naive join with no overlap removed. The two
sheets **share column 31** (hexes 31xx — the Heilsberg/Bischofsburg band): it
was printed on the left sheet's right margin *and* the right sheet's left
margin, so the board showed it **twice, side by side**. Symptoms:

- the same hex numbers (3112–3124) repeated near the center;
- the hex-grid x-calibration needed a **two-segment model** with a `+158.62px`
  "jog at column 31" — which is exactly the duplicate band width:
  `right_intercept 228.54 − left_intercept 69.92 = 158.62`.

Fix: drop the right sheet's ~159px duplicate band → board `6518×5139`, and the
x-model collapses to one line `x = 69.73 + col·104.08`. (`seams.detect_overlap`
estimates the band; the calibration confirms the exact value.)

## 2. Calibration

Flat-top, even-q offset (even columns shifted down half a row):

```
x = 69.73 + col·104.08
y = 522.3 + row·120.19 + (60.13 if col even else 0)
```

`fit_from_anchors` recovers this from a handful of read-off hex centers. Verify
by rendering `draw_centers` across the former seam — the dots must land on the
printed hexes with no half-column drift.

## 3. Terrain: lakes vs swamp (where reference-hex earns its keep)

First pass used absolute colour thresholds and **failed**: it found *zero* real
lakes among 92 "lake" hexes and smeared impassable water across the whole board
(160 hexes spread over every column) — when the real Masurian lakes are
confined to the southeast.

The reference-hex method, with a **correct blue sea exemplar** (RGB ≈
162,170,184, B > R,G), separated them cleanly:

- **Swamp** (89 hexes): stippled blue-grey marsh — passable, +2 DRM. Distinct
  from solid lake by *colour variance* (stipple = high variance).
- **Lakes** → confined to **13 hexes** in the Masurian district. 70 "lake"
  false-positives (grey town icons, clear hexes — including VP towns
  Pr.Holland / Mohrungen / Eylau / Hohenstein that were wrongly *impassable*)
  reclassified out.
- **Hexside lakes**: the genuine Masurian water runs along hex *edges* near
  Lötzen — kept as a confined region for a future edge layer, not forced into a
  full-hex type.

The morphology feature (forest = circular bulbs, swamp = lines) is what
disambiguates forest from swamp where colour alone can't — and why dark town
icons must not be auto-sorted as forest.

## Lessons (baked into the tool)

- A bad exemplar poisons everything — the first "sea" sample was tan land, and
  every match went wrong until it was fixed. Sanity-check centroids.
- Reference-matching is sharp for *is-this-water*; weaker for forest-vs-swamp on
  a cream palette. Default ambiguous noise to clear; verify with the overlay.
- Counts lie. Every step above was confirmed by *looking* at a rendered overlay.
