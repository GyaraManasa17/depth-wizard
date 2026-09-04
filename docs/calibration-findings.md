# Calibration (SRTM) — Findings So Far

## Method
Linear fit: `real_meters = a * relative_depth + b`, fit on 80% of pixels,
evaluated (RMSE/MAE/correlation) on the held-out 20% never seen during fitting.
Tested block-averaging depth values to ~30m (SRTM's native resolution) before
fitting, to rule out pixel-resolution mismatch as the cause of weak results.

## Results

| Terrain | Correlation | Test RMSE | Test MAE |
|---|---|---|---|
| City (Hitec City) | -0.244 | 9.40 m | 7.69 m |
| Hills (Bhongir rock, per-pixel) | 0.184 | 44.30 m | 34.22 m |
| Hills (Bhongir rock, block-averaged to ~30m) | 0.195 | 44.94 m | 34.46 m |

## Key finding
**Correlation is weak across every terrain type tested so far** — none of
these are close to a usable linear relationship yet. This isn't primarily a
resolution-mismatch problem: block-averaging depth values to match SRTM's
~30m resolution barely changed the correlation for hills (0.184 -> 0.195),
which rules that out as the main cause.

The likely real cause: this matches an earlier qualitative finding from the
depth model itself — on organic/rocky terrain with no hard edges (no building
corners, no sharp shadows), Depth Anything V2's output was a smooth, largely
featureless gradient with little real structure (see `docs/depth-backend-notes.md`).
A weak/uninformative relative depth signal can't be calibrated into a strong
absolute one, regardless of how good the calibration math is.

## Update: Depth Anything V2 Small vs. Base

Tested whether a larger depth model improves calibration correlation, using
the same train/test methodology as above.

| Terrain | Model | Correlation | Test RMSE | Test MAE |
|---|---|---|---|---|
| City | Small | -0.244 | 9.40 m | 7.69 m |
| City | Base | **0.375** | 8.99 m | 7.45 m |
| Hills | Small | 0.184 | 44.30 m | 34.22 m |
| Hills | Base | 0.194 | 44.17 m | 34.25 m |

**Finding**: Base meaningfully improved city correlation (weak/negative ->
moderate positive) but barely moved hills. This supports the earlier
hypothesis: the depth model's structure on organic/rocky terrain is the
bottleneck, not model size within the same family - a bigger model helps
where there's real geometric structure (building edges) to find, but doesn't
help much where there isn't.

**Decision**: switched the whole pipeline (API, sample generation) to use
Base by default, given the meaningful city accuracy gain, accepting the
~4x slower inference cost. Hills/organic terrain remains a known weak point
requiring a different approach later (more reference points, or
terrain-specific calibration), not just a bigger model.

## What this suggests for next steps (not yet tried)
- A larger depth model (Base or DA3) may capture more real structure on
  organic terrain — worth testing if compute allows
- Per-region/local calibration instead of one global linear fit per image
  may help, since a single global line may not capture local nonlinearities
- City terrain's negative correlation is a separate issue: likely SRTM
  genuinely cannot resolve individual building heights at 30m resolution,
  so building-detail depth signal has no real counterpart in SRTM at all

## Update: Full terrain-type sweep + bounding box precision

Tested all 5 required terrain types (city, hills, forest, sparse, mixed) with
real GeoTIFFs and SRTM ground truth, using Depth Anything V2 Base. Also
tested a second hill location and a tighter city crop to isolate whether
weak results were about terrain type or about crop quality.

| Terrain | Correlation | RMSE | MAE |
|---|---|---|---|
| City (tight box) | **0.776** | 5.83 m | 4.63 m |
| Forest (KBR National Park) | 0.66 | 6.04 m | 4.75 m |
| Sparse (Shamshabad) | 0.601 | 3.92 m | 3.14 m |
| Mixed (Kompally) | 0.453 | 5.86 m | 4.75 m |
| City (original, looser box) | 0.375 | 8.99 m | 7.45 m |
| Hills - Bhongir (bare rock) | 0.194 | 44.17 m | 34.25 m |
| Hills - Ananthagiri (vegetated) | **-0.293** | 29.34 m | 23.87 m |

## Two findings

**1. Bounding box precision matters more than expected.** Re-cropping the
same city location more tightly took correlation from 0.375 to 0.776 - more
than double. This means the sparse/forest/mixed numbers above likely have
real headroom left if their crops were tightened further; box quality is a
real, controllable source of noise in these results, separate from terrain
type itself.

**2. Hilly/undulating terrain is a genuine, repeated weak point, not a
one-location fluke.** Two different hill locations - Bhongir (bare exposed
rock) and Ananthagiri (more vegetated, rolling terrain) - both produced weak
or negative correlation, despite looking visually quite different from each
other. This rules out "picked an unlucky spot" and points to something more
fundamental about how this depth model + linear SRTM calibration handles
gradual, undulating elevation change, regardless of surface texture/vegetation.

## Ranking so far (best to worst structure for this method)
City (tight crop) > Forest > Sparse > Mixed > Hills

## Next steps for calibration (not pursued further for now - shifting focus
## to frontend/visualization to balance the 50/50 grading split)
- Tighter, more careful bounding boxes could likely improve sparse/forest/mixed further
- Hills remains the core unsolved problem - would need either more reference
  points per region, a fundamentally different method for gradual terrain,
  or acceptance as a documented limitation for the final submission