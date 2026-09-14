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

## Update: GAMUS dataset experiment (pooled multi-scene calibration)

Downloaded 40 real RGB+height pairs from the GAMUS dataset (30 train, 10 val;
Washington DC aerial imagery, height = above-ground-level in meters, 1024x1024,
1:1 pixel-matched with RGB - see calibration/download_gamus_subset.py).

Trained a single linear regression across all 30 training scenes pooled
together (491,520 pixel samples), evaluated on 10 held-out validation scenes
never seen during training (163,840 pixel samples):

| Metric | Value |
|---|---|
| Correlation | 0.321 |
| RMSE | 8.62 m |
| MAE | 7.12 m |

**Finding**: this did NOT clearly outperform the best single-image SRTM
calibrations (city tight-crop reached 0.776 correlation). Root cause: pooling
raw relative-depth values across 30 *different* images reintroduces the same
scale-mismatch problem documented earlier with tile-stitching - each image's
relative depth is independently normalized by the model, so combining raw
values from different scenes into one global fit mixes incompatible scales.
A per-image fit avoids this by only ever comparing a scene to itself, but
can't benefit from GAMUS's larger, more diverse training data as a result.

**Scope note**: all 40 GAMUS samples used are Washington DC aerial imagery -
useful as a proof of concept, but not representative of the Indian terrain
types (Hyderabad-area cities/hills/forest) this project is actually targeting.

**Not pursued further given time constraints, but the clear next step**:
normalize each scene's relative depth (e.g. z-score to its own mean/std)
*before* pooling across scenes, so the model learns the true shape
relationship between depth and height rather than being confused by
scale differences between scenes. Full fine-tuning of the depth model
backbone on GAMUS (as suggested by the official problem statement update)
remains the most promising path to closing the domain gap on organic/hilly
terrain specifically, but requires GPU compute not available for this project.