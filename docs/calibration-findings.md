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

## What this suggests for next steps (not yet tried)
- A larger depth model (Base or DA3) may capture more real structure on
  organic terrain — worth testing if compute allows
- Per-region/local calibration instead of one global linear fit per image
  may help, since a single global line may not capture local nonlinearities
- City terrain's negative correlation is a separate issue: likely SRTM
  genuinely cannot resolve individual building heights at 30m resolution,
  so building-detail depth signal has no real counterpart in SRTM at all