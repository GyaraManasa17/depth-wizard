# Depth Backend — Status & Known Limitations

## What this is
`backend/api.py` runs a `POST /predict-depth` endpoint: upload an image, get back
a relative depth map (PNG, grayscale). Brighter = the model thinks it's taller/closer.
This is NOT yet in real-world meters — that conversion is the calibration step.

## Sample data
`data/sample-outputs/` has 5 terrain types, each as an `{name}_original` +
`{name}_depth` pair: city, hills, sparse, forest, mixed.
Use these to start building/testing against real output.

## Known limitation: tile seam artifacts
Large images are split into 512x512 overlapping tiles, run through the depth
model separately, then stitched back together (see `backend/tile_and_stitch.py`).
Each tile's depth values are aligned to its neighbor's brightness/contrast at
the overlap, which fixes most of the mismatch — but it isn't perfect.

**You may still see faint vertical/horizontal seams**, especially on flatter,
low-texture terrain (open ground, forest canopy) where the model has fewer
strong edges to anchor onto. This is most visible in `mixed_depth.png` and
`sparse_depth.png` — look for a soft brightness shift roughly every ~450px.

**Why this matters for calibration/scoring:** don't treat small brightness
jumps at those regular intervals as real elevation change — it's very likely
tile artifact, not real signal. If your correction step ends up smoothing this
out naturally, great — but it's worth being aware of as a source of error
when comparing against SRTM/reference data, especially for non-urban terrain.

## Model in use
Depth Anything V2 - Small checkpoint (`depth-anything/Depth-Anything-V2-Small-hf`
via HuggingFace transformers). Chosen for CPU-friendliness during dev — may
switch to Base or DA3 later if accuracy needs it and GPU access becomes available.