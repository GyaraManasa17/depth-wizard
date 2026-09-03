import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
import rasterio
from rasterio.transform import xy
import srtm
from transformers import pipeline
from PIL import Image

# --- Config ---
import sys
if len(sys.argv) != 2:
    print("Usage: python calibrate.py <label>")
    print("Example: python calibrate.py hills   (expects ../data/test_hills_geo.tiff)")
    sys.exit(1)

LABEL = sys.argv[1]
INPUT_TIF = f"../data/test_{LABEL}_geo.tiff"
OUTPUT_TIF = f"../data/sample-outputs/{LABEL}_absolute_dsm.tif"

# --- Step 1: Load the georeferenced image ---
print("Loading GeoTIFF...")
with rasterio.open(INPUT_TIF) as src:
    img_array = src.read([1, 2, 3])  # RGB bands
    transform = src.transform
    crs = src.crs
    width = src.width
    height = src.height

# rasterio gives (bands, height, width) -> PIL wants (height, width, bands)
img_rgb = np.transpose(img_array, (1, 2, 0)).astype(np.uint8)
image = Image.fromarray(img_rgb)
print(f"Image size: {width}x{height}")

# --- Step 2: Run the depth model to get RELATIVE depth ---
print("Loading depth model...")
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
print("Running inference...")
result = pipe(image)
relative_depth = np.array(result["depth"]).astype(np.float64)

# --- Step 3: Get REAL elevation (SRTM) for every pixel ---
print("Fetching SRTM elevation data (first run downloads a tile, be patient)...")
elevation_data = srtm.get_data()

srtm_grid = np.zeros((height, width), dtype=np.float64)
for row in range(height):
    for col in range(width):
        lon, lat = xy(transform, row, col)
        elev = elevation_data.get_elevation(lat, lon)
        srtm_grid[row, col] = elev if elev is not None else np.nan
    if row % 50 == 0:
        print(f"  SRTM lookup row {row}/{height}")

# --- Step 3b: Group pixels into ~30m blocks (matching SRTM's real resolution),
# average both relative_depth and srtm within each block. This removes noise
# from comparing individual ~10m pixels against a coarser ~30m ground truth. ---
# Estimate how many image pixels correspond to ~30m on the ground
with rasterio.open(INPUT_TIF) as src:
    pixel_size_deg = src.transform[0]  # degrees per pixel (roughly)
meters_per_pixel = pixel_size_deg * 111320  # rough deg-to-meters at the equator-ish
block_size = max(1, round(30 / meters_per_pixel))
print(f"Approx. {meters_per_pixel:.1f} m/pixel -> grouping into {block_size}x{block_size} blocks")

def block_average(arr, block_size):
    h, w = arr.shape
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    trimmed = arr[:h_trim, :w_trim]
    reshaped = trimmed.reshape(h_trim // block_size, block_size, w_trim // block_size, block_size)
    return np.nanmean(reshaped, axis=(1, 3))

relative_depth_blocked = block_average(relative_depth, block_size)
srtm_blocked = block_average(srtm_grid, block_size)
print(f"Blocked grid size: {relative_depth_blocked.shape}")

# --- Step 4: Split into TRAIN (fit the line) and TEST (honestly check it) sets ---
# NOTE: now using block-averaged values, not raw per-pixel values
valid_mask = ~np.isnan(srtm_blocked)
depth_valid = relative_depth_blocked[valid_mask].ravel()
srtm_valid = srtm_blocked[valid_mask].ravel()

np.random.seed(42)  # reproducible split
n = len(depth_valid)
indices = np.random.permutation(n)
split = int(n * 0.8)
train_idx, test_idx = indices[:split], indices[split:]

# Fit ONLY on training points
a, b = np.polyfit(depth_valid[train_idx], srtm_valid[train_idx], 1)
print(f"\nFit result (trained on 80% of points): real_meters = {a:.4f} * relative_depth + {b:.4f}")

absolute_dsm = a * relative_depth + b

# --- Step 5: Evaluate on the held-out 20% the line never saw ---
test_predictions = a * depth_valid[test_idx] + b
test_actual = srtm_valid[test_idx]
residuals = test_predictions - test_actual
rmse = np.sqrt(np.mean(residuals ** 2))
mae = np.mean(np.abs(residuals))
correlation = np.corrcoef(depth_valid, srtm_valid)[0, 1]

print(f"HELD-OUT Test RMSE: {rmse:.2f} m | Test MAE: {mae:.2f} m")
print(f"Correlation (relative depth vs SRTM elevation): {correlation:.3f}")
print("(Correlation near 0 = SRTM and the depth model aren't agreeing meaningfully here.")
print(" Closer to +1 or -1 = a real, usable relationship exists.)")

# --- Step 6: Save the absolute DSM as a proper GeoTIFF ---
os.makedirs(os.path.dirname(OUTPUT_TIF), exist_ok=True)
with rasterio.open(
    OUTPUT_TIF, "w",
    driver="GTiff",
    height=height, width=width,
    count=1, dtype="float32",
    crs=crs, transform=transform,
) as dst:
    dst.write(absolute_dsm.astype(np.float32), 1)

print(f"\nSaved absolute DSM to {OUTPUT_TIF}")