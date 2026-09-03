import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
import rasterio
from rasterio.transform import xy
import srtm
from transformers import pipeline
from PIL import Image

def load_geotiff(path):
    with rasterio.open(path) as src:
        img_array = src.read([1, 2, 3])
        transform = src.transform
        width, height = src.width, src.height
    img_rgb = np.transpose(img_array, (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(img_rgb), transform, width, height

def get_srtm_grid(transform, width, height, elevation_data):
    grid = np.zeros((height, width), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            lon, lat = xy(transform, row, col)
            elev = elevation_data.get_elevation(lat, lon)
            grid[row, col] = elev if elev is not None else np.nan
    return grid

def evaluate_terrain(label, tif_path, pipe, elevation_data):
    print(f"\n=== Evaluating: {label} ===")
    image, transform, width, height = load_geotiff(tif_path)
    result = pipe(image)
    relative_depth = np.array(result["depth"]).astype(np.float64)
    srtm_grid = get_srtm_grid(transform, width, height, elevation_data)

    valid_mask = ~np.isnan(srtm_grid)
    depth_valid = relative_depth[valid_mask].ravel()
    srtm_valid = srtm_grid[valid_mask].ravel()

    np.random.seed(42)
    n = len(depth_valid)
    indices = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    a, b = np.polyfit(depth_valid[train_idx], srtm_valid[train_idx], 1)
    predictions = a * depth_valid[test_idx] + b
    actual = srtm_valid[test_idx]
    residuals = predictions - actual

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    correlation = float(np.corrcoef(depth_valid, srtm_valid)[0, 1])

    return {"terrain": label, "rmse_m": round(rmse, 2), "mae_m": round(mae, 2), "correlation": round(correlation, 3)}

if __name__ == "__main__":
    TEST_SETS = [
        ("city", "../data/test_city_geo.tiff"),
        ("hills", "../data/test_hills_geo.tiff"),
    ]

    MODELS = [
        ("Small", "depth-anything/Depth-Anything-V2-Small-hf"),
        ("Base", "depth-anything/Depth-Anything-V2-Base-hf"),
    ]

    elevation_data = srtm.get_data()
    all_results = []

    for model_label, model_id in MODELS:
        print(f"\n\n########## Loading model: {model_label} ({model_id}) ##########")
        pipe = pipeline(task="depth-estimation", model=model_id)

        for label, path in TEST_SETS:
            if not os.path.exists(path):
                print(f"SKIPPING {label} - {path} not found")
                continue
            result = evaluate_terrain(label, path, pipe, elevation_data)
            result["model"] = model_label
            all_results.append(result)

    print("\n\n=== SUMMARY: Small vs Base, by terrain ===")
    print(f"{'Model':<8} {'Terrain':<10} {'RMSE (m)':<12} {'MAE (m)':<12} {'Correlation':<12}")
    for r in all_results:
        print(f"{r['model']:<8} {r['terrain']:<10} {r['rmse_m']:<12} {r['mae_m']:<12} {r['correlation']:<12}")