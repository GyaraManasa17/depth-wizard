import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
import rasterio
from rasterio.transform import xy
import srtm
from transformers import pipeline
from PIL import Image

from model_config import MODEL_ID
from terrain_classifier import classify_terrain, get_calibration_strategy
from calibration_utils import smooth_depth, _robust_outlier_mask


def load_geotiff(path):
    with rasterio.open(path) as src:
        img_array = src.read([1, 2, 3])
        transform = src.transform
        crs = src.crs
        width, height = src.width, src.height
    img_rgb = np.transpose(img_array, (1, 2, 0)).astype(np.uint8)
    return Image.fromarray(img_rgb), transform, crs, width, height


def get_srtm_grid(transform, width, height, elevation_data):
    grid = np.zeros((height, width), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            lon, lat = xy(transform, row, col)
            elev = elevation_data.get_elevation(lat, lon)
            grid[row, col] = elev if elev is not None else np.nan
    return grid


def evaluate_sample(label, tif_path, pipe, elevation_data):
    print(f"\n==========================================")
    print(f"Evaluating: {label} ({tif_path})")
    print(f"==========================================")
    image, transform, crs, width, height = load_geotiff(tif_path)
    
    # 1. Depth prediction
    result = pipe(image)
    raw_depth = np.array(result["depth"]).astype(np.float64)
    
    # 2. Ground truth SRTM
    srtm_grid = get_srtm_grid(transform, width, height, elevation_data)
    valid_mask = ~np.isnan(srtm_grid)
    
    if np.sum(valid_mask) < 20:
        print(f"Warning: insufficient valid SRTM points for {label}")
        return None

    # 3. Detect terrain type
    terrain_type = classify_terrain(image)
    strategy = get_calibration_strategy(terrain_type)
    print(f"Detected Terrain: {terrain_type.upper()} | Strategy: {strategy}")

    # Train/Test Split (80/20) for fair evaluation
    np.random.seed(42)
    valid_indices = np.argwhere(valid_mask)
    n = len(valid_indices)
    shuffled = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = shuffled[:split], shuffled[split:]

    train_coords = valid_indices[train_idx]
    test_coords = valid_indices[test_idx]

    # --- BASELINE EVALUATION (Simple Global Linear Fit) ---
    train_depth_base = raw_depth[train_coords[:, 0], train_coords[:, 1]]
    train_srtm = srtm_grid[train_coords[:, 0], train_coords[:, 1]]
    test_depth_base = raw_depth[test_coords[:, 0], test_coords[:, 1]]
    test_srtm = srtm_grid[test_coords[:, 0], test_coords[:, 1]]

    a_base, b_base = np.polyfit(train_depth_base, train_srtm, 1)
    pred_base = a_base * test_depth_base + b_base

    res_base = pred_base - test_srtm
    rmse_base = float(np.sqrt(np.mean(res_base ** 2)))
    mae_base = float(np.mean(np.abs(res_base)))
    corr_base = float(np.corrcoef(raw_depth[valid_mask].ravel(), srtm_grid[valid_mask].ravel())[0, 1])

    # --- TERRAIN-ADAPTIVE EVALUATION ---
    pre_smooth = strategy.get("pre_smooth", False)
    eval_depth = smooth_depth(raw_depth) if pre_smooth else raw_depth

    train_depth_adapt = eval_depth[train_coords[:, 0], train_coords[:, 1]]
    test_depth_adapt = eval_depth[test_coords[:, 0], test_coords[:, 1]]

    method = strategy.get("method", "linear")
    if method == "polynomial":
        degree = strategy.get("degree", 2)
        coeffs = np.polyfit(train_depth_adapt, train_srtm, degree)
        inliers = _robust_outlier_mask(train_depth_adapt, train_srtm, coeffs)
        if np.sum(inliers) > 10:
            coeffs = np.polyfit(train_depth_adapt[inliers], train_srtm[inliers], degree)
        pred_adapt = np.polyval(coeffs, test_depth_adapt)
    else:
        coeffs = np.polyfit(train_depth_adapt, train_srtm, 1)
        inliers = _robust_outlier_mask(train_depth_adapt, train_srtm, coeffs)
        if np.sum(inliers) > 10:
            coeffs = np.polyfit(train_depth_adapt[inliers], train_srtm[inliers], 1)
        pred_adapt = coeffs[0] * test_depth_adapt + coeffs[1]

    res_adapt = pred_adapt - test_srtm
    rmse_adapt = float(np.sqrt(np.mean(res_adapt ** 2)))
    mae_adapt = float(np.mean(np.abs(res_adapt)))
    corr_adapt = float(np.corrcoef(eval_depth[valid_mask].ravel(), srtm_grid[valid_mask].ravel())[0, 1])

    print(f"Baseline (Global Linear): RMSE = {rmse_base:.2f}m | MAE = {mae_base:.2f}m | Corr = {corr_base:.3f}")
    print(f"Adaptive ({terrain_type}): RMSE = {rmse_adapt:.2f}m | MAE = {mae_adapt:.2f}m | Corr = {corr_adapt:.3f}")
    print(f"Improvement: RMSE Delta = {rmse_base - rmse_adapt:+.2f}m ({((rmse_base - rmse_adapt) / rmse_base) * 100:.1f}%)")

    return {
        "terrain": label,
        "classified_as": terrain_type,
        "method": method,
        "base_rmse": round(rmse_base, 2),
        "adapt_rmse": round(rmse_adapt, 2),
        "base_mae": round(mae_base, 2),
        "adapt_mae": round(mae_adapt, 2),
        "correlation": round(corr_adapt, 3),
        "rmse_reduction_pct": round(((rmse_base - rmse_adapt) / rmse_base) * 100, 1)
    }


if __name__ == "__main__":
    TEST_SETS = [
        ("Urban (City)", os.path.join(os.path.dirname(__file__), "..", "data", "test_city_geo.tiff")),
        ("Hilly / Mountainous", os.path.join(os.path.dirname(__file__), "..", "data", "test_hills_geo.tiff")),
        ("Forested Landscape", os.path.join(os.path.dirname(__file__), "..", "data", "test_forest_geo.tiff")),
        ("Sparse / Semi-Arid", os.path.join(os.path.dirname(__file__), "..", "data", "test_sparse_geo.tiff")),
        ("Mixed Geomorphology", os.path.join(os.path.dirname(__file__), "..", "data", "test_mixed_geo.tiff")),
    ]

    print(f"Loading Elevation Reference Source (SRTM)...")
    elevation_data = srtm.get_data()

    print(f"Loading Model: {MODEL_ID}...")
    pipe = pipeline(task="depth-estimation", model=MODEL_ID)

    results = []
    for label, path in TEST_SETS:
        if not os.path.exists(path):
            print(f"Skipping {label} - file not found: {path}")
            continue
        res = evaluate_sample(label, path, pipe, elevation_data)
        if res:
            results.append(res)

    print("\n\n" + "=" * 80)
    print("FINAL BENCHMARK: SIH 2026 EVALUATION METRICS (50% SCORE CRITERIA)")
    print("=" * 80)
    print(f"{'Terrain Type':<22} {'Classified':<12} {'Base RMSE':<11} {'Adapt RMSE':<12} {'Adapt MAE':<11} {'Correlation':<12} {'Improvement'}")
    print("-" * 90)
    for r in results:
        print(f"{r['terrain']:<22} {r['classified_as']:<12} {r['base_rmse']:<11.2f} {r['adapt_rmse']:<12.2f} {r['adapt_mae']:<11.2f} {r['correlation']:<12.3f} +{r['rmse_reduction_pct']}%")
    print("=" * 80)