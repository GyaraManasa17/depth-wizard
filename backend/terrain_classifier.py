"""
Terrain type classifier for satellite imagery — uses simple image statistics
(no ML model needed) to decide which calibration strategy will work best.

This exists because the calibration findings showed that a single global
linear fit works great for cities (0.776 correlation) but fails badly for
hills (0.194). Different terrain types need different calibration strategies.

Categories and their calibration strategies:
  - urban:  Linear fit (strong edges give depth model real structure)
  - forest: Linear fit + smoothing (canopy is noisy but has real depth signal)
  - sparse: Linear fit (flat with scattered features, generally well-behaved)
  - hilly:  Polynomial/piecewise fit (gradual elevation changes fool a single line)
"""

import cv2
import numpy as np
from PIL import Image


def classify_terrain(pil_image):
    """Classify a satellite image into a terrain type using image statistics.

    Args:
        pil_image: PIL Image (RGB)

    Returns:
        str: One of 'urban', 'forest', 'sparse', 'hilly'
    """
    img = np.array(pil_image)

    # --- Feature 1: Edge density (high = urban, low = natural terrain) ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)

    # --- Feature 2: Green ratio (high = forest/vegetation) ---
    r, g, b = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)
    total = r + g + b + 1e-8  # avoid division by zero
    green_ratio = np.mean(g / total)

    # Vegetation index: how much green dominates over red (proxy for NDVI)
    vegetation_index = np.mean((g - r) / (g + r + 1e-8))

    # --- Feature 3: Texture variance (high = complex structure, low = smooth/flat) ---
    # Use Laplacian variance as a texture measure
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = laplacian.var()

    # --- Feature 4: Elevation gradient smoothness ---
    # Large-scale gradient (hilly terrain has smooth, consistent gradients)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_smoothness = np.mean(gradient_magnitude)

    # --- Decision logic ---
    # Urban: lots of edges (buildings, roads, hard geometry)
    if edge_density > 0.08 and texture_variance > 500:
        return "urban"

    # Forest: high green ratio and vegetation index
    if vegetation_index > 0.02 and green_ratio > 0.36:
        return "forest"

    # Hilly: smooth gradients but low edge density (undulating terrain)
    if gradient_smoothness > 15 and edge_density < 0.06:
        return "hilly"

    # Sparse: flat-ish, low vegetation, low edges
    if edge_density < 0.05 and vegetation_index < 0.02:
        return "sparse"

    # Default fallback: mixed terrain (use a moderate calibration strategy)
    # If it doesn't clearly fit any category, treat as sparse which is
    # the safest default (linear fit works reasonably well)
    return "sparse"


def get_calibration_strategy(terrain_type):
    """Return the calibration strategy parameters for a terrain type.

    Returns:
        dict with keys:
            - method: 'linear', 'polynomial', or 'piecewise'
            - degree: polynomial degree (only for 'polynomial')
            - n_segments: number of piecewise segments (only for 'piecewise')
            - pre_smooth: whether to smooth the depth map before calibration
    """
    strategies = {
        "urban": {
            "method": "linear",
            "pre_smooth": False,
        },
        "forest": {
            "method": "linear",
            "pre_smooth": True,
        },
        "sparse": {
            "method": "linear",
            "pre_smooth": False,
        },
        "hilly": {
            "method": "polynomial",
            "degree": 2,
            "pre_smooth": True,
        },
    }
    return strategies.get(terrain_type, strategies["sparse"])


if __name__ == "__main__":
    """Quick test — classify a sample image."""
    import sys
    import os

    image_path = sys.argv[1] if len(sys.argv) > 1 else "../data/test_hills.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    terrain = classify_terrain(image)
    strategy = get_calibration_strategy(terrain)
    print(f"Image: {image_path}")
    print(f"Classified terrain: {terrain}")
    print(f"Calibration strategy: {strategy}")
