"""
Satellite image enhancement — preprocessing that helps the depth model
find sharper edges and real structure in aerial/satellite imagery.

These are standard remote sensing techniques, not cosmetic filters:
  - CLAHE: locally adaptive contrast (reveals detail in dark shadows AND bright rooftops)
  - Unsharp mask: edge sharpening (makes building boundaries crisper for the depth model)
  - Dehaze: reduces atmospheric haze that washes out distant terrain

Usage:
    from enhance import enhance_satellite_image
    enhanced = enhance_satellite_image(pil_image)
"""

import cv2
import numpy as np
from PIL import Image


def apply_clahe(img_array, clip_limit=2.0, grid_size=8):
    """Contrast Limited Adaptive Histogram Equalization.
    Unlike global histogram equalization, CLAHE works on local regions —
    so it brings out detail in both dark valleys and bright rooftops
    without blowing out either end."""
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def apply_unsharp_mask(img_array, sigma=1.0, strength=0.5):
    """Sharpens edges by subtracting a blurred version from the original.
    This makes building boundaries, roads, and terrain ridges more
    pronounced — exactly the features the depth model relies on."""
    blurred = cv2.GaussianBlur(img_array, (0, 0), sigma)
    sharpened = cv2.addWeighted(img_array, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_dehaze(img_array, strength=0.5):
    """Simple dark-channel-prior dehazing. Satellite images taken through
    thick atmosphere (haze, smog, high altitude) lose contrast in distant
    areas — this recovers it. The strength parameter (0-1) controls how
    aggressively haze is removed."""
    img_float = img_array.astype(np.float64) / 255.0

    # Dark channel: min across RGB in a local patch
    min_rgb = np.min(img_float, axis=2)
    kernel_size = max(15, min(img_array.shape[:2]) // 30) | 1  # ensure odd
    dark_channel = cv2.erode(min_rgb, np.ones((kernel_size, kernel_size)))

    # Estimate atmospheric light from the brightest region of the dark channel
    flat_dark = dark_channel.ravel()
    top_indices = np.argsort(flat_dark)[-max(1, len(flat_dark) // 1000):]
    h, w = dark_channel.shape
    atmos_light = np.zeros(3)
    for idx in top_indices:
        row, col = divmod(int(idx), w)
        atmos_light += img_float[row, col]
    atmos_light /= len(top_indices)

    # Transmission map (how much scene radiance reaches the camera)
    transmission = 1.0 - strength * dark_channel
    transmission = np.clip(transmission, 0.1, 1.0)

    # Recover scene radiance
    result = np.zeros_like(img_float)
    for c in range(3):
        result[:, :, c] = (img_float[:, :, c] - atmos_light[c]) / transmission + atmos_light[c]

    return np.clip(result * 255, 0, 255).astype(np.uint8)


def enhance_satellite_image(pil_image, clahe=True, sharpen=True, dehaze=True,
                            clahe_clip=2.0, sharpen_strength=0.5, dehaze_strength=0.5):
    """Full enhancement pipeline for satellite/aerial imagery.

    Args:
        pil_image: PIL Image (RGB)
        clahe: Apply local contrast enhancement
        sharpen: Apply edge sharpening
        dehaze: Apply atmospheric haze removal
        clahe_clip: CLAHE clip limit (higher = more aggressive contrast)
        sharpen_strength: Sharpening intensity (0.3-1.0 typical)
        dehaze_strength: Dehazing intensity (0.3-0.7 typical)

    Returns:
        Enhanced PIL Image (RGB)
    """
    img = np.array(pil_image)

    if dehaze:
        img = apply_dehaze(img, strength=dehaze_strength)

    if clahe:
        img = apply_clahe(img, clip_limit=clahe_clip)

    if sharpen:
        img = apply_unsharp_mask(img, strength=sharpen_strength)

    return Image.fromarray(img)


if __name__ == "__main__":
    """Quick test — run directly to compare before/after on a sample image."""
    import sys
    import os

    image_path = sys.argv[1] if len(sys.argv) > 1 else "../data/test_hills.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    enhanced = enhance_satellite_image(image)

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_enhanced{ext}"
    enhanced.save(output_path)
    print(f"Enhanced image saved to {output_path}")
    print(f"Original size: {image.size}, Enhanced size: {enhanced.size}")
