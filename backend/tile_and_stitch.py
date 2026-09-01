import numpy as np
from PIL import Image
from transformers import pipeline

def tile_image(image, tile_size=512, overlap=64):
    """Split image into overlapping tiles. Returns list of (tile, x, y)."""
    w, h = image.size
    stride = tile_size - overlap
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            box = (x, y, min(x + tile_size, w), min(y + tile_size, h))
            tile = image.crop(box)
            tiles.append((tile, x, y))
    return tiles

def stitch_tiles(tile_depths, positions, full_size, tile_size=512, overlap=64):
    """Blend depth tiles back into one full-size map, aligning each tile's
    brightness scale to its already-placed neighbors before blending."""
    w, h = full_size
    accum = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    for depth_arr, (x, y) in zip(tile_depths, positions):
        th, tw = depth_arr.shape

        # --- Scale alignment: shift this tile so its overlap region matches
        # what's already been placed there, instead of trusting its own scale ---
        existing_weight_region = weight[y:y+th, x:x+tw]
        existing_accum_region = accum[y:y+th, x:x+tw]
        overlap_mask = existing_weight_region > 0.01

        aligned_tile = depth_arr
        if np.any(overlap_mask):
            existing_vals = existing_accum_region[overlap_mask] / existing_weight_region[overlap_mask]
            new_vals = depth_arr[overlap_mask]

            existing_mean, existing_std = existing_vals.mean(), existing_vals.std()
            new_mean, new_std = new_vals.mean(), new_vals.std()

            if new_std > 1e-6:
                scale = existing_std / new_std
            else:
                scale = 1.0

            aligned_tile = (depth_arr - new_mean) * scale + existing_mean

        # --- Feathering weight mask (fades tile edges into neighbors) ---
        wy = np.ones(th, dtype=np.float32)
        wx = np.ones(tw, dtype=np.float32)
        fade_y = min(overlap, th // 2)
        fade_x = min(overlap, tw // 2)
        if fade_y > 0:
            ramp_y = np.linspace(0, 1, fade_y)
            wy[:fade_y] = ramp_y
            wy[-fade_y:] = ramp_y[::-1]
        if fade_x > 0:
            ramp_x = np.linspace(0, 1, fade_x)
            wx[:fade_x] = ramp_x
            wx[-fade_x:] = ramp_x[::-1]
        mask = np.outer(wy, wx)

        accum[y:y+th, x:x+tw] += aligned_tile * mask
        weight[y:y+th, x:x+tw] += mask

    weight[weight == 0] = 1
    stitched = accum / weight
    return stitched

def run_tiled_depth(image_path, output_path, tile_size=512, overlap=64):
    print("Loading model...")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    print(f"Full image size: {w}x{h}")

    tiles = tile_image(image, tile_size, overlap)
    print(f"Split into {len(tiles)} tiles")

    tile_depths = []
    positions = []
    for i, (tile, x, y) in enumerate(tiles):
        print(f"Running tile {i+1}/{len(tiles)} at ({x},{y})")
        result = pipe(tile)
        depth_img = result["depth"]  # PIL image
        depth_arr = np.array(depth_img).astype(np.float32)
        tile_depths.append(depth_arr)
        positions.append((x, y))

    print("Stitching tiles together...")
    stitched = stitch_tiles(tile_depths, positions, (w, h), tile_size, overlap)

    # normalize to 0-255 for saving as viewable PNG
    stitched_norm = (255 * (stitched - stitched.min()) / (stitched.max() - stitched.min() + 1e-8)).astype(np.uint8)
    Image.fromarray(stitched_norm).save(output_path)
    print(f"Saved stitched depth map to {output_path}")

if __name__ == "__main__":
    run_tiled_depth(
        image_path="../data/test_city.jpg",
        output_path="../data/test_city_tiled_output.png",
        tile_size=512,
        overlap=64
    )