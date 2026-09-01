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
    """Blend depth tiles back into one full-size map using a weighted average in overlap zones."""
    w, h = full_size
    accum = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    for depth_arr, (x, y) in zip(tile_depths, positions):
        th, tw = depth_arr.shape
        # simple weight mask: full weight in center, fades to 0 at edges (feathering)
        wy = np.ones(th, dtype=np.float32)
        wx = np.ones(tw, dtype=np.float32)
        fade = overlap
        if fade > 0:
            ramp = np.linspace(0, 1, fade)
            wy[:fade] = ramp
            wy[-fade:] = ramp[::-1]
            wx[:fade] = ramp
            wx[-fade:] = ramp[::-1]
        mask = np.outer(wy, wx)

        accum[y:y+th, x:x+tw] += depth_arr * mask
        weight[y:y+th, x:x+tw] += mask

    weight[weight == 0] = 1  # avoid divide-by-zero
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