import os
from PIL import Image
import numpy as np
from tile_and_stitch import tile_image, stitch_tiles
from transformers import pipeline
from model_config import MODEL_ID


# (source filename in data/, output name to use in sample-outputs/)
IMAGES = [
    ("test_city.jpg", "city"),
    ("test_hills.png", "hills"),
    ("sample_sparse.png", "sparse"),
    ("sample_forest.png", "forest"),
    ("sample_mixed.png", "mixed"),
]

DATA_DIR = "../data"
OUTPUT_DIR = "../data/sample-outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
pipe = pipeline(task="depth-estimation", model=MODEL_ID)
print("Model ready.\n")

for filename, label in IMAGES:
    src_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(src_path):
        print(f"SKIPPING {filename} - file not found at {src_path}")
        continue

    print(f"Processing {filename} ({label})...")
    image = Image.open(src_path).convert("RGB")
    w, h = image.size

    tiles = tile_image(image, tile_size=512, overlap=64)
    tile_depths = []
    positions = []
    for tile, x, y in tiles:
        result = pipe(tile)
        depth_arr = np.array(result["depth"]).astype(np.float32)
        tile_depths.append(depth_arr)
        positions.append((x, y))

    stitched = stitch_tiles(tile_depths, positions, (w, h), tile_size=512, overlap=64)
    stitched_norm = (255 * (stitched - stitched.min()) / (stitched.max() - stitched.min() + 1e-8)).astype(np.uint8)

    # Save the depth map
    depth_out_path = os.path.join(OUTPUT_DIR, f"{label}_depth.png")
    Image.fromarray(stitched_norm).save(depth_out_path)

    # Also copy the original image into sample-outputs, same name pattern, so Manasa/Afrah have image+depth pairs together
    original_out_path = os.path.join(OUTPUT_DIR, f"{label}_original{os.path.splitext(filename)[1]}")
    image.save(original_out_path)

    print(f"  -> saved {depth_out_path}")
    print(f"  -> saved {original_out_path}\n")

print("All done.")