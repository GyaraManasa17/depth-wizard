from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import io

from tile_and_stitch import tile_image, stitch_tiles
from transformers import pipeline

app = FastAPI(title="SatQuery AI - Depth Backend")

# Load the model ONCE when the server starts, not on every request
# (loading it per-request would make every call painfully slow)
print("Loading depth model at startup...")
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
print("Model ready.")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Depth backend is running"}


@app.post("/predict-depth")
async def predict_depth(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    w, h = image.size

    # Tile, run model on each tile, stitch back together
    tiles = tile_image(image, tile_size=512, overlap=64)
    tile_depths = []
    positions = []
    for tile, x, y in tiles:
        result = depth_pipe(tile)
        depth_arr = np.array(result["depth"]).astype(np.float32)
        tile_depths.append(depth_arr)
        positions.append((x, y))

    stitched = stitch_tiles(tile_depths, positions, (w, h), tile_size=512, overlap=64)

    # Normalize to viewable 0-255 PNG
    stitched_norm = (255 * (stitched - stitched.min()) / (stitched.max() - stitched.min() + 1e-8)).astype(np.uint8)
    output_image = Image.fromarray(stitched_norm)

    # Send the image back as the HTTP response, no temp file needed
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")