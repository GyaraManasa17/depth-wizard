from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io

from tile_and_stitch import tile_image, stitch_tiles
from transformers import pipeline
from model_config import MODEL_ID

import tempfile
import os
from calibration_utils import load_geotiff_from_bytes, get_srtm_grid, calibrate_to_absolute, save_dsm_geotiff
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI(title="Depth Wizard - Depth Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; restrict this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Fit-Params", "X-Elevation-Min", "X-Elevation-Max"],
)
# Load the model ONCE when the server starts, not on every request
# (loading it per-request would make every call painfully slow)
print("Loading depth model at startup...")
depth_pipe = pipeline(task="depth-estimation", model=MODEL_ID)
print("Model ready.")


@app.get("/health")
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

@app.post("/predict-elevation-geotiff")
async def predict_elevation_geotiff(file: UploadFile = File(...)):
    """Takes a georeferenced GeoTIFF, returns an absolute-elevation DSM GeoTIFF (real meters)."""
    contents = await file.read()

    from calibration_utils import NotGeoreferencedError
    try:
        img_rgb, transform, crs, width, height = load_geotiff_from_bytes(contents)
    except NotGeoreferencedError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not read this file: {str(e)}"})

    image = Image.fromarray(img_rgb)

    # Run the same depth model already loaded at startup
    result = depth_pipe(image)
    relative_depth = np.array(result["depth"]).astype(np.float64)

    # Get real SRTM elevation for this exact area and fit the correction
    srtm_grid = get_srtm_grid(transform, width, height)
    absolute_dsm, a, b = calibrate_to_absolute(relative_depth, srtm_grid)
    elevation_min = float(np.nanmin(absolute_dsm))
    elevation_max = float(np.nanmax(absolute_dsm))

    # Save to a temp file, then stream it back
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    save_dsm_geotiff(absolute_dsm, transform, crs, tmp_path)

    with open(tmp_path, "rb") as f:
        dsm_bytes = f.read()
    os.remove(tmp_path)

    return StreamingResponse(
        io.BytesIO(dsm_bytes),
        media_type="image/tiff",
        headers={
            "Content-Disposition": "attachment; filename=absolute_dsm.tif",
            "X-Fit-Params": f"a={a:.4f},b={b:.4f}",
            "X-Elevation-Min": f"{elevation_min:.2f}",
            "X-Elevation-Max": f"{elevation_max:.2f}"
        }
    )

@app.post("/geotiff-preview")
async def geotiff_preview(file: UploadFile = File(...)):
    """Converts a GeoTIFF's RGB content to a PNG the browser can actually display."""
    contents = await file.read()
    img_rgb, _, _, _, _ = load_geotiff_from_bytes(contents)
    image = Image.fromarray(img_rgb)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# Serve the frontend's static files directly from this same server.
# Must be mounted LAST, after all API routes, so it doesn't shadow them.
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")