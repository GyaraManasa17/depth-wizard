from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io

from tile_and_stitch import tile_image, stitch_tiles
import torch
from transformers import pipeline
from model_config import MODEL_ID
from enhance import enhance_satellite_image
from terrain_classifier import classify_terrain, get_calibration_strategy

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
device = 0 if torch.cuda.is_available() else -1
depth_pipe = pipeline(task="depth-estimation", model=MODEL_ID, device=device)
print(f"Model ready on device: {'GPU' if device == 0 else 'CPU'}")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Depth backend is running"}


TILE_SIZE = 512
TILE_OVERLAP = 192  # increased from 128 for smoother seams


def _run_tiled_depth(image):
    """Shared tile-and-stitch logic used by all depth endpoints."""
    w, h = image.size
    tiles = tile_image(image, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
    tile_depths = []
    positions = []
    for tile, x, y in tiles:
        result = depth_pipe(tile)
        depth_arr = np.array(result["depth"]).astype(np.float32)
        tile_depths.append(depth_arr)
        positions.append((x, y))
    return stitch_tiles(tile_depths, positions, (w, h), tile_size=TILE_SIZE, overlap=TILE_OVERLAP)


def _enhance_image(contents, clahe_clip, sharpen_strength, dehaze_strength):
    """Read uploaded bytes, enhance, return PIL image."""
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    return enhance_satellite_image(
        image, clahe_clip=clahe_clip,
        sharpen_strength=sharpen_strength, dehaze_strength=dehaze_strength,
    )


@app.post("/predict-depth")
async def predict_depth(
    file: UploadFile = File(...),
    clahe_clip: float = Query(2.0, description="CLAHE clip limit (higher = more contrast)"),
    sharpen_strength: float = Query(0.5, description="Edge sharpening intensity (0.3-1.0)"),
    dehaze_strength: float = Query(0.5, description="Atmospheric haze removal (0.3-0.7)"),
):
    contents = await file.read()
    image = _enhance_image(contents, clahe_clip, sharpen_strength, dehaze_strength)
    stitched = _run_tiled_depth(image)

    # Normalize to viewable 0-255 PNG
    stitched_norm = (255 * (stitched - stitched.min()) / (stitched.max() - stitched.min() + 1e-8)).astype(np.uint8)
    output_image = Image.fromarray(stitched_norm)

    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/predict-depth-data")
async def predict_depth_data(
    file: UploadFile = File(...),
    clahe_clip: float = Query(2.0),
    sharpen_strength: float = Query(0.5),
    dehaze_strength: float = Query(0.5),
):
    """Returns raw float32 depth values as JSON for accurate height querying
    in the frontend. Also returns dimensions for reconstruction."""
    contents = await file.read()
    image = _enhance_image(contents, clahe_clip, sharpen_strength, dehaze_strength)
    stitched = _run_tiled_depth(image)
    w, h = image.size

    # Normalize to 0-1 range
    d_min, d_max = float(stitched.min()), float(stitched.max())
    normalized = ((stitched - d_min) / (d_max - d_min + 1e-8)).astype(np.float32)

    # Also return the 8-bit PNG for texture
    stitched_norm = (normalized * 255).astype(np.uint8)
    output_image = Image.fromarray(stitched_norm)
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    depth_png_b64 = __import__('base64').b64encode(buf.getvalue()).decode('ascii')

    return JSONResponse(content={
        "width": w,
        "height": h,
        "depth_min": d_min,
        "depth_max": d_max,
        "depth_png_base64": depth_png_b64,
        "depth_values": normalized.tolist(),
    })

@app.post("/predict-elevation-geotiff")
async def predict_elevation_geotiff(
    file: UploadFile = File(...),
    clahe_clip: float = Query(2.0, description="CLAHE clip limit"),
    sharpen_strength: float = Query(0.5, description="Edge sharpening intensity"),
    dehaze_strength: float = Query(0.5, description="Haze removal intensity"),
):
    """Takes a georeferenced GeoTIFF, returns an absolute-elevation DSM GeoTIFF (real meters)."""
    contents = await file.read()

    from calibration_utils import NotGeoreferencedError
    try:
        img_rgb, transform, crs, width, height, valid_mask = load_geotiff_from_bytes(contents)
    except NotGeoreferencedError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not read this file: {str(e)}"})

    image = enhance_satellite_image(
        Image.fromarray(img_rgb), clahe_clip=clahe_clip,
        sharpen_strength=sharpen_strength, dehaze_strength=dehaze_strength,
    )

    # Classify terrain and pick calibration strategy
    terrain_type = classify_terrain(image)
    strategy = get_calibration_strategy(terrain_type)
    print(f"Detected terrain: {terrain_type}, strategy: {strategy}")

    relative_depth = _run_tiled_depth(image).astype(np.float64)
    relative_depth[~valid_mask] = np.nan

    # Get real SRTM elevation and calibrate with terrain-adaptive strategy
    srtm_grid = get_srtm_grid(transform, width, height)
    absolute_dsm, a, b = calibrate_to_absolute(
        relative_depth, srtm_grid, transform=transform, strategy=strategy
    )
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
            "X-Elevation-Max": f"{elevation_max:.2f}",
            "X-Terrain-Type": terrain_type,
        }
    )

@app.post("/geotiff-preview")
async def geotiff_preview(file: UploadFile = File(...)):
    """Converts a GeoTIFF's RGB content to a PNG the browser can actually display."""
    contents = await file.read()
    img_rgb, _, _, _, _, _ = load_geotiff_from_bytes(contents)
    image = Image.fromarray(img_rgb)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# Serve the frontend's static files directly from this same server.
# Must be mounted LAST, after all API routes, so it doesn't shadow them.
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")