import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.io import MemoryFile
import srtm

_elevation_data = None

def get_elevation_source():
    """Load SRTM data once and reuse it across requests."""
    global _elevation_data
    if _elevation_data is None:
        _elevation_data = srtm.get_data()
    return _elevation_data

def load_geotiff_from_bytes(file_bytes):
    """Open a GeoTIFF from raw uploaded bytes (no temp file needed)."""
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            img_array = src.read([1, 2, 3])
            transform = src.transform
            crs = src.crs
            width, height = src.width, src.height
    img_rgb = np.transpose(img_array, (1, 2, 0)).astype(np.uint8)
    return img_rgb, transform, crs, width, height

def get_srtm_grid(transform, width, height):
    elevation_data = get_elevation_source()
    grid = np.zeros((height, width), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            lon, lat = xy(transform, row, col)
            elev = elevation_data.get_elevation(lat, lon)
            grid[row, col] = elev if elev is not None else np.nan
    return grid

def calibrate_to_absolute(relative_depth, srtm_grid):
    """Fit real_meters = a * relative_depth + b using all valid pixels,
    return the absolute elevation grid plus the fit parameters."""
    valid_mask = ~np.isnan(srtm_grid)
    a, b = np.polyfit(relative_depth[valid_mask].ravel(), srtm_grid[valid_mask].ravel(), 1)
    absolute_dsm = a * relative_depth + b
    return absolute_dsm, float(a), float(b)

def save_dsm_geotiff(absolute_dsm, transform, crs, output_path):
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=absolute_dsm.shape[0], width=absolute_dsm.shape[1],
        count=1, dtype="float32",
        crs=crs, transform=transform,
    ) as dst:
        dst.write(absolute_dsm.astype(np.float32), 1)