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

class NotGeoreferencedError(Exception):
    """Raised when an uploaded file has no real coordinate data - e.g. a plain photo,
    not an actual GeoTIFF. Must be checked before any elevation lookups are attempted,
    since proceeding without this check causes lookups against meaningless coordinates."""
    pass

def load_geotiff_from_bytes(file_bytes):
    """Open a GeoTIFF from raw uploaded bytes (no temp file needed)."""
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            img_array = src.read([1, 2, 3])
            transform = src.transform
            crs = src.crs
            width, height = src.width, src.height

    if crs is None:
        raise NotGeoreferencedError(
            "This file has no coordinate reference system - it isn't actually "
            "georeferenced. Upload a real GeoTIFF, not a plain photo."
        )

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
    """Fit real_meters = a * relative_depth + b using an 80/20 train/test split
    of valid pixels (never fitting and testing on the same points), then apply
    the fit to the whole image. Returns the absolute elevation grid, fit
    parameters, and honest held-out validation metrics."""
    valid_mask = ~np.isnan(srtm_grid)
    depth_valid = relative_depth[valid_mask].ravel()
    srtm_valid = srtm_grid[valid_mask].ravel()

    np.random.seed(42)
    n = len(depth_valid)
    indices = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    a, b = np.polyfit(depth_valid[train_idx], srtm_valid[train_idx], 1)
    absolute_dsm = a * relative_depth + b

    test_predictions = a * depth_valid[test_idx] + b
    test_actual = srtm_valid[test_idx]
    residuals = test_predictions - test_actual
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    correlation = float(np.corrcoef(depth_valid, srtm_valid)[0, 1])

    metrics = {"rmse": rmse, "mae": mae, "correlation": correlation}
    return absolute_dsm, float(a), float(b), metrics

def save_dsm_geotiff(absolute_dsm, transform, crs, output_path):
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=absolute_dsm.shape[0], width=absolute_dsm.shape[1],
        count=1, dtype="float32",
        crs=crs, transform=transform,
    ) as dst:
        dst.write(absolute_dsm.astype(np.float32), 1)