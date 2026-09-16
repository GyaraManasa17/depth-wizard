import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.io import MemoryFile
import srtm
from scipy.ndimage import uniform_filter

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
    valid_mask = np.any(img_rgb > 0, axis=-1)  # True where pixel is not pure black [0, 0, 0]
    return img_rgb, transform, crs, width, height, valid_mask

def get_srtm_grid(transform, width, height):
    elevation_data = get_elevation_source()
    grid = np.zeros((height, width), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            lon, lat = xy(transform, row, col)
            elev = elevation_data.get_elevation(lat, lon)
            grid[row, col] = elev if elev is not None else np.nan
    return grid

def block_average(arr, block_size):
    """Downsample by averaging NxN blocks — matches SRTM's coarser resolution
    so we're not fitting noisy per-pixel values against ~30m ground truth."""
    h, w = arr.shape
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    trimmed = arr[:h_trim, :w_trim]
    reshaped = trimmed.reshape(h_trim // block_size, block_size, w_trim // block_size, block_size)
    return np.nanmean(reshaped, axis=(1, 3))


def smooth_depth(depth_arr, kernel_size=5):
    """Light spatial smoothing on the depth map before calibration.
    Helps with noisy canopy (forest) or tile-seam artifacts."""
    return uniform_filter(depth_arr.astype(np.float64), size=kernel_size)


def _robust_outlier_mask(depth_valid, srtm_valid, coeffs):
    """IQR-based outlier rejection — shared by all calibration methods."""
    if len(coeffs) == 2:
        predicted = coeffs[0] * depth_valid + coeffs[1]
    else:
        predicted = np.polyval(coeffs, depth_valid)
    residuals = predicted - srtm_valid
    q1, q3 = np.percentile(residuals, [25, 75])
    iqr = q3 - q1
    return (residuals >= q1 - 1.5 * iqr) & (residuals <= q3 + 1.5 * iqr)


def _prepare_fit_data(relative_depth, srtm_grid, transform, pre_smooth=False):
    """Block-average and mask NaN pixels. Returns (depth_for_fit, srtm_for_fit, block_size)."""
    depth_for_fit = relative_depth
    if pre_smooth:
        depth_for_fit = smooth_depth(depth_for_fit)

    srtm_for_fit = srtm_grid
    block_size = 1

    if transform is not None:
        pixel_size_deg = abs(transform[0])
        meters_per_pixel = pixel_size_deg * 111320
        block_size = max(1, round(30 / meters_per_pixel))
        if block_size > 1:
            depth_for_fit = block_average(depth_for_fit, block_size)
            srtm_for_fit = block_average(srtm_grid, block_size)

    valid_mask = ~np.isnan(srtm_for_fit) & ~np.isnan(depth_for_fit)
    return depth_for_fit[valid_mask].ravel(), srtm_for_fit[valid_mask].ravel()


def calibrate_linear(relative_depth, srtm_grid, transform=None, pre_smooth=False):
    """Standard linear fit: real_meters = a * relative_depth + b."""
    depth_valid, srtm_valid = _prepare_fit_data(
        relative_depth, srtm_grid, transform, pre_smooth
    )

    a, b = np.polyfit(depth_valid, srtm_valid, 1)
    inlier_mask = _robust_outlier_mask(depth_valid, srtm_valid, [a, b])
    if np.sum(inlier_mask) > 10:
        a, b = np.polyfit(depth_valid[inlier_mask], srtm_valid[inlier_mask], 1)

    src = smooth_depth(relative_depth) if pre_smooth else relative_depth
    absolute_dsm = a * src + b
    return absolute_dsm, float(a), float(b)


def calibrate_polynomial(relative_depth, srtm_grid, transform=None, degree=2, pre_smooth=True):
    """Polynomial fit for terrain with non-linear depth-to-elevation mapping
    (e.g. hilly terrain where a single line can't capture curvature)."""
    depth_valid, srtm_valid = _prepare_fit_data(
        relative_depth, srtm_grid, transform, pre_smooth
    )

    coeffs = np.polyfit(depth_valid, srtm_valid, degree)
    inlier_mask = _robust_outlier_mask(depth_valid, srtm_valid, coeffs)
    if np.sum(inlier_mask) > 10:
        coeffs = np.polyfit(depth_valid[inlier_mask], srtm_valid[inlier_mask], degree)

    src = smooth_depth(relative_depth) if pre_smooth else relative_depth
    absolute_dsm = np.polyval(coeffs, src)
    return absolute_dsm, coeffs[0], coeffs[-1]  # return leading coeff and intercept


def calibrate_to_absolute(relative_depth, srtm_grid, transform=None, strategy=None):
    """Main calibration entry point. Auto-selects method based on strategy dict.

    Args:
        relative_depth: 2D array of model-predicted relative depth values
        srtm_grid: 2D array of SRTM real elevation (meters), same shape
        transform: optional rasterio Affine transform
        strategy: optional dict from terrain_classifier.get_calibration_strategy().
                  If None, falls back to standard linear fit for backward compat.

    Returns:
        (absolute_dsm, a, b) — calibrated elevation grid and fit parameters
    """
    if strategy is None:
        strategy = {"method": "linear", "pre_smooth": False}

    method = strategy.get("method", "linear")
    pre_smooth = strategy.get("pre_smooth", False)

    if method == "polynomial":
        degree = strategy.get("degree", 2)
        return calibrate_polynomial(
            relative_depth, srtm_grid, transform, degree=degree, pre_smooth=pre_smooth
        )
    else:
        return calibrate_linear(
            relative_depth, srtm_grid, transform, pre_smooth=pre_smooth
        )


def save_dsm_geotiff(absolute_dsm, transform, crs, output_path):
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=absolute_dsm.shape[0], width=absolute_dsm.shape[1],
        count=1, dtype="float32",
        crs=crs, transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(absolute_dsm.astype(np.float32), 1)