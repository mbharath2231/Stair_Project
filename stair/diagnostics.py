"""
STAIR Diagnostics, Quality Tracking, and Agricultural Sampling
==============================================================
Provides tools for:
  - "Staircase" pixel origin tracking (Original vs Temporal vs Spatial)
  - USDA Cropland Data Layer (CDL) sampling for targeted crop monitoring
  - 5km bounding chip generation around agricultural center points
  - Time-lapse stack generation for validation
"""

from typing import Tuple, List, Optional
import ee
import config


def create_staircase_tracker(
    raw_mask: ee.Image,
    temporal_mask: ee.Image,
    aoi: Optional[ee.Geometry] = None
) -> ee.Image:
    """
    Generate a diagnostic classification tracker layer:
      - Class 0: Original clear pixel (preserved from raw data)
      - Class 1: Temporally imputed pixel (filled via Section 2.2.1 linear regression)
      - Class 2: Spatially / adaptively patched pixel (filled via Section 2.2.2)
      
    Args:
        raw_mask: Binary mask of target raw image (1 = clear, 0 = cloud/gap).
        temporal_mask: Binary mask of temporal regression availability.
        aoi: Optional region geometry to clip to.
        
    Returns:
        ee.Image with discrete values [0, 1, 2].
    """
    tracker = (
        ee.Image(0)
        .where(raw_mask.eq(0).And(temporal_mask.eq(1)), 1)
        .where(raw_mask.eq(0).And(temporal_mask.eq(0)), 2)
        .rename("staircase_tracker")
    )
    if aoi is not None:
        tracker = tracker.clip(aoi)
    return tracker


def sample_agricultural_pixel(
    aoi: ee.Geometry,
    year: int = 2021,
    crop_codes: Optional[List[int]] = None
) -> Tuple[ee.Geometry.Point, float, float]:
    """
    Sample an agricultural coordinate (default: Corn=1, Soy=5) using USDA CDL.
    
    Args:
        aoi: Area of interest geometry.
        year: Year for USDA Cropland Data Layer. Defaults to 2021.
        crop_codes: List of CDL crop integer codes. Defaults to [1, 5] (Corn, Soy).
        
    Returns:
        Tuple of (ee.Geometry.Point, longitude, latitude)
    """
    if crop_codes is None:
        crop_codes = [1, 5]
        
    cdl = ee.Image(f"USDA/NASS/CDL/{year}").select("cropland")
    
    crop_mask = ee.Image(0)
    for code in crop_codes:
        crop_mask = crop_mask.Or(cdl.eq(code))
        
    sampled_points = crop_mask.updateMask(crop_mask).sample(
        region=aoi,
        scale=config.DEFAULT_SCALE,
        numPixels=50,
        geometries=True
    )
    
    features = sampled_points.getInfo().get("features", [])
    if not features:
        raise ValueError(
            f"No agricultural pixels matching codes {crop_codes} found in provided AOI."
        )
        
    coords = features[0]["geometry"]["coordinates"]
    lon, lat = float(coords[0]), float(coords[1])
    point = ee.Geometry.Point([lon, lat])
    return point, lon, lat


def create_chip_bounds(
    center_point: ee.Geometry.Point,
    radius_meters: float = 2500.0
) -> ee.Geometry:
    """
    Create a square bounding box chip (e.g. 5km x 5km with 2500m radius) around a center point.
    
    Args:
        center_point: Center point geometry.
        radius_meters: Buffer radius in meters. Default 2500m (5km diameter).
        
    Returns:
        ee.Geometry rectangular bounds.
    """
    return center_point.buffer(radius_meters).bounds()


def build_time_series_frame(
    raw_image: ee.Image,
    slope: ee.Image,
    intercept: ee.Image,
    band_name: str = "SR_B5",
    bounds: Optional[ee.Geometry] = None
) -> ee.Image:
    """
    Build a 3-layer diagnostic stack for a single time series overpass:
      - Band 1: Raw data (including clouds)
      - Band 2: STAIR imputed only (mathematical prediction)
      - Band 3: Final patched data (clear pixels + imputed holes)
      
    Args:
        raw_image: Unmasked Landsat overpass image.
        slope: Temporal regression slope.
        intercept: Temporal regression intercept.
        band_name: Spectral band.
        bounds: Optional bounding geometry to clip to.
        
    Returns:
        3-band ee.Image with double precision.
    """
    from .core import mask_landsat_clouds
    
    raw_band = raw_image.select(band_name).rename("Band1_Raw_With_Clouds")
    
    date_days = (
        ee.Image(raw_image.date().millis())
        .divide(1000 * 60 * 60 * 24)
        .toFloat()
    )
    stair_imputed = date_days.multiply(slope).add(intercept).rename("Band2_STAIR_Imputed_Only")
    
    masked_band = mask_landsat_clouds(raw_image).select(band_name)
    final_patched = masked_band.unmask(stair_imputed).rename("Band3_Final_Patched")
    
    stack = ee.Image([raw_band, stair_imputed, final_patched]).toDouble()
    if bounds is not None:
        stack = stack.clip(bounds)
    return stack
