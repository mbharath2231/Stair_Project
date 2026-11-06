"""
STAIR Diagnostics, Quality Tracking, and Agricultural Sampling
==============================================================
Provides tools for:
  - "Staircase" pixel origin tracking (Original vs Temporal vs Spatial)
  - USDA Cropland Data Layer (CDL) sampling for targeted crop monitoring
  - 5km bounding chip generation around agricultural center points
  - Time-lapse stack generation for validation
"""

from typing import List, Optional, Tuple
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

