"""
Core Utilities for Earth Engine, Geospatial Loading, and Preprocessing
"""

from pathlib import Path
from typing import Tuple, Optional
import geopandas as gpd
import ee

import config


def init_earth_engine(project_id: Optional[str] = None) -> None:
    """
    Initialize Google Earth Engine with authentication fallback.
    
    Args:
        project_id: Google Cloud Project ID. Defaults to config.EE_PROJECT_ID.
    """
    proj = project_id or config.EE_PROJECT_ID
    try:
        ee.Initialize(project=proj)
        print(f"Earth Engine initialized successfully with project: '{proj}'")
    except Exception as e:
        print(f"Direct initialization failed ({e}). Triggering authentication...")
        ee.Authenticate()
        ee.Initialize(project=proj)
        print(f"Earth Engine authenticated and online for project: '{proj}'")


def load_aoi(shapefile_path: Optional[Path] = None) -> Tuple[ee.Geometry, gpd.GeoDataFrame]:
    """
    Load Area of Interest boundary from shapefile and reproject to WGS84 (EPSG:4326).
    
    Args:
        shapefile_path: Path to ESRI shapefile. Defaults to config.SHAPEFILE_PATH.
        
    Returns:
        Tuple of (ee.Geometry, GeoDataFrame)
    """
    path = Path(shapefile_path or config.SHAPEFILE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Shapefile not found at {path}. Please place your boundary file in data/."
        )
    
    gdf = gpd.read_file(path)
    if gdf.crs != "EPSG:4326":
        print(f"Reprojecting boundary from {gdf.crs} to EPSG:4326 (WGS84)...")
        gdf = gdf.to_crs("EPSG:4326")
        
    geojson = gdf.geometry.iloc[0].__geo_interface__
    aoi = ee.Geometry(geojson)
    return aoi, gdf


def mask_landsat_clouds(image: ee.Image) -> ee.Image:
    """
    Mask cloud and cloud shadow pixels in Landsat Collection 2 Level-2 Surface Reflectance.
    Uses bits 3 (Cloud Shadow) and 4 (Cloud) from the QA_PIXEL band.
    
    Args:
        image: Landsat 8/9 Image with QA_PIXEL band.
        
    Returns:
        Masked ee.Image.
    """
    qa = image.select(config.LANDSAT_BANDS["qa"])
    cloud_shadow_bitmask = (1 << 3)
    clouds_bitmask = (1 << 4)
    mask = (
        qa.bitwiseAnd(cloud_shadow_bitmask).eq(0)
        .And(qa.bitwiseAnd(clouds_bitmask).eq(0))
    )
    return image.updateMask(mask)


def get_landsat_collection(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud_cover: Optional[float] = None,
    apply_mask: bool = True
) -> ee.ImageCollection:
    """
    Query Landsat 8 surface reflectance collection with optional cloud filtering and masking.
    
    Args:
        aoi: Earth Engine geometry region of interest.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        max_cloud_cover: Optional maximum CLOUD_COVER percentage metadata filter.
        apply_mask: Whether to apply QA_PIXEL cloud masking.
        
    Returns:
        ee.ImageCollection
    """
    col = (
        ee.ImageCollection(config.LANDSAT8_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    if max_cloud_cover is not None:
        col = col.filter(ee.Filter.lt("CLOUD_COVER", max_cloud_cover))
    if apply_mask:
        col = col.map(mask_landsat_clouds)
    return col


def get_modis_collection(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str
) -> ee.ImageCollection:
    """
    Query MODIS 250m surface reflectance collection.
    
    Args:
        aoi: Earth Engine geometry region of interest.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        
    Returns:
        ee.ImageCollection
    """
    return (
        ee.ImageCollection(config.MODIS_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )


def calculate_ndvi(nir: ee.Image, red: ee.Image, name: str = "ndvi") -> ee.Image:
    """
    Calculate Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red).
    
    Args:
        nir: Near-infrared band image.
        red: Red band image.
        name: Name for the resulting NDVI band.
        
    Returns:
        ee.Image with single float band.
    """
    return nir.subtract(red).divide(nir.add(red)).rename(name)
