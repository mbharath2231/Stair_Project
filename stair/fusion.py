"""
STAIR Spatiotemporal Data Fusion: Landsat 8 and MODIS
=====================================================
Learns spatial regression relationships between high-resolution (30m) Landsat
and high-frequency (daily, 250m) MODIS surface reflectance on a concurrent
clear day ("Golden Day"), then synthesizes 30m resolution reflectance and NDVI
for dates when Landsat is unavailable or obscured by clouds.
"""

from typing import Tuple, Dict
import ee
import config
from .core import calculate_ndvi


def learn_spatial_rules(
    golden_landsat: ee.Image,
    golden_modis: ee.Image,
    kernel_radius: int = config.DEFAULT_KERNEL_RADIUS,
    bands: list = ["SR_B5", "SR_B4", "SR_B3"]
) -> Dict[str, Dict[str, ee.Image]]:
    """
    Learn the spatial relationship between Landsat and MODIS on the Golden Day.
    Instead of unstable local linear regression, we use the robust difference method.
    We just need to store the Golden Landsat and Golden MODIS for the synthesis phase!
    """
    rules = {}
    
    for band in bands:
        # Resample Golden MODIS to 30m
        modis_30m = golden_modis.select(band).resample('bilinear').reproject(
            crs=golden_landsat.projection(),
            scale=30.0
        )
        
        rules[band] = {
            "L_golden": golden_landsat.select(band),
            "M_golden": modis_30m
        }
        
    return rules


def synthesize_missing_day(
    target_modis: ee.Image,
    rules: Dict[str, Dict[str, ee.Image]],
    aoi: ee.Geometry
) -> ee.Image:
    """
    Synthesize the high-resolution prediction using the robust formula:
    L_target = L_golden + (M_target - M_golden)
    
    This perfectly preserves the sharp 30m spatial details of L_golden, while
    updating the color/phenology using the difference between the MODIS dates.
    """
    synthetic_bands = []
    
    for band, params in rules.items():
        # Resample Target MODIS to 30m
        M_target = target_modis.select(band).resample('bilinear').reproject(
            crs=params["L_golden"].projection(),
            scale=30.0
        )
        
        L_golden = params["L_golden"]
        M_golden = params["M_golden"]
        
        # Calculate Delta: M_target - M_golden (This is a blurry 30m image of the change)
        delta = M_target.subtract(M_golden)
        
        # Apply Delta to Golden Landsat: L_target = L_golden + delta
        synth = L_golden.add(delta).rename(band)
        
        synthetic_bands.append(synth)
        
    return ee.Image(synthetic_bands).clip(aoi)


def gap_fill_multiband(collection: ee.ImageCollection, target_img: ee.Image, target_date: str, bands: list) -> ee.Image:
    """Helper to run temporal gap filling across multiple bands."""
    from .temporal import impute_temporal
    filled_bands = []
    for b in bands:
        _, patched = impute_temporal(collection, target_img, target_date, band_name=b)
        filled_bands.append(patched.rename(b)) # Keep the original band name
    return ee.Image(filled_bands)


def fuse_date(
    prediction_date: str,
    golden_date: str, # Need the golden date string to temporally interpolate golden MODIS
    golden_landsat: ee.Image,
    golden_modis: ee.Image,
    aoi: ee.Geometry,
    kernel_radius: int = config.DEFAULT_KERNEL_RADIUS,
    bands: list = ["SR_B5", "SR_B4", "SR_B3"]
) -> ee.Image:
    """
    Convenience end-to-end function to learn rules and synthesize high-res for a target date.
    Note: Temporally gap-fills MODIS data before spatial fusion!
    """
    from .core import get_modis_collection
    
    # 0. Get the full MODIS collection for the summer (needed for temporal interpolation)
    full_modis_col = get_modis_collection(aoi, "2021-05-01", "2021-10-31", apply_mask=True)
    
    # 1. Temporally gap-fill the Golden Day MODIS (e.g. July 26)
    golden_modis_filled = gap_fill_multiband(full_modis_col, golden_modis, golden_date, bands)
    
    # 2. Learn the spatial rules using the gap-free Golden MODIS and Golden Landsat
    rules = learn_spatial_rules(golden_landsat, golden_modis_filled, kernel_radius=kernel_radius, bands=bands)
    
    # 3. Temporally gap-fill the target date MODIS
    pred_start = ee.Date(prediction_date)
    pred_end = pred_start.advance(1, "day")
    
    target_modis_col = get_modis_collection(aoi, prediction_date, pred_start.advance(1, 'day').format('YYYY-MM-dd').getInfo(), apply_mask=True)
    target_modis_raw = target_modis_col.first()
    
    target_modis_filled = gap_fill_multiband(full_modis_col, target_modis_raw, prediction_date, bands)
    
    # 4. Apply the math to synthesize 30m Landsat!
    return synthesize_missing_day(target_modis_filled, rules, aoi)
