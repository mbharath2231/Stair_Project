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
    kernel_radius: int = config.DEFAULT_KERNEL_RADIUS
) -> Dict[str, ee.Image]:
    """
    Execute moving-window spatial linear regression between MODIS and Landsat
    on a concurrent clear day.
    
    Args:
        golden_landsat: Landsat 8 image on clear reference day.
        golden_modis: MODIS surface reflectance image on exact same day.
        kernel_radius: Radius of square moving window kernel in pixels.
        
    Returns:
        Dictionary with keys: 'red_alpha', 'red_beta', 'nir_alpha', 'nir_beta'.
    """
    landsat_red = golden_landsat.select(config.LANDSAT_BANDS["red"]).rename("landsat_red")
    landsat_nir = golden_landsat.select(config.LANDSAT_BANDS["nir"]).rename("landsat_nir")
    
    modis_red = golden_modis.select(config.MODIS_BANDS["red"]).rename("modis_red")
    modis_nir = golden_modis.select(config.MODIS_BANDS["nir"]).rename("modis_nir")
    
    # Stacks: [X (MODIS), Y (Landsat)]
    red_stack = modis_red.addBands(landsat_red)
    nir_stack = modis_nir.addBands(landsat_nir)
    
    kernel = ee.Kernel.square(radius=kernel_radius, units="pixels")
    
    red_weights = red_stack.reduceNeighborhood(reducer=ee.Reducer.linearFit(), kernel=kernel)
    nir_weights = nir_stack.reduceNeighborhood(reducer=ee.Reducer.linearFit(), kernel=kernel)
    
    return {
        "red_alpha": red_weights.select("scale").rename("red_alpha"),
        "red_beta": red_weights.select("offset").rename("red_beta"),
        "nir_alpha": nir_weights.select("scale").rename("nir_alpha"),
        "nir_beta": nir_weights.select("offset").rename("nir_beta"),
    }


def synthesize_missing_day(
    target_modis: ee.Image,
    rules: Dict[str, ee.Image],
    aoi: ee.Geometry
) -> ee.Image:
    """
    Synthesize high-resolution (30m) Red, NIR, and NDVI from a target MODIS image
    using previously learned spatial regression rules.
    
    Args:
        target_modis: MODIS image for the prediction day.
        rules: Dictionary of alphas and betas from learn_spatial_rules().
        aoi: Bounding geometry to clip the resulting image.
        
    Returns:
        ee.Image with bands ['synthetic_red', 'synthetic_nir', 'synthetic_ndvi'].
    """
    modis_red = target_modis.select(config.MODIS_BANDS["red"])
    modis_nir = target_modis.select(config.MODIS_BANDS["nir"])
    
    synth_red = (
        modis_red.multiply(rules["red_alpha"])
        .add(rules["red_beta"])
        .rename("synthetic_red")
    )
    synth_nir = (
        modis_nir.multiply(rules["nir_alpha"])
        .add(rules["nir_beta"])
        .rename("synthetic_nir")
    )
    
    synth_ndvi = calculate_ndvi(synth_nir, synth_red, name="synthetic_ndvi")
    
    return ee.Image([synth_red, synth_nir, synth_ndvi]).clip(aoi)


def fuse_date(
    prediction_date: str,
    golden_landsat: ee.Image,
    golden_modis: ee.Image,
    aoi: ee.Geometry,
    kernel_radius: int = config.DEFAULT_KERNEL_RADIUS
) -> ee.Image:
    """
    Convenience end-to-end function to learn rules and synthesize high-res NDVI for a target date.
    
    Args:
        prediction_date: Target date string (YYYY-MM-DD).
        golden_landsat: Landsat 8 image on clear reference day.
        golden_modis: MODIS image on clear reference day.
        aoi: Area of interest.
        kernel_radius: Moving window size in pixels.
        
    Returns:
        Synthesized multi-band ee.Image clipped to aoi.
    """
    rules = learn_spatial_rules(golden_landsat, golden_modis, kernel_radius=kernel_radius)
    
    pred_start = ee.Date(prediction_date)
    pred_end = pred_start.advance(1, "day")
    
    modis_col = ee.ImageCollection(config.MODIS_COLLECTION).filterBounds(aoi).filterDate(pred_start, pred_end)
    target_modis = modis_col.first()
    
    return synthesize_missing_day(target_modis, rules, aoi)
