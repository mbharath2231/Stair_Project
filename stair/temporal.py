"""
STAIR Section 2.2.1: Temporal Linear Interpolation
=================================================
Calculates pixel-wise time-weighted linear interpolation between the nearest 
clear observation BEFORE the target date and the nearest clear observation AFTER.
    L_linear(t0) = R(t1) + [R(t2) - R(t1)] * (t0 - t1) / (t2 - t1)
"""

from typing import Tuple
import ee

def impute_temporal(
    collection: ee.ImageCollection,
    raw_image: ee.Image,
    target_date: str,
    band_name: str = "SR_B5"
) -> Tuple[ee.Image, ee.Image]:
    """
    Apply STAIR time-weighted linear interpolation to fill gaps in target image.
    For every pixel, finds the closest clear observation before and after t0,
    and linearly interpolates the reflectance value.
    
    Args:
        collection: Masked historical image collection.
        raw_image: Cloud-masked target image on target_date.
        target_date: Target date string (YYYY-MM-DD).
        band_name: Band to impute.
        
    Returns:
        Tuple of (L_linear prediction, blended gap_filled_image)
    """
    t0_millis = ee.Date(target_date).millis()
    
    # 1. Add 't' (time in millis) to all images
    def add_time(img):
        t_band = ee.Image.constant(img.date().millis()).rename('t').toFloat()
        return img.addBands(t_band)
        
    col_with_time = collection.map(add_time)
    
    # 2. Split collection into Before and After target date
    col_before = col_with_time.filterDate('2000-01-01', target_date)
    # Add a negative time band to the 'after' collection so we can use qualityMosaic to find the MINIMUM time
    col_after = col_with_time.filterDate(ee.Date(target_date).advance(1, 'day'), '2100-01-01')
    col_after = col_after.map(lambda img: img.addBands(img.select('t').multiply(-1).rename('neg_t')))
    
    # 3. Find nearest clear pixels
    # qualityMosaic('t') gets the max time (most recent clear pixel before t0)
    # Failsafe: if collection is empty, qualityMosaic returns an image with NO bands. 
    # We add a fallback to an empty constant image with the right bands to prevent select() crashes.
    dummy_img = ee.Image.constant(0).rename(band_name).addBands(ee.Image.constant(0).rename('t')).addBands(ee.Image.constant(0).rename('neg_t'))
    
    img_before = ee.Image(ee.Algorithms.If(
        col_before.size().gt(0),
        col_before.qualityMosaic('t'),
        dummy_img
    ))
    
    img_after = ee.Image(ee.Algorithms.If(
        col_after.size().gt(0),
        col_after.qualityMosaic('neg_t'),
        dummy_img
    ))
    
    # Extract values and times
    v_before = img_before.select(band_name)
    t_before = img_before.select('t')
    
    v_after = img_after.select(band_name)
    t_after = img_after.select('t')
    
    # 4. Compute STAIR Temporal Interpolation
    # weight = (t0 - t_before) / (t_after - t_before)
    time_diff = t_after.subtract(t_before)
    # Safeguard against division by zero if somehow t_after == t_before (should be impossible due to dates)
    safe_time_diff = time_diff.where(time_diff.eq(0), 1)
    
    w = ee.Image.constant(t0_millis).subtract(t_before).divide(safe_time_diff)
    
    # L = v_before + w * (v_after - v_before)
    predicted_linear = v_before.add(w.multiply(v_after.subtract(v_before))).rename(f"{band_name}_linear")
    
    # Failsafe: If a pixel has NO observations after t0, just use v_before (forward fill). 
    # If no observations before t0, use v_after (backward fill).
    # If both were empty (meaning dummy_img was used), it just returns 0, but unmasking with t1 baseline later in adaptive.py will save it.
    predicted_linear = predicted_linear.unmask(v_before).unmask(v_after)
    
    # 5. Patch the raw image gaps
    target_band = raw_image.select(band_name)
    patched_image = target_band.unmask(predicted_linear).rename(f"{band_name}_temporal_patched")
    
    return predicted_linear, patched_image
