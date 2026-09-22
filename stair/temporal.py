"""
STAIR Section 2.2.1: Temporal Linear Interpolation
=================================================
Calculates pixel-wise linear regression over clear historical observations to
predict reflectance for missing or cloud-obscured pixels on a target date:
    L_linear(p_g, t0) = a * t0 + b
"""

from typing import Tuple, Optional
import ee
import config


def add_time_band(image: ee.Image, band_name: str = "SR_B5") -> ee.Image:
    """
    Add a normalized time band 't' (days since epoch) as the independent variable.
    
    Args:
        image: ee.Image from time series.
        band_name: Target spectral band (dependent variable).
        
    Returns:
        ee.Image with bands ['t', band_name]
    """
    date_days = (
        ee.Image(image.date().millis())
        .divide(1000 * 60 * 60 * 24)
        .toFloat()
        .rename("t")
    )
    return image.select(band_name).addBands(date_days).select(["t", band_name])


def compute_temporal_regression(
    collection: ee.ImageCollection,
    band_name: str = "SR_B5"
) -> Tuple[ee.Image, ee.Image]:
    """
    Fit pixel-wise linear regression across time using ee.Reducer.linearFit().
    
    Args:
        collection: Masked historical image collection.
        band_name: Target band to fit (e.g. 'SR_B5' for NIR, 'SR_B4' for Red).
        
    Returns:
        Tuple of (slope 'a', offset 'b')
    """
    reg_col = collection.map(lambda img: add_time_band(img, band_name))
    fit = reg_col.reduce(ee.Reducer.linearFit())
    slope = fit.select("scale").rename("slope")
    intercept = fit.select("offset").rename("intercept")
    return slope, intercept


def compute_regression_metrics(
    collection: ee.ImageCollection,
    band_name: str = "SR_B5",
    aoi: Optional[ee.Geometry] = None
) -> ee.Image:
    """
    Compute evaluation metrics for the temporal regression:
      - Valid observation count (N_Points)
      - Pearson's correlation coefficient squared (R_Square)
      - Slope and Intercept
      
    Args:
        collection: Masked historical image collection.
        band_name: Target band.
        aoi: Optional region geometry to clip to.
        
    Returns:
        Multi-band ee.Image [Slope, Intercept, R_Square, N_Points]
    """
    reg_col = collection.map(lambda img: add_time_band(img, band_name))
    fit = reg_col.reduce(ee.Reducer.linearFit())
    slope = fit.select("scale").rename("Slope")
    intercept = fit.select("offset").rename("Intercept")
    
    n_points = collection.select(band_name).count().rename("N_Points")
    pearsons = reg_col.reduce(ee.Reducer.pearsonsCorrelation())
    r_square = pearsons.select("correlation").pow(2).rename("R_Square")
    
    eval_matrix = ee.Image([slope, intercept, r_square, n_points]).toDouble()
    if aoi is not None:
        eval_matrix = eval_matrix.clip(aoi)
    return eval_matrix


def compute_temporal_support(
    collection: ee.ImageCollection,
    target_date: str,
    band_name: str = "SR_B5",
    aoi: Optional[ee.Geometry] = None
) -> ee.Image:
    """
    Evaluate temporal support for every pixel relative to target_date:
      - N_Before: Count of clear observations before target date (t < t0)
      - N_After: Count of clear observations after target date (t > t0)
      - N_Total: Total clear observations used in regression
      - Temporal_Type:
          2 = True Interpolation (bracketed: N_Before >= 1 and N_After >= 1)
          1 = Forward Extrapolation (N_Before >= 1 and N_After == 0)
          3 = Backward Extrapolation (N_Before == 0 and N_After >= 1)
          0 = Insufficient data (N_Total < 2)

    Args:
        collection: Masked historical image collection.
        target_date: Date string (YYYY-MM-DD).
        band_name: Target spectral band.
        aoi: Optional boundary geometry.

    Returns:
        Multi-band ee.Image [N_Before, N_After, N_Total, Temporal_Type]
    """
    t0_millis = ee.Date(target_date).millis()
    t0_next_day_millis = ee.Date(target_date).advance(1, "day").millis()

    col_before = collection.filter(ee.Filter.lt("system:time_start", t0_millis))
    col_after = collection.filter(ee.Filter.gte("system:time_start", t0_next_day_millis))

    n_before = col_before.select(band_name).count().rename("N_Before")
    n_after = col_after.select(band_name).count().rename("N_After")
    n_total = collection.select(band_name).count().rename("N_Total")

    temporal_type = (
        ee.Image(0)
        .where(n_before.gte(1).And(n_after.gte(1)), 2)
        .where(n_before.gte(1).And(n_after.eq(0)), 1)
        .where(n_before.eq(0).And(n_after.gte(1)), 3)
        .rename("Temporal_Type")
    )

    support_stack = ee.Image([n_before, n_after, n_total, temporal_type]).toInt()
    if aoi is not None:
        support_stack = support_stack.clip(aoi)
    return support_stack


def impute_temporal(
    raw_image: ee.Image,
    slope: ee.Image,
    intercept: ee.Image,
    target_date: str,
    band_name: str = "SR_B5"
) -> Tuple[ee.Image, ee.Image]:
    """
    Apply linear temporal equation L_linear = a * t0 + b to fill gaps in raw target image.
    
    Args:
        raw_image: Cloud-masked target image on target_date.
        slope: Temporal regression slope (scale).
        intercept: Temporal regression intercept (offset).
        target_date: Target date string (YYYY-MM-DD).
        band_name: Band to impute.
        
    Returns:
        Tuple of (L_linear prediction, blended gap_filled_image)
    """
    target_date_ee = ee.Date(target_date)
    t0_millis = (
        ee.Image(target_date_ee.millis())
        .divide(1000 * 60 * 60 * 24)
        .toFloat()
    )
    predicted_linear = t0_millis.multiply(slope).add(intercept).rename(f"{band_name}_linear")
    
    # Unmask missing pixels with the linear prediction
    target_band = raw_image.select(band_name)
    patched_image = target_band.unmask(predicted_linear).rename(f"{band_name}_temporal_patched")
    
    return predicted_linear, patched_image
