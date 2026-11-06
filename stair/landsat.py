"""Physical-reflectance inputs for the new workflow, separate from DN loaders."""
from __future__ import annotations

DEFAULT_BANDS = ("SR_B5", "SR_B4", "SR_B3")  # NIR, Red, Green (FCC)
SCALE = 0.0000275
OFFSET = -0.2
BAD_QA_BITS = sum(1 << bit for bit in (0, 1, 2, 3, 4, 5))


def prepare_landsat(image, bands=DEFAULT_BANDS):
    """Mask fill/cloud/dilated/cirrus/shadow/snow and saturated/occluded pixels.

    Requested bands share a mask. Negative reflectance is retained; no brightness
    cutoff, water exemption, smoothing, or clipping. QA_RADSAT==0 is conservative.
    """
    import ee
    selected = image.select(list(bands))
    valid = image.select("QA_PIXEL").bitwiseAnd(BAD_QA_BITS).eq(0)
    valid = valid.And(image.select("QA_RADSAT").eq(0))
    valid = valid.And(selected.mask().reduce(ee.Reducer.min()))
    return ee.Image(selected.multiply(SCALE).add(OFFSET).toFloat().updateMask(valid)
            .copyProperties(image, ["system:time_start", "SPACECRAFT_ID", "LANDSAT_PRODUCT_ID"])
            .set("source_id", image.id()))


def landsat_reflectance_collection(region, start_date, end_date, bands=DEFAULT_BANDS):
    """Merge Landsat 8/9 C2 L2, QA-mask, scale, and retain source IDs."""
    import ee
    merged = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
    return (merged.filterBounds(region).filterDate(start_date, end_date)
            .map(lambda image: prepare_landsat(image, bands)).sort("system:time_start"))
