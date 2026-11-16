"""Physical-reflectance inputs for the new workflow, separate from DN loaders."""
from __future__ import annotations

DEFAULT_BANDS = ("SR_B5", "SR_B4", "SR_B3")  # NIR, Red, Green (FCC)
SCALE = 0.0000275
OFFSET = -0.2
BAD_QA_BITS = sum(1 << bit for bit in (0, 1, 2, 3, 4, 5))
OLI_PRODUCTS = ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2")
HISTORICAL_PRODUCTS = ("LANDSAT/LT05/C02/T1_L2", "LANDSAT/LE07/C02/T1_L2")


def prepare_landsat(image, bands=DEFAULT_BANDS, *, source_bands=None):
    """Mask fill/cloud/dilated/cirrus/shadow/snow and saturated/occluded pixels.

    Requested bands share a mask. Negative reflectance is retained; no brightness
    cutoff, water exemption, smoothing, or clipping. QA_RADSAT==0 is conservative.
    """
    import ee
    selected = image.select(list(source_bands or bands), list(bands))
    valid = image.select("QA_PIXEL").bitwiseAnd(BAD_QA_BITS).eq(0)
    valid = valid.And(image.select("QA_RADSAT").eq(0))
    valid = valid.And(selected.mask().reduce(ee.Reducer.min()))
    return ee.Image(selected.multiply(SCALE).add(OFFSET).toFloat().updateMask(valid)
            .copyProperties(image, ["system:time_start", "SPACECRAFT_ID", "LANDSAT_PRODUCT_ID"])
            .set("source_id", image.id()))


def landsat_reflectance_collection(region, start_date, end_date, bands=DEFAULT_BANDS, *, historical=False):
    """Prepare Landsat C2 L2 reflectance, optionally including Landsat 5/7."""
    import ee
    def filtered_pair(products):
        return (ee.ImageCollection(products[0]).merge(ee.ImageCollection(products[1]))
                .filterBounds(region).filterDate(start_date, end_date))

    merged = filtered_pair(OLI_PRODUCTS).map(lambda image: prepare_landsat(image, bands))
    if historical:
        # The output names preserve the existing NIR/Red/Green convention.
        legacy_bands = {"SR_B5": "SR_B4", "SR_B4": "SR_B3", "SR_B3": "SR_B2"}
        source_bands = [legacy_bands[band] for band in bands]
        legacy = filtered_pair(HISTORICAL_PRODUCTS).map(
            lambda image: prepare_landsat(image, bands, source_bands=source_bands))
        merged = merged.merge(legacy)
    return merged.sort("system:time_start")
