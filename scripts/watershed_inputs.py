"""Download reproducible River Raisin inputs on explicit Landsat and MODIS grids."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
from math import ceil, floor
from pathlib import Path
import time
from tempfile import NamedTemporaryFile

from affine import Affine
import ee
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.windows import Window
import requests

from stair.landsat import DEFAULT_BANDS as BANDS, OFFSET, SCALE, prepare_landsat

LANDSAT_PRODUCTS = ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2")
MODIS_PRODUCT = "MODIS/061/MCD43A4"
MODIS_BANDS = (2, 1, 4)
NODATA = -9999.0
TILE = 1024


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: Affine
    width: int
    height: int
    ee_crs: str | None = field(default=None, repr=False)

    @classmethod
    def covering(cls, crs, base, bounds, padding=0, ee_crs=None):
        left, bottom, right, top = bounds
        col, row = ~base * (left, top)
        end_col, end_row = ~base * (right, bottom)
        x, y = floor(col) - padding, floor(row) - padding
        return cls(crs, base * Affine.translation(x, y),
                   ceil(end_col) + padding - x, ceil(end_row) + padding - y, ee_crs)

    @property
    def shape(self):
        return self.height, self.width

    @property
    def profile(self):
        return dict(driver="GTiff", crs=self.crs, transform=self.transform,
                    width=self.width, height=self.height, tiled=True,
                    compress="deflate", nodata=NODATA, dtype="float32")

    def windows(self, step=TILE):
        for y in range(0, self.height, step):
            for x in range(0, self.width, step):
                yield Window(x, y, min(step, self.width - x), min(step, self.height - y))

    def polygon_mask(self, geometry):
        return geometry_mask([geometry], self.shape, self.transform, invert=True)

    def metadata(self):
        return {"crs": self.crs, "transform": list(self.transform)[:6],
                "width": self.width, "height": self.height}


def write_raster(path, grid, values, bands=BANDS, **tags):
    with rasterio.open(path, "w", **grid.profile, count=len(bands)) as dst:
        dst.write(np.where(np.isfinite(values), values, NODATA).astype(np.float32))
        dst.descriptions = tuple(bands)
        dst.update_tags(**tags)


def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read()
    data[data == NODATA] = np.nan
    return data


def download(image, grid, window):
    transform = grid.transform * Affine.translation(window.col_off, window.row_off)
    last_error = None
    for attempt in range(3):
        try:
            url = image.getDownloadURL({
                "crs": grid.ee_crs or grid.crs, "crs_transform": list(transform)[:6],
                "dimensions": [window.width, window.height], "format": "GEO_TIFF",
            })
            response = requests.get(url, timeout=(20, 180))
            if response.status_code == 429 or response.status_code >= 500:
                raise RuntimeError(f"Earth Engine download HTTP {response.status_code}")
            response.raise_for_status()
            with MemoryFile(response.content) as memory, memory.open() as src:
                crs_matches = src.crs == rasterio.crs.CRS.from_string(grid.crs)
                # EE encodes SR-ORG:6974 downloads with a WGS84 sinusoidal tag.
                # The request uses the product's exact native affine and grid ID;
                # local exports use the MODIS sphere specified by NASA.
                if grid.ee_crs == "SR-ORG:6974" and src.crs.to_dict().get("proj") == "sinu":
                    crs_matches = True
                if (not crs_matches
                        or not src.transform.almost_equals(transform, precision=1e-7)
                        or src.shape != (window.height, window.width)):
                    raise ValueError(f"Earth Engine download differs from its requested grid: "
                                     f"{src.crs}, {src.transform}, {src.shape}")
                return src.read().astype(np.float32)
        except (ee.EEException, requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise last_error


def tiled_cache(path, image, grid, inside):
    """Download one three-band day with bounded requests and publish it atomically."""
    if path.is_file():
        return path
    with NamedTemporaryFile(dir=path.parent, suffix=".partial.tif", delete=False) as file:
        temp = Path(file.name)
    image = image.toFloat().unmask(NODATA, sameFootprint=False)

    def fetch(window):
        pixels = window.toslices()
        mask = inside[pixels]
        if not mask.any():
            return window, np.full((len(BANDS), *mask.shape), NODATA, dtype=np.float32)
        data = download(image, grid, window)
        data[:, ~mask] = NODATA
        return window, data

    try:
        with rasterio.open(temp, "w", **grid.profile, count=len(BANDS)) as dst:
            dst.descriptions = tuple(BANDS)
            with ThreadPoolExecutor(max_workers=4) as executor:
                for window, values in executor.map(fetch, grid.windows()):
                    dst.write(values, window=window)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def day_filter(collection, day):
    return collection.filterDate(day, ee.Date(day).advance(1, "day"))


def landsat_collection(geometry, start, end):
    region = ee.Geometry(geometry, proj="EPSG:4326", geodesic=False)
    collection = (ee.ImageCollection(LANDSAT_PRODUCTS[0]).merge(ee.ImageCollection(LANDSAT_PRODUCTS[1]))
                  .filterBounds(region).filterDate(start, end).sort("system:index"))
    scenes = collection.map(lambda img: img.set("scene_day", img.date().format("YYYY-MM-dd")))
    rows = scenes.reduceColumns(ee.Reducer.toList(2), ["scene_day", "system:index"]).get("list").getInfo()
    return scenes, rows


def landsat_day(collection, day, masked):
    images = day_filter(collection, day)
    if masked:
        return images.map(prepare_landsat).sort("source_id").mosaic().select(list(BANDS))
    return images.mosaic().select(list(BANDS)).multiply(SCALE).add(OFFSET)


def read_modis_inputs(grid, collection, days, review_days, inside_buffer):
    """Fetch native-resolution MODIS in small batches, QA before interpolation."""
    result, originals = {}, {}
    for i in range(0, len(days), 16):
        exports, keys = [], []
        for day in days[i:i + 16]:
            image = ee.Image(day_filter(collection, day).first())
            refl = image.select([f"Nadir_Reflectance_Band{band}" for band in MODIS_BANDS], list(BANDS))
            qa = image.select([f"BRDF_Albedo_Band_Mandatory_Quality_Band{band}" for band in MODIS_BANDS])
            good = qa.eq(0).reduce(ee.Reducer.min())
            original = refl.updateMask(refl.gte(0).And(refl.lte(32766)).reduce(ee.Reducer.min())).multiply(.0001)
            exports.append(original.updateMask(good))
            keys.append((result, day))
            if day in review_days:
                exports.append(original)
                keys.append((originals, day))
        stack = download(ee.Image.cat(exports).toFloat().unmask(NODATA, sameFootprint=False), grid,
                         Window(0, 0, grid.width, grid.height))
        stack[(stack == NODATA) | ~inside_buffer[None]] = np.nan
        for (store, day), data in zip(keys, np.split(stack, len(keys))):
            store[day] = data
    return result, originals


def cache_root(base, boundary, grid):
    key = sha256(boundary.wkb + repr(grid).encode()).hexdigest()[:14]
    path = Path(base) / f"landsat_c2l2_v1_{key}"
    path.mkdir(parents=True, exist_ok=True)
    return path
