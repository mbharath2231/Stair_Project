#!/usr/bin/env python3
"""Download Clayton Landsat scenes and daily MODIS as color-infrared GeoTIFFs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

from affine import Affine
import ee
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import calendar, season_bounds
from scripts.export_stair_stages import CRS, NODATA, SIZE, TRANSFORM
from scripts.run_fusion_week import MODIS_PRODUCT
from stair.landsat import HISTORICAL_PRODUCTS, OLI_PRODUCTS, OFFSET, SCALE

OUTPUT = ROOT / "data" / "clayton_fcc_march_october"
CHANNELS = ("NIR", "Red", "Green")
LEGACY_BANDS = ("SR_B4", "SR_B3", "SR_B2")
OLI_BANDS = ("SR_B5", "SR_B4", "SR_B3")
MODIS_BANDS = tuple(f"Nadir_Reflectance_Band{n}" for n in (2, 1, 4))
MODIS_CRS = "+proj=sinu +R=6371007.181 +units=m +no_defs"


@dataclass(frozen=True)
class Grid:
    ee_crs: str
    crs: str
    transform: Affine
    width: int
    height: int


def modis_grid(region):
    projection = ee.Image(f"{MODIS_PRODUCT}/2005_04_18").select(MODIS_BANDS[0]).projection()
    info = projection.getInfo()
    # The affine-aware Earth Engine projection represents coordinates in pixel space.
    pixels = np.asarray(region.transform(projection, 1).bounds(1, projection).coordinates().getInfo()[0])
    left, top = np.floor(pixels.min(axis=0)).astype(int)
    right, bottom = np.ceil(pixels.max(axis=0)).astype(int)
    offset = Affine(*info["transform"]) * Affine.translation(left, top)
    return Grid(info["crs"], MODIS_CRS, offset, int(right - left), int(bottom - top))


def list_landsat(region, start, end):
    collections = []
    for product in HISTORICAL_PRODUCTS + OLI_PRODUCTS:
        collection = ee.ImageCollection(product).filterBounds(region).filterDate(start, end)
        collections.append(collection.map(lambda image, product=product: image.set(
            "fcc_asset", ee.String(product).cat("/").cat(ee.String(image.get("system:index"))))))
    combined = collections[0]
    for collection in collections[1:]:
        combined = combined.merge(collection)
    rows = combined.reduceColumns(ee.Reducer.toList(2), ["fcc_asset", "system:time_start"]).get("list").getInfo()
    return sorted((datetime.fromtimestamp(ms / 1000, timezone.utc).date().isoformat(), asset) for asset, ms in rows)


def read_batch(images, grid):
    image = ee.Image.cat(images).toFloat().unmask(NODATA, sameFootprint=False)
    request = {"crs": grid.ee_crs, "crs_transform": list(grid.transform)[:6],
               "dimensions": [grid.width, grid.height], "format": "GEO_TIFF"}
    for attempt in range(3):
        try:
            response = requests.get(image.getDownloadURL(request), timeout=(20, 180))
            response.raise_for_status()
            with MemoryFile(response.content) as memory, memory.open() as src:
                if (not src.transform.almost_equals(grid.transform, precision=1e-7)
                        or src.shape != (grid.height, grid.width) or src.count != 3 * len(images)):
                    raise ValueError("Downloaded FCC image does not match the requested grid")
                return src.read().reshape(len(images), 3, grid.height, grid.width)
        except (requests.RequestException, ee.EEException):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def write_batch(folder, sensor, records, grid):
    images = []
    for day, asset in records:
        scene = ee.Image(asset)
        if sensor == "Landsat":
            bands = LEGACY_BANDS if asset.rsplit("/", 1)[0] in HISTORICAL_PRODUCTS else OLI_BANDS
            image = scene.select(list(bands), list(CHANNELS))
            image = image.updateMask(scene.select("QA_PIXEL").bitwiseAnd(1).eq(0)).multiply(SCALE).add(OFFSET)
        else:
            image = scene.select(list(MODIS_BANDS), list(CHANNELS))
            image = image.updateMask(image.gte(0).And(image.lte(32766)).reduce(ee.Reducer.min())).multiply(0.0001)
        images.append(image.updateMask(image.mask().reduce(ee.Reducer.min())))
    arrays = read_batch(images, grid)
    result = []
    for (day, asset), values in zip(records, arrays):
        filename = f"{day}_{asset.rsplit('/', 1)[-1]}_FCC.tif" if sensor == "Landsat" else f"{day}_MCD43A4_FCC.tif"
        with rasterio.open(folder / filename, "w", driver="GTiff", count=3, dtype="float32",
                           crs=grid.crs, transform=grid.transform, width=grid.width, height=grid.height,
                           nodata=NODATA, compress="deflate", predictor=3, interleave="pixel") as dst:
            dst.write(values)
            dst.descriptions = CHANNELS
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            dst.update_tags(date=day, source_asset=asset, units="surface_reflectance",
                            display_channels="NIR,Red,Green")
        result.append({"date": day, "asset": asset, "file": f"{sensor}/{filename}",
                       "valid_pixels": int((values != NODATA).all(axis=0).sum())})
    return result


def collect_year(year, output, region, grids, pool):
    target = output / str(year)
    if target.exists():
        manifest = json.loads((target / "index.json").read_text(encoding="utf-8"))
        if not all((target / row["file"]).is_file() for rows in manifest["images"].values() for row in rows):
            raise ValueError(f"Missing a TIFF listed in {target / 'index.json'}")
        return manifest
    first, last = season_bounds(year, march_october=True)
    start, end = first.isoformat(), last.isoformat()
    landsat = list_landsat(region, start, end)
    indexes = ee.ImageCollection(MODIS_PRODUCT).filterDate(start, end).aggregate_array("system:index").getInfo()
    modis = sorted((index.replace("_", "-"), f"{MODIS_PRODUCT}/{index}") for index in indexes)
    if not landsat or [day for day, _ in modis] != calendar(first, last):
        raise RuntimeError(f"Unexpected source availability in {year}")
    with TemporaryDirectory(prefix=f"{year}_", dir=output) as temporary:
        work = Path(temporary) / str(year)
        work.mkdir()
        jobs = []
        for sensor, rows, size in (("Landsat", landsat, 45), ("MODIS", modis, 60)):
            folder = work / sensor
            folder.mkdir()
            for offset in range(0, len(rows), size):
                jobs.append((sensor, pool.submit(write_batch, folder, sensor, rows[offset:offset + size], grids[sensor])))
        images = {"Landsat": [], "MODIS": []}
        for sensor, job in jobs:
            images[sensor].extend(job.result())
        manifest = {"year": year, "images": images}
        (work / "index.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        work.replace(target)
    print(f"{year}: {len(landsat)} Landsat FCCs and {len(modis)} MODIS FCCs", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--project", default=os.getenv("EE_PROJECT_ID", "stair-project-509517"))
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--end-year must be at least --start-year")
    ee.Initialize(project=args.project)
    x, y = TRANSFORM[2], TRANSFORM[5]
    region = ee.Geometry.Rectangle([x, y - SIZE * 30, x + SIZE * 30, y], proj=CRS, geodesic=False)
    grids = {"Landsat": Grid(CRS, CRS, Affine(*TRANSFORM), SIZE, SIZE), "MODIS": modis_grid(region)}
    args.output.mkdir(parents=True, exist_ok=True)
    years = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        for year in range(args.start_year, args.end_year + 1):
            manifest = collect_year(year, args.output, region, grids, pool)
            years.append({"year": year, "index": f"{year}/index.json",
                          **{sensor: len(rows) for sensor, rows in manifest["images"].items()}})
    inventory = {"channels": CHANNELS, "season": "March 1–October 31", "years": years}
    (args.output / "index.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
