#!/usr/bin/env python3
"""Cache quality-screened Landsat and daily MODIS NIR/Red on the Clayton grid."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import ee
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_stair_stages import CRS, SIZE, TRANSFORM, WINDOW, download
from scripts.run_fusion_week import MODIS_PRODUCT, prepare_modis
from stair.landsat import HISTORICAL_PRODUCTS, OLI_PRODUCTS, landsat_reflectance_collection

BANDS = ["SR_B5", "SR_B4"]
CACHE = ROOT / "data" / "ndvi_inputs"
SPRING_FALL_CACHE = CACHE / "march_october"
GRID = {"crs": CRS, "transform": TRANSFORM, "dimensions": [SIZE, SIZE]}


def calendar(start, end):
    return [(start + timedelta(days=i)).isoformat() for i in range((end - start).days)]


def download_stack(images):
    for attempt in range(3):
        try:
            return download(ee.Image.cat(images)).astype(np.float32).reshape(len(images), len(BANDS), SIZE, SIZE)
        except (ee.EEException, requests.RequestException, RuntimeError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def season_bounds(year, march_october=False):
    if march_october:
        return date(year, 3, 1), date(year, 11, 1)
    return date(year, 1, 1), date(year + 1, 1, 1)


def collect_year(year, destination, *, march_october=False):
    season_start, season_end = season_bounds(year, march_october)
    start = season_start - timedelta(days=WINDOW)
    end = season_end + timedelta(days=WINDOW)
    x, y = TRANSFORM[2], TRANSFORM[5]
    region = ee.Geometry.Rectangle([x, y - 30 * SIZE, x + 30 * SIZE, y], proj=CRS, geodesic=False)
    landsat = landsat_reflectance_collection(region, start.isoformat(), end.isoformat(), BANDS,
                                             historical=march_october)
    rows = landsat.reduceColumns(ee.Reducer.toList(2), ["source_id", "system:time_start"]).get("list").getInfo()
    scenes = [(scene, datetime.fromtimestamp(ms / 1000, timezone.utc).date().isoformat()) for scene, ms in rows]
    landsat_dates = sorted({day for _, day in scenes})
    landsat_images = [
        landsat.filterDate(day, ee.Date(day).advance(1, "day")).sort("source_id").mosaic()
        for day in landsat_dates
    ]
    coarse = ee.ImageCollection(MODIS_PRODUCT).filterDate(start.isoformat(), end.isoformat())
    modis_dates = sorted(day.replace("_", "-") for day in coarse.aggregate_array("system:index").getInfo())
    modis_images = [
        prepare_modis(ee.Image(f"{MODIS_PRODUCT}/{day.replace('-', '_')}"), BANDS) for day in modis_dates
    ]
    # One batch is below the Earth Engine interactive TIFF size limit.
    batch_size = 60
    stacks = [landsat_images[i:i + batch_size] for i in range(0, len(landsat_images), batch_size)]
    landsat_batches = len(stacks)
    stacks.extend(modis_images[i:i + batch_size] for i in range(0, len(modis_images), batch_size))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download_stack, stacks))
    empty = np.empty((0, len(BANDS), SIZE, SIZE), dtype=np.float32)
    landsat_values = np.concatenate(results[:landsat_batches]) if landsat_batches else empty
    modis_values = np.concatenate(results[landsat_batches:]) if modis_dates else empty
    metadata = {
        "year": year, "grid": GRID, "bands": BANDS, "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "season_start": season_start.isoformat(), "season_end_exclusive": season_end.isoformat(),
        "landsat_scenes": scenes, "modis_product": MODIS_PRODUCT,
        "landsat_products": list(HISTORICAL_PRODUCTS + OLI_PRODUCTS if march_october else OLI_PRODUCTS),
        "landsat_spectral_harmonization": "None; archived as mission-native surface reflectance",
        "landsat_qa": "QA_PIXEL bits 0–5 clear and QA_RADSAT zero; common NIR and Red support",
        "modis_qa": "full BRDF inversion in NIR and Red before bilinear resampling",
    }
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as stream:
        partial = Path(stream.name)
    try:
        np.savez_compressed(partial, landsat_dates=landsat_dates, landsat=landsat_values,
                            modis_dates=modis_dates, modis=modis_values, manifest=json.dumps(metadata))
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    print(f"{year} cached {len(landsat_dates)} Landsat and {len(modis_dates)} MODIS dates", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--march-october", action="store_true",
                        help="Download March–October with interpolation padding and include Landsat 5/7")
    parser.add_argument("--project", default=os.getenv("EE_PROJECT_ID", "stair-project-509517"))
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--end-year must be at least --start-year")
    if args.start_year < 2013 and not args.march_october:
        parser.error("Use --march-october for Landsat 5/7 input before 2013")
    ee.Initialize(project=args.project)
    cache = args.cache or (SPRING_FALL_CACHE if args.march_october else CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    for year in range(args.start_year, args.end_year + 1):
        path = cache / f"clayton_{year}.npz"
        if path.is_file():
            if args.march_october:
                with np.load(path, allow_pickle=False) as existing:
                    manifest = json.loads(existing["manifest"].item())
                start, end = season_bounds(year, True)
                if (manifest.get("season_start"), manifest.get("season_end_exclusive"),
                    manifest.get("landsat_products")) != (
                    start.isoformat(), end.isoformat(), list(HISTORICAL_PRODUCTS + OLI_PRODUCTS)
                ):
                    raise ValueError(f"{path} is not the March–October Landsat 5/7/8/9 cache")
            print(f"{year} cache exists at {path}", flush=True)
        else:
            collect_year(year, path, march_october=args.march_october)


if __name__ == "__main__":
    main()
