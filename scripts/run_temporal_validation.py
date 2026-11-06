#!/usr/bin/env python3
"""Fetch/cache one real Landsat chip and validate temporal interpolation only."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stair.temporal import interpolate_temporal_numpy
from stair.validation import make_holdout


def fetch_chip(args, cache):
    import ee
    import rasterio
    from rasterio.io import MemoryFile
    from affine import Affine
    from pyproj import Transformer
    from pyproj import CRS
    import requests
    from stair.landsat import DEFAULT_BANDS, landsat_reflectance_collection

    try:
        ee.Initialize(project=args.project)
    except Exception as exc:
        raise RuntimeError("Earth Engine initialization failed. Run .venv\\Scripts\\earthengine.exe authenticate "
                           "and check --project. Original error: " + str(exc)) from exc
    t0 = date.fromisoformat(args.target_date)
    resolution = 30
    size = math.ceil(args.size_m / resolution)
    if size < 10 or size > 512:
        raise ValueError("Use a small chip between 300 m and 15360 m for direct download")
    crs = CRS.from_user_input(args.crs)
    if not crs.is_projected or any(axis.unit_name != "metre" for axis in crs.axis_info):
        raise ValueError("--crs must be a projected CRS measured in meters")
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(args.lon, args.lat)
    xmin = math.floor((x - args.size_m / 2) / resolution) * resolution
    ymax = math.ceil((y + args.size_m / 2) / resolution) * resolution
    transform = [resolution, 0, xmin, 0, -resolution, ymax]
    region = ee.Geometry.Rectangle([xmin, ymax - size * resolution, xmin + size * resolution, ymax],
                                   proj=args.crs, geodesic=False)
    start = (t0 - timedelta(days=args.max_distance_days)).isoformat()
    end = (t0 + timedelta(days=args.max_distance_days + 1)).isoformat()
    bands = list(DEFAULT_BANDS)
    collection = landsat_reflectance_collection(region, start, end, bands)
    ids = collection.aggregate_array("source_id").getInfo()
    timestamps = collection.aggregate_array("system:time_start").getInfo()
    source_dates = [datetime.fromtimestamp(t / 1000, timezone.utc).date().isoformat() for t in timestamps]
    days = sorted(set(source_dates))
    if args.target_date not in days:
        raise ValueError(f"No Landsat acquisition on {args.target_date}; available dates: {days}")
    manifest = {
        "label": "Clayton Township farmland, Michigan" if (args.lon, args.lat) == (-84.2333, 41.8611) else "User-selected land area",
        "target_date": args.target_date, "center_lon_lat": [args.lon, args.lat],
        "requested_size_m": args.size_m, "actual_size_m": size * resolution,
        "crs": args.crs, "transform": transform, "width": size, "height": size,
        "bands": bands, "units": "surface_reflectance", "date_convention": "UTC calendar day",
        "sources": [{"id": i, "date": d} for i, d in zip(ids, source_dates)],
        "daily_mosaic": "source_id ascending; last valid pixel wins (Earth Engine mosaic)",
        "qa_policy": "QA_PIXEL bits 0-5 clear; QA_RADSAT == 0; common mask across selected bands",
        "reflectance_scale": .0000275, "reflectance_offset": -.2,
        "fetch_start": start, "fetch_end_exclusive": end,
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "lineage": "Original QA-masked Landsat C2 L2; no imputed reference rasters",
    }
    arrays = []
    for day in days:
        next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        image = collection.filterDate(day, next_day).sort("source_id").mosaic()
        image = image.select(bands).unmask(-9999).toFloat()
        for attempt in range(3):
            try:
                url = image.getDownloadURL({"crs": args.crs, "crs_transform": transform,
                                            "dimensions": [size, size], "format": "GEO_TIFF"})
                response = requests.get(url, timeout=(20, 120))
                response.raise_for_status()
                with MemoryFile(response.content) as memory, memory.open() as src:
                    if src.crs != rasterio.crs.CRS.from_string(args.crs) or src.transform != Affine(*transform):
                        raise ValueError("Downloaded scene grid does not match the requested grid")
                    if (src.width, src.height, src.count) != (size, size, len(bands)):
                        raise ValueError("Unexpected downloaded scene dimensions/bands")
                    values = src.read().astype(float)
                values[values == -9999] = np.nan
                arrays.append(values)
                print(f"{day}: {np.all(np.isfinite(values), axis=0).mean():.1%} QA-clear", flush=True)
                break
            except (requests.RequestException, ee.EEException):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    stack = np.stack(arrays)
    manifest["qa_clear_fraction_by_date"] = dict(zip(days, np.all(np.isfinite(stack), axis=1).mean(axis=(1, 2)).tolist()))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as file:
        np.savez_compressed(file, observations=stack, dates=np.array(days), manifest=json.dumps(manifest))
    print(f"Cached original observations: {cache}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Download original observations using authenticated Earth Engine")
    parser.add_argument("--input", type=Path, default=ROOT / "data/temporal/clayton_20220722.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/temporal_validation/clayton_20220722")
    parser.add_argument("--target-date", default="2022-07-22")
    parser.add_argument("--max-distance-days", type=int, default=32)
    parser.add_argument("--allow-one-sided", action="store_true")
    parser.add_argument("--pattern", choices=["cloud", "stripes", "scene"], default="cloud")
    parser.add_argument("--project", default=os.getenv("EE_PROJECT_ID", "stair-project-509517"))
    parser.add_argument("--lon", type=float, default=-84.2333)
    parser.add_argument("--lat", type=float, default=41.8611)
    parser.add_argument("--size-m", type=float, default=5000)
    parser.add_argument("--crs", default="EPSG:32616", help="Projected CRS with meter units")
    args = parser.parse_args()
    if args.max_distance_days < 1:
        parser.error("--max-distance-days must be positive")
    if args.fetch:
        fetch_chip(args, args.input)
    if not args.input.exists():
        parser.error("Input cache is missing. Authenticate Earth Engine, then run with --fetch.")
    with np.load(args.input, allow_pickle=False) as cache:
        values, dates = cache["observations"], cache["dates"].tolist()
        metadata = json.loads(str(cache["manifest"]))
    if metadata.get("units") != "surface_reflectance":
        raise ValueError("The cache must contain physical surface reflectance")
    if values.ndim != 4 or values.shape[1] != len(metadata["bands"]):
        raise ValueError("Expected observations in (time, band, y, x) order")
    if metadata["bands"] != ["SR_B5", "SR_B4", "SR_B3"]:
        raise ValueError("This initial visual experiment requires NIR, Red, Green in that order")
    if args.target_date not in dates:
        raise ValueError("Requested target date is absent from cache")
    target_index = dates.index(args.target_date)
    truth = values[target_index].copy()
    holdout = make_holdout(truth.shape[1:], args.pattern)
    observed = np.where(holdout[None], np.nan, truth)
    # Explicitly remove target from the reference stack before the engine call.
    reference = np.array([d != args.target_date for d in dates])
    result = interpolate_temporal_numpy(values[reference], list(np.array(dates)[reference]),
        args.target_date, observed, max_distance_days=args.max_distance_days, allow_one_sided=args.allow_one_sided)
    metadata.update(target_date=args.target_date, max_distance_days=args.max_distance_days,
        allow_one_sided=args.allow_one_sided, holdout_pattern=args.pattern,
        evaluation_kind="Artificial gaps in a real QA-screened Landsat observation",
        reference_dates=list(np.array(dates)[reference]), excluded_target_day=args.target_date,
        cache_path=str(args.input.resolve()), cache_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        code_files_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                          ["stair/temporal.py", "stair/landsat.py", "stair/validation.py", "stair/temporal_report.py"]})
    metadata["software_versions"] = {name: importlib.metadata.version(name) for name in
                                     ("numpy", "rasterio", "matplotlib", "earthengine-api", "pyproj")}
    from stair.temporal_report import write_temporal_report
    metrics = write_temporal_report(truth, observed, result, holdout, metadata, args.output,
                                    observations=values, observation_dates=dates)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Visual report: {(args.output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
