#!/usr/bin/env python3
"""Export the cached Clayton Red and NIR inputs as annual GeoTIFFs for QGIS."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from affine import Affine
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import GRID, SPRING_FALL_CACHE, calendar, season_bounds
from scripts.export_stair_stages import NODATA
from scripts.run_daily_ndvi import DEFAULT_OUTPUT, read_cache


def write_stack(path, observations, dates, channel, profile, tags):
    with NamedTemporaryFile(dir=path.parent, suffix=".tif", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        missing = np.full((profile["height"], profile["width"]), NODATA, dtype=np.float32)
        with rasterio.open(temporary, "w", **profile) as dst:
            dst.update_tags(**tags, units="surface_reflectance", date_order="UTC calendar days")
            for band, day in enumerate(dates, start=1):
                image = observations.get(day)
                values = missing if image is None else np.where(np.isfinite(image[channel]), image[channel], NODATA)
                dst.write(values, band)
                dst.set_band_description(band, day)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_year(year, cache, output):
    start, end = season_bounds(year, march_october=True)
    dates = calendar(start, end)
    manifest, landsat, modis = read_cache(cache / f"clayton_{year}.npz", year)
    if manifest["start"] > dates[0] or manifest["end_exclusive"] < end.isoformat():
        raise ValueError(f"Input archive lacks the requested season for {year}")
    width, height = GRID["dimensions"]
    profile = dict(driver="GTiff", width=width, height=height, count=len(dates), dtype="float32",
                   crs=GRID["crs"], transform=Affine(*GRID["transform"]), compress="deflate",
                   predictor=3, interleave="band", nodata=NODATA)
    folder = output / str(year)
    folder.mkdir(parents=True, exist_ok=True)
    for sensor, values, bands in (
        ("Landsat", landsat, ("Landsat 5/7 SR_B4; Landsat 8/9 SR_B5", "Landsat 5/7 SR_B3; Landsat 8/9 SR_B4")),
        ("MODIS", modis, ("MCD43A4 Nadir_Reflectance_Band2", "MCD43A4 Nadir_Reflectance_Band1")),
    ):
        for channel, name in enumerate(("NIR", "Red")):
            tags = {"sensor": sensor, "spectral_band": name, "input_band": bands[channel],
                    "sampling": "Landsat grid" if sensor == "Landsat" else "500 m product resampled onto 30 m grid"}
            write_stack(folder / f"{sensor}_{name}_{year}.tif", values, dates, channel, profile, tags)
    print(f"{year}: Landsat and MODIS Red and NIR exported", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--cache", type=Path, default=SPRING_FALL_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "march_october" / "reflectance")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--end-year must be at least --start-year")
    for year in range(args.start_year, args.end_year + 1):
        export_year(year, args.cache, args.output)


if __name__ == "__main__":
    main()
