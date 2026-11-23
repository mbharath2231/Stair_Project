#!/usr/bin/env python3
"""Recover Landsat sensor provenance for mixed-mission Clayton daily mosaics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import BANDS, SPRING_FALL_CACHE
from scripts.export_stair_stages import download
from scripts.run_daily_ndvi import read_cache
from stair.landsat import HISTORICAL_PRODUCTS, OLI_PRODUCTS, prepare_landsat
from stair.landsat_harmonization import LEGACY, mission, scene_sources, sensor_families

DEFAULT_PROVENANCE = SPRING_FALL_CACHE / "landsat_mixed_provenance.npz"
PRODUCTS = {name.split("/")[1]: name for name in (*HISTORICAL_PRODUCTS, *OLI_PRODUCTS)}


def load_provenance(path):
    with np.load(path, allow_pickle=False) as cache:
        return dict(zip(cache["dates"].tolist(), zip(cache["legacy"], cache["reflectance"], strict=True), strict=True))


def mosaic(ee, ids):
    images = []
    for sort_id in ids:
        code = mission(sort_id)
        scene_id = re.search(rf"{code}_\d{{6}}_\d{{8}}$", sort_id)
        if scene_id is None:
            raise ValueError(f"Unexpected archive scene key {sort_id}")
        scene = ee.Image(f"{PRODUCTS[code]}/{scene_id[0]}")
        source_bands = ("SR_B4", "SR_B3") if code in LEGACY else BANDS
        prepared = prepare_landsat(scene, BANDS, source_bands=source_bands)
        legacy = prepared.select(0).multiply(0).add(int(code in LEGACY)).rename("legacy")
        images.append(prepared.addBands(legacy).set("archive_sort", sort_id))
    return ee.ImageCollection.fromImages(images).sort("archive_sort").mosaic()


def collect(cache, output, project):
    import ee

    records = {}
    for year in range(2005, 2025):
        path = cache / f"clayton_{year}.npz"
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(archive["manifest"].item())
        mixed = {day: ids for day, ids in scene_sources(manifest).items() if len(sensor_families(ids)) > 1}
        if mixed:
            _, images, _ = read_cache(path, year)
            records.update({day: (ids, images[day]) for day, ids in mixed.items()})
    if not records:
        raise ValueError("No mixed-mission dates in the input archive")
    ee.Initialize(project=project)
    dates = sorted(records)
    arrays = download(ee.Image.cat([mosaic(ee, records[day][0]) for day in dates]))
    reflectance, masks = [], []
    for day, image in zip(dates, np.split(arrays, len(dates)), strict=True):
        native = image[:2].astype(np.float32)
        np.testing.assert_array_equal(native, records[day][1], err_msg=f"Mosaic mismatch on {day}")
        valid = np.isfinite(native).all(axis=0)
        if not np.isin(image[2, valid], [0, 1]).all() or np.isfinite(image[2, ~valid]).any():
            raise ValueError(f"Invalid sensor provenance mask on {day}")
        reflectance.append(native)
        masks.append(image[2] == 1)
        print(f"{day}: original mosaic verified, {int(masks[-1].sum()):,} legacy pixels", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        np.savez_compressed(staged, dates=dates, legacy=masks, reflectance=reflectance)
        staged.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=SPRING_FALL_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--project", default=os.getenv("EE_PROJECT_ID", "stair-project-509517"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path")
    collect(args.cache, args.output, args.project)


if __name__ == "__main__":
    main()
