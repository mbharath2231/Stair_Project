#!/usr/bin/env python3
"""Explain the QA exclusions and fusion gaps in a completed watershed run."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

GAP_LABELS = {
    1: "Target MODIS unavailable",
    2: "Prior anchor pair unavailable",
    3: "Target MODIS and prior pair unavailable",
    4: "Later anchor pair unavailable",
    5: "Target MODIS and later pair unavailable",
    6: "Prior and later pairs unavailable",
    7: "Target MODIS and both pairs unavailable",
}
BYTE_NODATA = 255


def validity(path, grid=None):
    with rasterio.open(path) as src:
        if grid and (src.crs, src.transform, src.shape) != grid:
            raise ValueError(f"Unexpected grid for {path}")
        return (src.read_masks() != 0).all(axis=0)


def write_overlay(path, profile, data, label):
    profile = {**profile, "count": 1, "dtype": "uint8", "nodata": BYTE_NODATA, "compress": "deflate"}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, label)


def analyze(run, output):
    report = json.loads((run / "run.json").read_text(encoding="utf-8"))
    boundary = gpd.read_file(run / "watershed.gpkg")
    first = run / f'{report["days"][0]}_Prediction'
    with rasterio.open(first / "MODIS_01_original.tif") as native:
        native_profile = native.profile
        native_grid = native.crs, native.transform, native.shape
        native_inside = geometry_mask(boundary.to_crs(native.crs).geometry,
                                      native.shape, native.transform, invert=True)
    with rasterio.open(first / "STAIR_09_fusion_support.tif") as fine:
        fine_profile = fine.profile
        fine_grid = fine.crs, fine.transform, fine.shape
        fine_inside = fine.read_masks(1) != 0
    limit = report["temporal"]["landsat_days_per_side"]
    anchor_days = {key: date.fromisoformat(key) for key in report["anchors"]}
    pairs = {key: validity(run / f"{key}_Anchor" / "STAIR_06_anchor_residual.tif", fine_grid)
             for key in anchor_days}

    dates = {}
    for day in report["days"]:
        prediction = run / f"{day}_Prediction"
        original = validity(prediction / "MODIS_01_original.tif", native_grid)
        masked = validity(prediction / "MODIS_02_QA_masked.tif", native_grid)
        restored = validity(prediction / "MODIS_05_adaptive.tif", native_grid)
        rejected = native_inside & original & ~masked
        target_valid = validity(prediction / "MODIS_06_adaptive_aligned_30m.tif", fine_grid)
        actual_fusion = validity(prediction / "STAIR_07_fusion_30m.tif", fine_grid)
        target_date = date.fromisoformat(day)
        prior = np.zeros_like(fine_inside)
        later = np.zeros_like(fine_inside)
        for anchor, anchor_date in anchor_days.items():
            age = (target_date - anchor_date).days
            if 0 < age <= limit:
                prior |= pairs[anchor]
            elif -limit <= age < 0:
                later |= pairs[anchor]
        codes = (~target_valid).astype(np.uint8) | ((~prior).astype(np.uint8) << 1) | ((~later).astype(np.uint8) << 2)
        if not np.array_equal(actual_fusion, fine_inside & (codes == 0)):
            raise ValueError(f"Gap reasons do not reproduce the exported fusion mask on {day}")
        counts = np.bincount(codes[fine_inside], minlength=8).tolist()
        if sum(counts[1:]) != report["dates"][day]["fusion"]["unresolved_pixels"]:
            raise ValueError(f"Unexpected number of unresolved pixels on {day}")

        qa_file = f"{day}_MODIS_QA_exclusions.tif"
        gap_file = f"{day}_fusion_gap_reasons.tif"
        write_overlay(output / qa_file, native_profile,
                      np.where(rejected, 1, BYTE_NODATA).astype(np.uint8),
                      "Original reflectance removed by mandatory QA")
        write_overlay(output / gap_file, fine_profile,
                      np.where(fine_inside & (codes != 0), codes, BYTE_NODATA).astype(np.uint8),
                      "Missing target MODIS (1), prior pair (2), later pair (4)")
        dates[day] = {
            "native_watershed_cells": int(native_inside.sum()),
            "modis_original_valid_cells": int((native_inside & original).sum()),
            "modis_excluded_by_qa_cells": int(rejected.sum()),
            "modis_rejected_cells_restored": int((rejected & restored).sum()),
            "modis_unresolved_native_cells": int((native_inside & ~restored).sum()),
            "valid_fusion_pixels": counts[0],
            "unresolved_fusion_pixels": sum(counts[1:]),
            "gap_pixels_by_reason": {code: counts[code] for code in GAP_LABELS},
            "qa_overlay": qa_file,
            "gap_overlay": gap_file,
        }

    summary = {"codes": GAP_LABELS, "no_data": BYTE_NODATA, "dates": dates}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Completed two-sided watershed fusion run")
    args = parser.parse_args()
    run = args.directory.resolve()
    output = run / "gap_diagnostics"
    if output.exists():
        parser.error(f"Gap diagnostics already exist: {output}")
    with TemporaryDirectory(prefix=".gap_review_", dir=run) as temporary:
        working = Path(temporary) / output.name
        working.mkdir()
        analyze(run, working)
        working.rename(output)
    print(output)


if __name__ == "__main__":
    main()
