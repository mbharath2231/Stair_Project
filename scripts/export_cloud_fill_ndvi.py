#!/usr/bin/env python3
"""Derive NDVI and withheld-pixel errors from the Clayton stage GeoTIFFs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stair.ndvi import ndvi
from stair.validation import make_holdout, score_holdout

NODATA = -9999.0
ERROR_STAGES = {
    "03_temporal": "Temporal error",
    "04_global": "Global error",
    "05_adaptive": "Adaptive error",
}


def read_stage(path, grid, bands):
    with rasterio.open(path) as source:
        if (source.crs != rasterio.crs.CRS.from_string(grid["crs"])
                or source.transform != Affine(*grid["transform"])
                or [source.width, source.height] != grid["dimensions"]
                or source.descriptions != tuple(bands)):
            raise ValueError(f"Unexpected grid or band order: {path}")
        return source.read(masked=True).astype(np.float64).filled(np.nan), source.profile


def write_index(path, values, profile, label, day):
    profile = {**profile, "count": 1, "dtype": "float32", "nodata": NODATA}
    with rasterio.open(path, "w", **profile) as output:
        output.write(np.where(np.isfinite(values), values, NODATA).astype(np.float32), 1)
        output.set_band_description(1, label)
        output.update_tags(target_date=day, units="unitless")


def export(source, output, pattern):
    run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if pattern not in run["patterns"] or not run.get("stage_tiffs"):
        raise ValueError("Stage GeoTIFFs are required. Run cloud density validation with --export-stages first.")
    bands = run["reflectance_source_bands"]
    if bands[:2] != ["SR_B5", "SR_B4"]:
        raise ValueError("Expected NIR and red in the first two surface reflectance bands")
    report = {
        "grid": run["grid"],
        "patterns": [pattern],
        "stage_tiffs": run["stage_tiffs"],
        "error_tiffs": {stem + "_error": label for stem, label in ERROR_STAGES.items()},
        "ndvi_formula": "(NIR - red) / (NIR + red) using each stage's surface reflectance",
        "ndvi_validity": "finite, nonnegative NIR and red with positive sum",
        "error_definition": "restored minus original NDVI, inside the mask on common valid support",
        "ndvi_display_limits": [-0.2, 1.0],
        "error_display_limit": 0.2,
        "dates": {},
    }
    for day, info in run["dates"].items():
        source_dir = source / pattern / day
        target = output / pattern / day
        target.mkdir(parents=True)
        indices = {}
        for stem in report["stage_tiffs"]:
            values, profile = read_stage(source_dir / f"{stem}.tif", run["grid"], bands)
            index = ndvi(values)[0]
            indices[stem] = index
            write_index(target / f"{stem}.tif", index, profile, "NDVI", day)

        original = indices["01_original"]
        held = make_holdout(original.shape, pattern)
        np.testing.assert_array_equal(indices["02_cloud_masked"], np.where(held, np.nan, original))
        common = held & np.isfinite(original)
        for stem in ERROR_STAGES:
            np.testing.assert_array_equal(indices[stem][~held], original[~held])
            common &= np.isfinite(indices[stem])

        scores = {}
        for stem in ERROR_STAGES:
            errors = np.where(common, indices[stem] - original, np.nan)
            write_index(target / f"{stem}_error.tif", errors, profile, "Restored minus original NDVI", day)
            scores[stem] = score_holdout(original[None], indices[stem][None], common, ["NDVI"])["bands"]["NDVI"]
        report["dates"][day] = {
            "reference": info["reference"],
            "cloud_mask_pixels": int(held.sum()),
            "common_scored_pixels": int(common.sum()),
            "original_ndvi_median": float(np.nanmedian(original)),
            "paired_scores": scores,
        }

    (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed reflectance stage export")
    parser.add_argument("output", type=Path)
    parser.add_argument("--pattern", choices=("cloud", "stripes"), default="cloud")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Choose a fresh output directory: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix=".ndvi_") as temporary:
        staging = Path(temporary) / args.output.name
        staging.mkdir()
        export(args.source, staging, args.pattern)
        staging.rename(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
