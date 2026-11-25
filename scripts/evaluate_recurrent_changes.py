#!/usr/bin/env python3
"""Score changes in mean NDVI on common pixels within fixed 150 m tiles."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_spatial_history import MAX_PAIR_DAYS, read_ndvi
from stair.validation import make_holdout

METHODS = ("adaptive", "twenty_full", "twenty_full_spatial_boosting", "twenty_full_cnn_tuned",
           "pixel_lstm", "conv_lstm")
SIDE = 5
MIN_PIXELS = 13


def tile_sum(values):
    rows, cols = (size // SIDE for size in values.shape)
    return values[:rows * SIDE, :cols * SIDE].reshape(rows, SIDE, cols, SIDE).sum(axis=(1, 3))


def score_pattern(directory, grid, pattern, dates):
    width, height = grid["dimensions"]
    holdout = make_holdout((height, width), pattern)
    all_errors = {name: [] for name in METHODS}
    previous = None
    date_pairs = 0
    for row in dates:
        current_day = date.fromisoformat(row["date"])
        images = {name: read_ndvi(directory / row["rasters"][name], grid) for name in ("truth", *METHODS)}
        valid = holdout & np.logical_and.reduce([np.isfinite(a) for a in images.values()])
        if previous is not None:
            earlier_day, earlier, earlier_valid = previous
            shared = valid & earlier_valid
            support = tile_sum(shared)
            use = support >= MIN_PIXELS
            if use.any() and 0 < (current_day - earlier_day).days <= MAX_PAIR_DAYS:
                changes = {name: tile_sum(np.where(shared, array - earlier[name], 0))[use] / support[use]
                           for name, array in images.items()}
                for name in METHODS:
                    all_errors[name].append(changes[name] - changes["truth"])
                date_pairs += 1
        previous = current_day, images, valid
    scores = {}
    for name, partitions in all_errors.items():
        if partitions:
            errors = np.concatenate(partitions)
            scores[name] = {"tile_date_pairs": len(errors), "mae": float(np.abs(errors).mean()),
                            "bias": float(errors.mean())}
    return {"date_pairs": date_pairs, "methods": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "run.json").read_text(encoding="utf-8"))
    summary = {"tile_metres": SIDE * abs(manifest["grid"]["transform"][0]), "minimum_common_pixels": MIN_PIXELS,
               "maximum_pair_days": MAX_PAIR_DAYS, "mask": "Common valid pixels across exported methods and paired dates",
               "patterns": {pattern: score_pattern(args.directory, manifest["grid"], pattern, values["dates"])
                            for pattern, values in manifest["patterns"].items()}}
    path = args.directory / "tile_change.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
