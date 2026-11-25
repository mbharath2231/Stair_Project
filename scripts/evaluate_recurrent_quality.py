#!/usr/bin/env python3
"""Evaluate local structure, sharp NDVI transitions and the direction of change."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_recurrent_changes import METHODS, MIN_PIXELS, SIDE, tile_sum
from scripts.evaluate_spatial_history import CHANGE_THRESHOLD, MAX_PAIR_DAYS, WINDOW, read_ndvi, ssim_windows
from stair.validation import make_holdout

SPATIAL = ROOT / "outputs/spatial_validation"
EDGE_THRESHOLDS = (0.05, 0.10, 0.15)
PRIMARY_EDGE = "0.10"
COMPARATOR = "twenty_full_spatial_boosting"
NEIGHBORS = ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:]))
NDVI_ROUNDING = 2 * np.finfo(np.float32).eps


def differences(image, pairs):
    return np.concatenate([np.abs(image[a] - image[b])[valid]
                           for (a, b), valid in zip(NEIGHBORS, pairs, strict=True)])


def edge_counts(reference, predicted, threshold):
    actual = reference >= threshold
    marked = predicted >= threshold
    return [int((actual & marked).sum()), int((~actual & marked).sum()), int((actual & ~marked).sum())]


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def edge_metrics(counts):
    tp, fp, fn = map(int, counts)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": safe_ratio(tp, tp + fp), "recall": safe_ratio(tp, tp + fn),
            "f1": safe_ratio(2 * tp, 2 * tp + fp + fn)}


def change_signs(previous, current, shared):
    counts = tile_sum(shared)
    use = counts >= MIN_PIXELS
    changes = {name: tile_sum(np.where(shared, newest - previous[name], 0))[use] / counts[use]
               for name, newest in current.items()}
    substantial = np.abs(changes["truth"]) >= CHANGE_THRESHOLD
    return int(substantial.sum()), {
        name: int((np.sign(changes[name][substantial]) == np.sign(changes["truth"][substantial])).sum())
        for name in METHODS
    }


def date_metrics(day, images, common):
    windows = sliding_window_view(common, (WINDOW, WINDOW)).all(axis=(-2, -1))
    actual = sliding_window_view(images["truth"], (WINDOW, WINDOW))[windows].reshape(-1, WINDOW ** 2)
    pairs = [common[a] & common[b] for a, b in NEIGHBORS]
    actual_edges = differences(images["truth"], pairs)
    row = {
        "date": day.isoformat(), "common_pixels": int(common.sum()), "ssim_windows": len(actual),
        "adjacent_pairs": len(actual_edges), "qualifying_change_tiles": 0, "methods": {},
    }
    for name in METHODS:
        predicted = sliding_window_view(images[name], (WINDOW, WINDOW))[windows].reshape(-1, WINDOW ** 2)
        edges = differences(images[name], pairs)
        row["methods"][name] = {
            "ssim_sum": float(ssim_windows(actual, predicted).sum()),
            "edge_counts": {f"{threshold:.2f}": edge_counts(actual_edges, edges, threshold)
                            for threshold in EDGE_THRESHOLDS},
            "correct_change_tiles": 0,
        }
    return row


def monthly_intervals(rows):
    months = sorted({row["date"][:7] for row in rows})
    draws = np.random.default_rng(20261118).integers(len(months), size=(5000, len(months)))
    month_scores = {}
    for name in METHODS:
        blocks = np.zeros((len(months), 7))
        for row in rows:
            values = row["methods"][name]
            blocks[months.index(row["date"][:7])] += (
                values["ssim_sum"], row["ssim_windows"], *values["edge_counts"][PRIMARY_EDGE],
                values["correct_change_tiles"], row["qualifying_change_tiles"],
            )
        ssim, windows, tp, fp, fn, correct, total = blocks[draws].sum(axis=1).T
        with np.errstate(divide="ignore", invalid="ignore"):
            month_scores[name] = np.array([ssim / windows, 2 * tp / (2 * tp + fp + fn), correct / total])
    intervals = {}
    for name in METHODS:
        if name == COMPARATOR:
            continue
        differences = month_scores[name] - month_scores[COMPARATOR]
        intervals[name] = {
            metric: np.quantile(values[np.isfinite(values)], [0.025, 0.975]).tolist()
            if np.isfinite(values).any() else None
            for metric, values in zip(("ssim", "edge_f1", "change_direction"), differences, strict=True)
        }
    return {"reference_model": COMPARATOR, "month_blocks": months, "replicates": len(draws),
            "candidate_minus_reference_95_intervals": intervals}


def summarize(rows):
    windows = sum(row["ssim_windows"] for row in rows)
    change = sum(row["qualifying_change_tiles"] for row in rows)
    methods = {}
    for name in METHODS:
        records = [row["methods"][name] for row in rows]
        methods[name] = {
            "ssim": safe_ratio(sum(record["ssim_sum"] for record in records), windows),
            "edges": {
                threshold: edge_metrics(np.sum([record["edge_counts"][threshold] for record in records], axis=0))
                for threshold in records[0]["edge_counts"]
            },
            "correct_change_direction": safe_ratio(sum(record["correct_change_tiles"] for record in records), change),
        }
    return {"dates": len(rows), "common_pixel_dates": sum(row["common_pixels"] for row in rows),
            "ssim_windows": windows, "adjacent_pixel_pairs": sum(row["adjacent_pairs"] for row in rows),
            "qualifying_tile_changes": change, "methods": methods,
            "max_observed_ndvi_rounding": max(row["max_observed_ndvi_rounding"] for row in rows),
            "paired_month_block_bootstrap": monthly_intervals(rows), "per_date": rows}


def read_reflectance(path):
    with rasterio.open(path) as dataset:
        return dataset.read((1, 2), masked=True).filled(np.nan)


def evaluate(directory, benchmark, manifest, prior, pattern):
    grid = manifest["grid"]
    width, height = grid["dimensions"]
    gap = make_holdout((height, width), pattern)
    rows, previous = [], None
    for current, archived in zip(manifest["patterns"][pattern]["dates"], prior["patterns"][pattern]["dates"], strict=True):
        if (current["date"], current["reference"]) != (archived["date"], archived["reference"]):
            raise ValueError(f"Archived reference differs on {current['date']}")
        images = {name: read_ndvi(directory / current["rasters"][name], grid) for name in ("truth", *METHODS)}
        masked = directory / current["rasters"]["masked"]
        observed = read_ndvi(masked, grid)
        observed_reflectance = read_reflectance(masked)
        clear_reflectance = np.isfinite(observed_reflectance).all(axis=0)
        clear_ndvi = np.isfinite(observed)
        common = gap & np.logical_and.reduce([np.isfinite(image) for image in images.values()])
        for name, rel in archived["rasters"].items():
            if name not in ("truth", "masked"):
                common &= np.isfinite(read_ndvi(benchmark / rel, grid))
        for name in METHODS:
            corrected = read_reflectance(directory / current["rasters"][name])
            np.testing.assert_array_equal(corrected[:, clear_reflectance], observed_reflectance[:, clear_reflectance])
            np.testing.assert_allclose(images[name][clear_ndvi], observed[clear_ndvi], atol=NDVI_ROUNDING, rtol=0)
        if int(common.sum()) != current["scores"]["NDVI"]["common_pixels"]:
            raise ValueError(f"Shared scoring pixels differ on {current['date']}")
        day = date.fromisoformat(current["date"])
        row = date_metrics(day, images, common)
        row["max_observed_ndvi_rounding"] = max(
            float(np.abs(images[name][clear_ndvi] - observed[clear_ndvi]).max(initial=0)) for name in METHODS
        )
        if previous is not None:
            earlier_day, earlier_images, earlier_common = previous
            if 0 < (day - earlier_day).days <= MAX_PAIR_DAYS:
                total, correct = change_signs(earlier_images, images, earlier_common & common)
                row["qualifying_change_tiles"] = total
                for name in METHODS:
                    row["methods"][name]["correct_change_tiles"] = correct[name]
        previous = day, images, common
        rows.append(row)
    return summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=SPATIAL / "clayton_recurrent_2005_2025")
    parser.add_argument("--benchmark", type=Path, default=SPATIAL / "clayton_cnn_2005_2025")
    args = parser.parse_args()
    manifest = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    prior = json.loads((args.benchmark / "run.json").read_text(encoding="utf-8"))
    output = {
        "settings": {"ssim_window_pixels": WINDOW, "edge_thresholds_ndvi": EDGE_THRESHOLDS,
                     "change_threshold_ndvi": CHANGE_THRESHOLD, "tile_side_pixels": SIDE,
                     "minimum_shared_tile_pixels": MIN_PIXELS, "max_days_between_acquisitions": MAX_PAIR_DAYS},
        "patterns": {pattern: evaluate(args.run, args.benchmark, manifest, prior, pattern)
                     for pattern in manifest["patterns"]},
    }
    path = args.run / "additional_quality.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path.resolve(), flush=True)


if __name__ == "__main__":
    main()
