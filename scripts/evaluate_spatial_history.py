#!/usr/bin/env python3
"""Evaluate spatial structure, gap depth, vegetation and change on saved holdouts."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stair.validation import make_holdout

METHODS = {"adaptive": "Adaptive", "one_season": "2024 season", "twenty_full": "2005–2024 seasons"}
WINDOW = 5
HIGH_NDVI = 0.60
CHANGE_THRESHOLD = 0.05
MAX_PAIR_DAYS = 32
DEPTHS = {"Up to 90 m": (0, 90), "Over 90 to 300 m": (90, 300), "Over 300 m": (300, np.inf)}


def gap_depth(mask, resolution):
    """Euclidean distance from each cell center to the nearest outside cell center."""
    height, width = mask.shape
    x = np.arange(width, dtype=float) * resolution
    y = np.arange(height, dtype=float) * resolution
    dx_squared = (x[:, None] - x) ** 2
    dy_squared = (y[:, None] - y) ** 2
    row_distance = np.where(mask[:, None, :], np.inf, dx_squared[None]).min(axis=2)
    return np.sqrt((dy_squared[:, :, None] + row_distance[None]).min(axis=1))


def ssim_windows(actual, prediction):
    """SSIM for paired, fully valid windows with uniform weights and NDVI range 2."""
    a, b = actual.mean(axis=1), prediction.mean(axis=1)
    da, db = actual - a[:, None], prediction - b[:, None]
    sample_dof = actual.shape[1] - 1
    v_a = (da * da).sum(axis=1) / sample_dof
    v_b = (db * db).sum(axis=1) / sample_dof
    cov = (da * db).sum(axis=1) / sample_dof
    c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
    return (2 * a * b + c1) * (2 * cov + c2) / ((a * a + b * b + c1) * (v_a + v_b + c2))


def read_ndvi(path, grid):
    with rasterio.open(path) as source:
        if (source.descriptions != ("NIR", "Red", "NDVI") or source.crs.to_string() != grid["crs"]
                or [source.width, source.height] != grid["dimensions"]
                or list(source.transform)[:6] != grid["transform"]):
            raise ValueError(f"Unexpected grid or band order in {path}")
        return source.read(3, masked=True).astype(float).filled(np.nan)


def summarize_changes(rows, methods):
    scored = sum(row["pixels"] for row in rows)
    substantial = sum(row["substantial_changes"] for row in rows)
    return {"date_pairs": len(rows), "pixel_pairs": scored, "substantial_changes": substantial,
            "methods": {name: {
                "change_rmse": (sum(row["methods"][name]["squared_error"] for row in rows) / scored) ** 0.5 if scored else None,
                "direction_agreement": sum(row["methods"][name]["correct_direction"] for row in rows) / substantial
                if substantial else None} for name in methods}, "pairs": rows}


def evaluate_pattern(source, runs, grid, pattern):
    width, height = grid["dimensions"]
    gap = make_holdout((height, width), pattern)
    depth = gap_depth(gap, abs(grid["transform"][0]))
    depth_masks = {name: (depth > low) & (depth <= high) for name, (low, high) in DEPTHS.items()}
    depth_errors = {name: {group: [] for group in DEPTHS} for name in METHODS}
    structures = {name: [] for name in METHODS}
    confusion = {name: np.zeros(4, dtype=np.int64) for name in METHODS}
    date_rows, time_rows = [], []
    previous = None
    for row in runs:
        day = row["date"]
        truth = read_ndvi(source / row["rasters"]["truth"], grid)
        estimates = {name: read_ndvi(source / row["rasters"][name], grid) for name in row["scores"]["NDVI"]["methods"]}
        common = gap & np.isfinite(truth) & np.logical_and.reduce([np.isfinite(array) for array in estimates.values()])
        count = int(common.sum())
        if count != row["scores"]["NDVI"]["common_pixels"]:
            raise ValueError(f"Stored score support disagrees with GeoTIFFs for {day}")
        if not count:
            continue
        good_windows = sliding_window_view(common, (WINDOW, WINDOW)).all(axis=(-2, -1))
        truth_windows = sliding_window_view(truth, (WINDOW, WINDOW))[good_windows].reshape(-1, WINDOW ** 2)
        high = truth[common] >= HIGH_NDVI
        entry = {"date": day, "common_pixels": count, "ssim_windows": len(truth_windows), "methods": {}}
        for name in METHODS:
            array = estimates[name]
            error = array - truth
            depth_row = {}
            for group, mask in depth_masks.items():
                errors = np.abs(error[mask & common])
                depth_errors[name][group].append(errors)
                depth_row[group] = {"pixels": len(errors), "mae": float(errors.mean()) if len(errors) else None}
            layout = sliding_window_view(array, (WINDOW, WINDOW))[good_windows].reshape(-1, WINDOW ** 2)
            ssim = ssim_windows(truth_windows, layout)
            structures[name].append(ssim)
            fit_high = array[common] >= HIGH_NDVI
            confusion[name] += [np.sum(fit_high & high), np.sum(fit_high & ~high),
                                np.sum(~fit_high & high), np.sum(~fit_high & ~high)]
            entry["methods"][name] = {"ssim": float(ssim.mean()) if len(ssim) else None, "gap_depth": depth_row}
        date_rows.append(entry)

        if previous is not None:
            earlier, earlier_truth, earlier_estimates, earlier_common = previous
            days = (date.fromisoformat(day) - date.fromisoformat(earlier)).days
            shared = common & earlier_common
            if 0 < days <= MAX_PAIR_DAYS and shared.any():
                observed_change = truth[shared] - earlier_truth[shared]
                large = np.abs(observed_change) >= CHANGE_THRESHOLD
                change = {"first": earlier, "last": day, "days": days, "pixels": int(shared.sum()),
                          "substantial_changes": int(large.sum()), "methods": {}}
                for name in METHODS:
                    prediction = estimates[name][shared] - earlier_estimates[name][shared]
                    change["methods"][name] = {"squared_error": float(np.square(prediction - observed_change).sum()),
                                               "correct_direction": int((np.sign(prediction[large]) == np.sign(observed_change[large])).sum())}
                time_rows.append(change)
        previous = day, truth, {name: estimates[name] for name in METHODS}, common

    if not date_rows:
        raise ValueError(f"No usable dates for {pattern}")
    structure, depth_stats, detection = {}, {}, {}
    for name in METHODS:
        similarity = np.concatenate(structures[name])
        structure[name] = {"mean_ssim": float(similarity.mean()) if len(similarity) else None, "windows": len(similarity)}
        depth_stats[name] = {}
        for group, samples in depth_errors[name].items():
            errors = np.concatenate(samples)
            depth_stats[name][group] = {"pixels": len(errors), "mae": float(errors.mean()) if len(errors) else None,
                                       "p90_absolute_error": float(np.quantile(errors, 0.9)) if len(errors) else None}
        tp, fp, fn, tn = map(int, confusion[name])
        detection[name] = {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
                           "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                           "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None}
    return {"dates": date_rows, "structural_similarity": structure, "gap_depth": depth_stats,
            "high_ndvi_detection": detection, "temporal_change": summarize_changes(time_rows, METHODS)}


def plot(directory, pattern, results):
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(11, 7.2), layout="constrained")
    structure, gap, vegetation, change = figure.subplots(2, 2).flat
    colors = ("#717680", "#246da0", "#bd4b24")
    locations = np.arange(len(METHODS))
    groups = [key for key, stats in results["gap_depth"]["adaptive"].items() if stats["pixels"]]
    for index, (name, color) in enumerate(zip(METHODS, colors)):
        structure.scatter(index, results["structural_similarity"][name]["mean_ssim"], color=color, s=58, zorder=3)
        gap.plot(groups, [results["gap_depth"][name][group]["mae"] for group in groups],
                 "o-", label=METHODS[name], color=color, linewidth=1.8)
        vegetation.scatter(index, results["high_ndvi_detection"][name]["f1"], color=color, s=58, zorder=3)
        agreement = results["temporal_change"]["methods"][name]["direction_agreement"]
        change.scatter(index, agreement, color=color, s=58, zorder=3)
    for ax in (structure, vegetation, change):
        ax.set_xticks(locations, METHODS.values())
        ax.set_xlim(-0.45, len(METHODS) - 0.55)
    structure.set(title="Local structure", ylabel="Mean SSIM · higher is better")
    gap.set(title="Accuracy by gap depth", ylabel="NDVI MAE · lower is better")
    vegetation.set(title=f"High NDVI detection (≥ {HIGH_NDVI:.2f})", ylabel="F1 · higher is better")
    change.set(title=f"Direction of observed change (|ΔNDVI| ≥ {CHANGE_THRESHOLD:.2f})", ylabel="Agreement · higher is better")
    change.yaxis.set_major_formatter(PercentFormatter(1))
    gap.legend(fontsize=8, frameon=False)
    for ax in (structure, gap, vegetation, change):
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["right", "top"]].set_visible(False)
    figure.suptitle(f"Clayton Township | {pattern.capitalize()} holdouts, March–October 2025", fontsize=15, weight="bold")
    figure.savefig(directory / f"{pattern}_complementary_metrics.png", dpi=180)


def run(source, output):
    manifest = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if not set(METHODS) <= manifest["methods"].keys():
        raise ValueError("Use the saved matched March–October spatial calibration run")
    results = {}
    for pattern, report in manifest["patterns"].items():
        results[pattern] = evaluate_pattern(source, report["dates"], manifest["grid"], pattern)
        plot(output, pattern, results[pattern])
        print(f"{pattern}: evaluated {len(results[pattern]['dates'])} dates", flush=True)
    report = {"source": str(source.resolve()),
              "settings": {"ssim_window_pixels": [WINDOW, WINDOW], "ssim_window_weights": "uniform",
                           "ssim_data_range": 2, "high_ndvi_threshold": HIGH_NDVI,
                           "substantial_change": CHANGE_THRESHOLD, "max_pair_days": MAX_PAIR_DAYS,
                           "gap_depth_meters": {name: [a, b if np.isfinite(b) else None]
                                                for name, (a, b) in DEPTHS.items()}},
              "support": "Original matched withheld NDVI pixels; every SSIM window needs all pixels supported",
              "methods": METHODS, "patterns": results}
    (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/spatial_validation/clayton_full_season_history_2005_2025")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source / "additional_evaluation"
    if output.exists():
        parser.error("Choose a fresh output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".spatial_metrics_", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        run(args.source, staging)
        staging.rename(output)
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
