#!/usr/bin/env python3
"""Evaluate historical calibration of STAIR spatial cloud filling offline."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import GRID, SPRING_FALL_CACHE
from scripts.compare_training_history import (VARIABLES, compare_variables, monthly_bootstrap, pooled_scores,
                                              sha256, write_metrics, write_raster)
from scripts.export_stair_stages import WINDOW
from scripts.run_daily_ndvi import read_cache
from scripts.run_watershed_fusion import restore, validity
from stair.history_model import FEATURES_BY_SOURCE, HistoryCorrection, RidgeFit
from stair.holdout_density import density_panel
from stair.ndvi import ndvi
from stair.spatial import segment_reference
from stair.temporal import interpolate_temporal_numpy
from stair.validation import make_holdout

BASELINES = {
    "temporal": "Temporal interpolation", "global": "Global correction", "adaptive": "Adaptive correction",
}
TRAINING_WINDOWS = {
    "six_month": (2024, 8, "March–August 2024", "Six-month calibration"),
    "twenty_matched": (2005, 8, "March–August 2005–2024", "20 years, March–August"),
    "twenty_full": (2005, 10, "March–October 2005–2024", "20 years, March–October"),
    "one_season": (2024, 10, "March–October 2024", "One season, March–October"),
}
PATTERNS = ("cloud", "stripes")
REFERENCE_DAYS = 21
MIN_TARGET_CLEAR = 0.60
MIN_REFERENCE_CLEAR = 0.95
SEGMENT_SETTINGS = {"sample_limit": 320, "null_draws": 4, "seed": 42}


def scenarios(cache, year, patterns, skipped):
    """Generate artificial masks and estimate stages from eligible nearby Landsat."""
    _, images, _ = read_cache(cache / f"clayton_{year}.npz", year)
    yield from scenarios_for_images(images, patterns, skipped)


def scenarios_for_images(images, patterns, skipped):
    days = sorted(images)
    fractions = {day: float(validity(image).mean()) for day, image in images.items()}
    references = [(date.fromisoformat(day), day) for day, fraction in fractions.items()
                  if fraction >= MIN_REFERENCE_CLEAR]
    segments = {}
    masks = {pattern: make_holdout(next(iter(images.values())).shape[1:], pattern) for pattern in patterns}
    for day in days:
        target = date.fromisoformat(day)
        if not 3 <= target.month <= 10:
            continue
        if fractions[day] < MIN_TARGET_CLEAR:
            skipped.append({"date": day, "reason": "less than 60% valid target reflectance"})
            continue
        options = [(abs((d - target).days), d > target, key) for d, key in references
                   if 0 < abs((d - target).days) <= REFERENCE_DAYS]
        if not options:
            skipped.append({"date": day, "reason": "no independent 95% clear Landsat reference within 21 days"})
            continue
        reference_day = min(options)[2]
        reference = images[reference_day]
        if reference_day not in segments:
            segments[reference_day] = segment_reference(reference, **SEGMENT_SETTINGS)
        lo = bisect_left(days, (target - timedelta(days=WINDOW)).isoformat())
        hi = bisect_right(days, (target + timedelta(days=WINDOW)).isoformat())
        neighbors = [key for key in days[lo:hi] if key != day]
        stack = np.stack([images[key] for key in neighbors])
        truth = images[day]
        for pattern, forbidden in masks.items():
            observed = np.where(forbidden, np.nan, truth)
            temporal = interpolate_temporal_numpy(stack, neighbors, day, observed, max_distance_days=WINDOW).filled
            global_fill, adaptive, metadata = restore(observed, temporal, reference, segmentation=segments[reference_day])
            yield day, pattern, forbidden, truth, observed, reference, reference_day, metadata, {
                "temporal": temporal, "global": global_fill, "adaptive": adaptive,
            }


def train(cache, work, calibrations):
    fits = {name: RidgeFit(input_source="spatial") for name in calibrations}
    skipped, models = [], {}
    for year in range(2005, 2025):
        count = 0
        for day, _, holdout, truth, _, reference, _, _, stages in scenarios(cache, year, ("cloud",), skipped):
            sample = (day, stages["adaptive"], reference, np.where(holdout, truth, np.nan))
            for name, fit in fits.items():
                first_year, last_month, _, _ = TRAINING_WINDOWS[name]
                if first_year <= year and int(day[5:7]) <= last_month:
                    fit.add(*sample)
            count += 1
        print(f"{year}: {count} eligible Landsat dates for spatial training", flush=True)
    for name, fit in fits.items():
        path = work / f"{name}_spatial_model.npz"
        fit.finish().save(path)
        models[name] = HistoryCorrection.load(path)
    return models, skipped


def calibrate(model, adaptive, reference, day, holdout):
    predicted = model.predict(adaptive, reference, day)
    replace = holdout & validity(predicted)
    return np.where(replace, predicted, adaptive), int(replace.sum())


def gradient_scores(truth, stages, common):
    """Score contrast between neighboring 30 m pixels within the gap."""
    offsets = []
    for axis in range(2):
        a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        offsets.append((a, b, common[a] & common[b]))
    count = sum(int(mask.sum()) for _, _, mask in offsets)
    truth_ndvi = ndvi(truth)[0]
    values = {}
    for name, raster in stages.items():
        error = ndvi(raster)[0] - truth_ndvi
        values[name] = sum(float(np.square((error[b] - error[a])[mask]).sum()) for a, b, mask in offsets)
    return {"pairs": count, "squared_error": values}


def evaluate(cache, work, models, *, linear_fallbacks=None):
    rows = {pattern: [] for pattern in PATTERNS}
    samples = {pattern: [] for pattern in PATTERNS}
    skipped = []
    for day, pattern, holdout, truth, observed, reference, reference_day, info, stages in scenarios(
            cache, 2025, PATTERNS, skipped):
        applied = {}
        for name, model in models.items():
            stages[name], applied[name] = calibrate(model, stages["adaptive"], reference, day, holdout)
        fallback_counts = {}
        for name, linear in (linear_fallbacks or {}).items():
            fallback = (holdout & ~np.isfinite(ndvi(stages[name])[0]) & np.isfinite(ndvi(stages[linear])[0]))
            stages[name][:, fallback] = stages[linear][:, fallback]
            fallback_counts[name] = int(fallback.sum())
        shown = validity(observed)
        for image in stages.values():
            np.testing.assert_array_equal(image[:, shown], observed[:, shown])
        withheld = np.where(holdout, truth, np.nan)
        scores, sample = compare_variables(withheld, stages)
        common = np.logical_and.reduce([np.isfinite(ndvi(image)[0]) for image in (withheld, *stages.values())])
        gradients = gradient_scores(withheld, stages, common)
        folder = work / "rasters" / pattern / day
        folder.mkdir(parents=True)
        paths = {}
        for name, image in {"truth": truth, "masked": observed, **stages}.items():
            path = folder / f"{name}.tif"
            write_raster(path, image, day, name, spatial_holdout=pattern)
            paths[name] = path.relative_to(work).as_posix()
        rows[pattern].append({"date": day, "reference": reference_day, "spatial_fit": info,
                              "calibrated_pixels": applied, "linear_fallback_pixels": fallback_counts,
                              "scores": scores, "gradients": gradients,
                              "rasters": paths})
        samples[pattern].append(sample)
    return rows, samples, skipped


def summarize(rows, samples, methods, short, long):
    pooled = pooled_scores(samples, methods)
    counts = [row["scores"]["NDVI"]["common_pixels"] for row in rows]
    pair_count = sum(row["gradients"]["pairs"] for row in rows)
    gradients = {name: (sum(row["gradients"]["squared_error"][name] for row in rows) / pair_count) ** 0.5
                 if pair_count else None for name in methods}
    coverage = {}
    intervals = {}
    comparisons = [(name, short) for name in methods if name not in BASELINES and name != short]
    comparisons.extend(((long, "adaptive"), (short, "adaptive")))
    for variable in VARIABLES:
        scores = [row["scores"][variable] for row in rows]
        eligible = sum(score["eligible_pixels"] for score in scores)
        coverage[variable] = {name: sum(score["methods"][name]["covered_pixels"] for score in scores) / eligible
                              if eligible else None for name in methods}
        intervals[variable] = monthly_bootstrap(rows, variable=variable, comparisons=comparisons)
    return {"scored_ndvi_dates": sum(n > 0 for n in counts), "summary": pooled, "coverage": coverage,
            "ndvi_neighbor_pairs": pair_count, "ndvi_gradient_rmse": gradients, "month_block_bootstrap": intervals,
            "matched_history_wins": sum(row["scores"]["NDVI"]["methods"][long]["rmse"]
                                        < row["scores"]["NDVI"]["methods"][short]["rmse"]
                                        for n, row in zip(counts, rows) if n)}


def density_figure(work, pattern, samples, calibrations, methods, period):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 7), layout="constrained")
    axes = figure.subplots(2, 3)
    names = ("truth", "adaptive", *calibrations)
    colors = ("#1d2734", "#8b8c94", "#296ca8", "#cb5023", "#217e6e")
    lines = ("-", ":", "--", "-", "-.")
    styles = {name: {"label": "Withheld Landsat" if name == "truth" else methods[name],
                     "color": color, "linestyle": line, "linewidth": 1.8}
              for name, color, line in zip(names, colors, lines)}
    bandwidths = {}
    for column, (variable, title) in enumerate(zip(VARIABLES, ("NIR reflectance", "Red reflectance", "NDVI"))):
        pooled = {name: np.concatenate([sample[variable][name] for sample in samples]) for name in names}
        count = len(pooled["truth"])
        if not count:
            raise ValueError(f"No common pixels for the {variable} density on {pattern} masks")
        # Use equal sample locations and a shared bandwidth for every curve in a panel.
        indices = np.random.default_rng(2025).choice(count, min(count, 16000), replace=False)
        values = {name: series[indices] for name, series in pooled.items()}
        errors = {name: values[name] - values["truth"] for name in names if name != "truth"}
        bandwidths[variable] = {"common_pixels": count, "plotted_pixels": len(indices),
                                "values": density_panel(axes[0, column], values, styles),
                                "errors": density_panel(axes[1, column], errors, styles, centered=True)}
        axes[0, column].set(xlabel=title, title=f"{title} values")
        axes[1, column].set(xlabel=f"{title} error", title=f"{title} paired errors")
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability density")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=9)
    figure.suptitle(f"Clayton Township | {pattern.capitalize()} masks, {period} 2025\n"
                    "Gaussian density estimates on shared withheld Landsat pixels", fontsize=15, fontweight="bold")
    for suffix in ("png", "pdf"):
        figure.savefig(work / f"{pattern}_density_curves.{suffix}", dpi=180)
    return bandwidths


def run(cache, work, *, matched_season=False):
    short, long = ("one_season", "twenty_full") if matched_season else ("six_month", "twenty_matched")
    calibrations = (short, long) if matched_season else (short, long, "twenty_full")
    methods = BASELINES | {name: TRAINING_WINDOWS[name][3] for name in calibrations}
    primary_period = "March–October" if matched_season else "March–August"
    fingerprints = {str(year): sha256(cache / f"clayton_{year}.npz") for year in range(2005, 2026)}
    models, training_skips = train(cache, work, calibrations)
    rows, samples, test_skips = evaluate(cache, work, models)
    results = {}
    for pattern, entries in rows.items():
        if not entries:
            raise ValueError(f"No eligible 2025 acquisitions for {pattern}")
        spring_summer = [(row, sample) for row, sample in zip(entries, samples[pattern]) if row["date"][5:7] <= "08"]
        early_rows, early_samples = zip(*spring_summer) if spring_summer else ((), ())
        plot_samples = samples[pattern] if matched_season else early_samples
        results[pattern] = {"march_october": summarize(entries, samples[pattern], methods, short, long),
                            "density": density_figure(work, pattern, plot_samples, calibrations, methods, primary_period),
                            "dates": entries}
        if early_rows:
            results[pattern]["march_august"] = summarize(early_rows, early_samples, methods, short, long)
        metrics = work / pattern
        metrics.mkdir()
        write_metrics(metrics, entries)
        print(f"{pattern}: {len(entries)} eligible 2025 dates", flush=True)
    report = {
        "grid": GRID, "methods": methods, "patterns": results, "input_sha256": fingerprints,
        "training_skipped": training_skips, "test_skipped": test_skips,
        "training": {name: {"dates": model.training_dates, "pixels": model.training_pixel_dates,
                             "coefficients": model.coefficients.tolist()} for name, model in models.items()},
        "settings": {"training": {name: TRAINING_WINDOWS[name][2] for name in calibrations},
                     "primary_comparison": [long, short], "training_mask": "cloud", "test_year": 2025,
                     "primary_test_period": primary_period,
                     "min_target_clear": MIN_TARGET_CLEAR, "min_reference_clear": MIN_REFERENCE_CLEAR,
                     "reference_days": REFERENCE_DAYS, "max_temporal_days_per_side": WINDOW,
                     "segmentation": SEGMENT_SETTINGS, "ridge": 0.001, "features": FEATURES_BY_SOURCE["spatial"]},
    }
    (work / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=SPRING_FALL_CACHE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matched-season", action="store_true", help="Train both histories using March–October")
    args = parser.parse_args()
    if args.output is None:
        name = "clayton_full_season_history_2005_2025" if args.matched_season else "clayton_history_2005_2025"
        args.output = ROOT / "outputs" / "spatial_validation" / name
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".spatial_history_", dir=args.output.parent) as temporary:
        staging = Path(temporary) / args.output.name
        staging.mkdir()
        run(args.cache, staging, matched_season=args.matched_season)
        staging.rename(args.output)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
