#!/usr/bin/env python3
"""Train 20-year and six-month fusion corrections, then evaluate withheld 2025 dates."""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

from affine import Affine
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import GRID, SPRING_FALL_CACHE
from scripts.export_stair_stages import NODATA, WINDOW
from scripts.run_daily_ndvi import read_cache
from stair.history_model import FEATURES, HistoryCorrection, fit_correction
from stair.ndvi import DailyNDVI, ndvi
from stair.validation import score_holdout

METHODS = {
    "fusion": "Existing residual fusion",
    "six_month": "Correction trained March–August 2024",
    "twenty_year": "Correction trained 2005–2024",
    "same_pairs_landsat": "Landsat interpolation with matched pairs",
    "landsat_only": "Nearest-clear Landsat interpolation",
    "modis_only": "Daily MODIS resampled to 30 m",
}
VARIABLES = ("NIR", "Red", "NDVI")
SHORT_START, SHORT_END = "2024-03-01", "2024-09-01"
TEST_START, TEST_END = "2025-03-01", "2025-11-01"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_samples(cache, years, first, end, minimum_clear, records):
    """Use known clear targets as labels after excluding them from their features."""
    for year in years:
        manifest, landsat, modis = read_cache(cache / f"clayton_{year}.npz", year)
        if manifest["end_exclusive"] > "2025-01-01":
            raise ValueError("Training context reaches the test year")
        model = DailyNDVI(landsat, modis, min_clear_fraction=minimum_clear)
        year_count = 0
        for day in model.landsat_days:
            if not first <= day < end or not 3 <= date.fromisoformat(day).month <= 10:
                continue
            truth = landsat[day]
            clear = np.isfinite(ndvi(truth)[0])
            if clear.mean() < minimum_clear:
                continue
            fused = model.holdout_reflectance(day)["fusion"]
            coarse = modis.get(day, model.missing)
            valid = np.isfinite(np.concatenate((truth, fused, coarse))).all(axis=0)
            records.append({"date": day, "qa_clear_pixels": int(clear.sum()),
                            "training_pixels": int(valid.sum())})
            year_count += int(valid.any())
            yield day, fused, coarse, truth
        print(f"Training {year}: {year_count} supported withheld acquisitions", flush=True)


def values_score(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if not len(actual):
        return {"pixels": 0, **dict.fromkeys(("rmse", "mae", "bias", "r2", "pearson_r", "wasserstein_1"))}
    metrics = score_holdout(actual[None, None], predicted[None, None],
                            np.ones((1, len(actual)), dtype=bool), ["value"])["bands"]["value"]
    return {"pixels": len(actual), **metrics,
            "wasserstein_1": float(np.mean(np.abs(np.sort(actual) - np.sort(predicted))))}


def compare_variables(truth, methods):
    """All methods share exactly the same pixels within each evaluated variable."""
    arrays = {"truth": np.concatenate((truth, ndvi(truth))),
              **{name: np.concatenate((image, ndvi(image))) for name, image in methods.items()}}
    scores, samples = {}, {}
    for index, variable in enumerate(VARIABLES):
        eligible = np.isfinite(arrays["truth"][index])
        valid = {name: np.isfinite(arrays[name][index]) for name in methods}
        common = eligible & np.logical_and.reduce(list(valid.values()))
        samples[variable] = {name: image[index][common].astype(float) for name, image in arrays.items()}
        scores[variable] = {
            "eligible_pixels": int(eligible.sum()), "common_pixels": int(common.sum()),
            "methods": {
                name: {**values_score(samples[variable]["truth"], samples[variable][name]),
                       "coverage": float((eligible & valid[name]).sum() / eligible.sum()) if eligible.any() else None}
                for name in methods
            },
        }
    return scores, samples


def pooled_scores(samples):
    report = {}
    for variable in VARIABLES:
        pooled = {name: np.concatenate([sample[variable][name] for sample in samples])
                  for name in ("truth", *METHODS)}
        report[variable] = {name: values_score(pooled["truth"], pooled[name]) for name in METHODS}
    return report


def monthly_bootstrap(rows, *, seed=20261118, repeats=5000):
    """Resample shared calendar-month blocks, preserving paired method comparisons."""
    months = sorted({row["date"][:7] for row in rows})
    totals = np.zeros((len(months), 4))
    names = ("fusion", "six_month", "twenty_year")
    for index, month in enumerate(months):
        for row in rows:
            if not row["date"].startswith(month):
                continue
            scores = row["scores"]["NDVI"]
            count = scores["common_pixels"]
            totals[index, 0] += count
            if count:
                totals[index, 1:] += [count * scores["methods"][name]["rmse"] ** 2 for name in names]
    totals = totals[totals[:, 0] > 0]
    if len(totals) < 2:
        return {"supported_month_blocks": len(totals), "comparisons": {}}
    rng = np.random.default_rng(seed)
    selections = rng.integers(len(totals), size=(repeats, len(totals)))
    sums = totals[selections].sum(axis=1)
    bootstrap_rmse = np.sqrt(sums[:, 1:] / sums[:, :1])
    combined = totals.sum(axis=0)
    actual_rmse = np.sqrt(combined[1:] / combined[0])
    comparisons = {}
    for left, right in ((2, 1), (2, 0), (1, 0)):
        delta = bootstrap_rmse[:, left] - bootstrap_rmse[:, right]
        comparisons[f"{names[left]}_minus_{names[right]}"] = {
            "rmse_difference": float(actual_rmse[left] - actual_rmse[right]),
            "percentile_95_interval": np.quantile(delta, [0.025, 0.975]).tolist(),
        }
    return {"supported_month_blocks": len(totals), "replicates": repeats, "seed": seed,
            "sampling_unit": "calendar month within Clayton in 2025", "comparisons": comparisons}


def write_raster(path, reflectance, day, method):
    values = np.concatenate((reflectance, ndvi(reflectance)))
    width, height = GRID["dimensions"]
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=3, dtype="float32", crs=GRID["crs"], transform=Affine(*GRID["transform"]),
                       nodata=NODATA, compress="deflate", predictor=3) as output:
        output.write(np.where(np.isfinite(values), values, NODATA).astype(np.float32))
        output.update_tags(target_date=day, method=method,
                           target_landsat_excluded=str(method != "truth").lower())
        for index, variable in enumerate(VARIABLES, start=1):
            output.set_band_description(index, variable)
            output.update_tags(index, units="unitless" if variable == "NDVI" else "surface_reflectance")


def evaluate(cache, work, models, minimum_clear):
    _, landsat, modis = read_cache(cache / "clayton_2025.npz", 2025)
    baseline = DailyNDVI(landsat, modis, min_clear_fraction=minimum_clear)
    rows, samples = [], []
    for day in baseline.landsat_days:
        if not TEST_START <= day < TEST_END:
            continue
        truth = landsat[day]
        if np.isfinite(ndvi(truth)[0]).mean() < minimum_clear:
            continue
        controls = baseline.holdout_reflectance(day)
        coarse = modis.get(day, baseline.missing)
        methods = {"fusion": controls["fusion"],
                   **{name: model.predict(controls["fusion"], coarse, day) for name, model in models.items()},
                   **{name: controls[name] for name in ("same_pairs_landsat", "landsat_only", "modis_only")}}
        scores, sample = compare_variables(truth, methods)
        folder = work / "rasters" / day
        folder.mkdir(parents=True)
        paths = {}
        for name, image in {"truth": truth, **methods}.items():
            path = folder / f"{name}.tif"
            write_raster(path, image, day, name)
            paths[name] = path.relative_to(work).as_posix()
        support = {name: int(np.isfinite(image).all(axis=0).sum()) for name, image in (
            ("target_modis", coarse), ("matched_pair_interpolation", controls["same_pairs_landsat"]),
            ("landsat_interpolation", controls["landsat_only"]))}
        rows.append({"date": day, "input_support_pixels": support, "scores": scores, "rasters": paths})
        samples.append(sample)
        print(f"Test {day}: {scores['NDVI']['common_pixels']} common NDVI pixels", flush=True)
    if not rows or not any(row["scores"]["NDVI"]["common_pixels"] for row in rows):
        raise ValueError("No shared evaluation pixels in 2025")
    monthly = {}
    for month in sorted({row["date"][:7] for row in rows}):
        group = [sample for row, sample in zip(rows, samples) if row["date"].startswith(month)]
        monthly[month] = pooled_scores(group)
    return rows, samples, monthly


def plot_results(work, rows, samples):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    names = ("fusion", "six_month", "twenty_year")
    labels = ("Existing fusion", "Six-month training", "20-year training")
    colors = ("#636b78", "#2878a6", "#c05a24")
    figure = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(figure)
    density, errors = figure.subplots(2, 1)
    pooled = {name: np.concatenate([sample["NDVI"][name] for sample in samples]) for name in ("truth", *names)}
    bins = np.linspace(-1, 1, 101)
    density.hist(pooled["truth"], bins=bins, density=True, histtype="step", color="#20262d", label="QA-clear Landsat", linewidth=1.8)
    for name, label, color in zip(names, labels, colors):
        density.hist(pooled[name], bins=bins, density=True, histtype="step", label=label, color=color)
        x = [datetime.fromisoformat(row["date"]) for row in rows]
        y = [row["scores"]["NDVI"]["methods"][name]["rmse"] for row in rows]
        errors.plot(x, y, marker="o", markersize=3, color=color, label=label)
    density.set(xlabel="NDVI", ylabel="Probability density", title="Withheld 2025 acquisitions on identical pixels")
    density.legend(loc="upper left", fontsize=9)
    errors.set(xlabel="Withheld Landsat acquisition", ylabel="Paired NDVI RMSE")
    errors.grid(alpha=0.2)
    errors.legend(fontsize=9)
    figure.suptitle("Clayton Township | Effect of training history", fontsize=17, fontweight="bold")
    figure.savefig(work / "history_comparison.png", dpi=180)


def write_metrics(work, rows):
    with (work / "per_date_metrics.csv").open("w", newline="", encoding="utf-8") as output:
        fields = ["date", "variable", "method", "eligible_pixels", "common_pixels", "coverage",
                  "rmse", "mae", "bias", "r2", "pearson_r", "wasserstein_1"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for variable, scores in row["scores"].items():
                for name, metrics in scores["methods"].items():
                    writer.writerow({"date": row["date"], "variable": variable, "method": name,
                                     "eligible_pixels": scores["eligible_pixels"],
                                     "common_pixels": scores["common_pixels"],
                                     **{key: metrics[key] for key in fields[5:]}})


def run(cache, work, ridge, minimum_clear):
    sources = []
    for year in range(2005, 2026):
        path = cache / f"clayton_{year}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing annual input {path}")
        sources.append({"year": year, "file": path.name, "sha256": sha256(path)})
    records, models = {}, {}
    for name, years, first, end in (
        ("twenty_year", range(2005, 2025), "2005-03-01", "2024-11-01"),
        ("six_month", [2024], SHORT_START, SHORT_END),
    ):
        records[name] = []
        samples = training_samples(cache, years, first, end, minimum_clear, records[name])
        model = fit_correction(samples, ridge=ridge)
        if any(day >= TEST_START for day in model.training_dates):
            raise RuntimeError("Test acquisition entered training")
        path = work / f"{name}_model.npz"
        model.save(path)
        models[name] = HistoryCorrection.load(path)
        np.testing.assert_array_equal(models[name].coefficients, model.coefficients)
    rows, samples, monthly = evaluate(cache, work, models, minimum_clear)
    summary = pooled_scores(samples)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__, "grid": GRID,
        "input_cache": str(cache.resolve()), "input_fingerprints": sources,
        "source_fingerprints": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in (
            ROOT / "scripts/compare_training_history.py", ROOT / "stair/history_model.py",
            ROOT / "stair/ndvi.py", ROOT / "stair/temporal.py", ROOT / "stair/validation.py")},
        "model": "Date-balanced ridge correction of NIR/Red residual-fusion errors",
        "features": FEATURES, "ridge": ridge, "minimum_target_and_anchor_clear_fraction": minimum_clear,
        "max_anchor_days_per_side": WINDOW,
        "training_seasons": "March–October for 2005–2024; March–August 2024 for the short history",
        "training_context": "Identical cached +/-32-day contexts for each target; labels restricted to each training period",
        "test_period": [TEST_START, TEST_END], "test_end_exclusive": True,
        "test_protocol": "Retrospective leave-one-acquisition-out reconstruction; other 2025 Landsat acquisitions are allowed anchors",
        "training_protocol": "Test year excluded from all coefficient fitting and training contexts; no tuning on 2025",
        "score_protocol": "Same originally QA-clear pixels for all methods, separately for NIR, red and NDVI",
        "limitations": [
            "One Clayton chip and one test year; spatial transfer and multiple independent test years are untested",
            "Landsat mission-native reflectance pooled across Landsat 5/7/8/9 without spectral harmonization",
            "Independent cross-sensor registration remains unverified",
            "MCD43A4 daily dates represent overlapping 16-day retrieval windows",
            "QA-clear Landsat provides a satellite reference with its own uncertainty",
            "The trainable correction extends the existing residual-fusion baseline",
            "The short training period ends in August; September and October are outside its seasonal training coverage",
            "History length, recency, sensor mix and seasonal coverage all differ between the two training sets",
        ],
        "training": {name: {"supported_dates": len(model.training_dates),
                             "training_pixel_dates": model.training_pixel_dates,
                             "dates": model.training_dates, "candidate_dates": records[name],
                             "coefficients": model.coefficients.tolist()}
                     for name, model in models.items()},
        "methods": METHODS, "raster_bands": VARIABLES, "ndvi_display_limits": [-0.2, 1.0],
        "test_acquisitions": len(rows), "summary": summary, "monthly": monthly,
        "march_august_summary": pooled_scores([sample for row, sample in zip(rows, samples) if row["date"] < "2025-09-01"]),
        "month_block_bootstrap": monthly_bootstrap(rows), "dates": rows,
    }
    (work / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_metrics(work, rows)
    plot_results(work, rows, samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=SPRING_FALL_CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fusion_validation/history_comparison_2005_2025")
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--minimum-clear", type=float, default=0.60)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    if not np.isfinite(args.ridge) or args.ridge <= 0 or not 0 < args.minimum_clear <= 1:
        parser.error("Use a positive finite ridge and a clear fraction in (0, 1]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".history_", dir=args.output.parent) as temporary:
        staging = Path(temporary) / args.output.name
        staging.mkdir()
        run(args.cache, staging, args.ridge, args.minimum_clear)
        staging.rename(args.output)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
