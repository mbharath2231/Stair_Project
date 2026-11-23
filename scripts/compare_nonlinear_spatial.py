#!/usr/bin/env python3
"""Benchmark nonlinear calibrations of Clayton Landsat spatial gap filling."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import GRID, SPRING_FALL_CACHE
from scripts.compare_spatial_history import BASELINES, density_figure, evaluate, scenarios, summarize
from scripts.compare_spatial_harmonization import DEFAULT_BASELINE
from scripts.compare_training_history import sha256, write_metrics
from scripts.export_stair_stages import WINDOW
from stair.history_model import HistoryCorrection
from stair.ndvi import ndvi
from stair.nonlinear_spatial import ALGORITHMS, BASE_COLUMNS, design, fit_model

HISTORIES = {"one_season": "2024 season", "twenty_full": "2005–2024 seasons"}
DEVELOPMENT_MONTHS = (5, 8)


def collect_examples(cache):
    """Collect the exact predictors, targets and weights used on the original training masks."""
    dates, feature_rows, bases, residuals, weights = [], [], [], [], []
    for year in range(2005, 2025):
        before = len(dates)
        for day, _, holdout, truth, _, reference, _, _, stages in scenarios(cache, year, ("cloud",), []):
            adaptive = stages["adaptive"]
            x = design(adaptive, reference, day, spatial=True)
            eligible = holdout & np.isfinite(truth).all(axis=0) & np.isfinite(x[:BASE_COLUMNS]).all(axis=0)
            count = int(eligible.sum())
            if not count:
                continue
            selected = x[:, eligible].T
            if not np.isfinite(selected).all():
                raise ValueError(f"Neighborhood predictors removed eligible pixels on {day}")
            dates.append((day, count))
            feature_rows.append(selected)
            bases.append(adaptive[:, eligible].T)
            residuals.append((truth[:, eligible] - adaptive[:, eligible]).T)
            weights.append(np.full(count, 1.0 / count))
        print(f"{year}: {len(dates) - before} eligible training acquisitions", flush=True)
    counts = [n for _, n in dates]
    return (dates, np.repeat(np.array([d for d, _ in dates], dtype="datetime64[D]"), counts),
            np.concatenate(feature_rows), np.concatenate(bases), np.concatenate(residuals), np.concatenate(weights))


def development_splits(history, days):
    """Select model families using date blocks within the training archive."""
    if history == "twenty_full":
        yield "2023", days < np.datetime64("2023-01-01"), (days >= np.datetime64("2023-01-01")) & (days < np.datetime64("2024-01-01"))
        return
    in_2024 = (days >= np.datetime64("2024-01-01")) & (days < np.datetime64("2025-01-01"))
    for month in DEVELOPMENT_MONTHS:
        first = date(2024, month, 1)
        last = date(2024, month + 1, 1) - timedelta(days=1)
        validation = (days >= np.datetime64(first)) & (days <= np.datetime64(last))
        outside = (days < np.datetime64(first - timedelta(days=WINDOW))) | (days > np.datetime64(last + timedelta(days=WINDOW)))
        yield first.strftime("%Y-%m"), in_2024 & outside, validation


def validation_ridge(x, y, weights):
    predictors = np.column_stack((np.ones(len(x)), x[:, :BASE_COLUMNS]))
    weighted = predictors * (weights / weights.sum())[:, None]
    penalty = np.eye(BASE_COLUMNS + 1) * 0.001
    penalty[0, 0] = 0
    return np.linalg.solve(weighted.T @ predictors + penalty, weighted.T @ y)


def ndvi_rows(values):
    return ndvi(values.T[:, None, :]).ravel()


def choose_family(history, days, x, base, residual, weights, workers):
    """Rank fixed candidates by pooled NDVI RMSE across development blocks."""
    folds = []
    squared_errors = dict.fromkeys(("ridge", *ALGORITHMS), 0.0)
    pixel_count = 0
    for name, train, validation in development_splits(history, days):
        if not train.any() or not validation.any() or (train & validation).any():
            raise ValueError(f"Invalid development fold {history} {name}")
        x_train, y_train, w_train = x[train], residual[train], weights[train]
        x_val, base_val = x[validation], base[validation]
        target = ndvi_rows(base_val + residual[validation])
        beta = validation_ridge(x_train, y_train, w_train)
        linear = ndvi_rows(base_val + np.column_stack((np.ones(len(x_val)), x_val[:, :BASE_COLUMNS])) @ beta)
        common = np.isfinite(target) & np.isfinite(linear)
        count = int(common.sum())
        if not count:
            raise ValueError(f"No eligible NDVI comparison pixels in {history} {name}")
        squared = {"ridge": float(np.square(linear[common] - target[common]).sum())}
        for algorithm in ALGORITHMS:
            start = perf_counter()
            model = fit_model(algorithm, x_train, y_train, w_train, workers)
            pred = ndvi_rows(model.predict_rows(x_val, base_val))
            pred = np.where(np.isfinite(pred), pred, linear)
            squared[algorithm] = float(np.square(pred[common] - target[common]).sum())
            print(f"{history} development {name} {algorithm}: RMSE {(squared[algorithm] / count) ** .5:.5f}, "
                  f"{perf_counter() - start:.1f}s", flush=True)
        folds.append({"block": name, "training_dates": np.unique(days[train]).astype(str).tolist(),
                      "validation_dates": np.unique(days[validation]).astype(str).tolist(), "pixels": count,
                      "rmse": {key: (value / count) ** 0.5 for key, value in squared.items()}})
        for key, value in squared.items():
            squared_errors[key] += value
        pixel_count += count
    pooled = {key: (value / pixel_count) ** 0.5 for key, value in squared_errors.items()}
    winner = min(ALGORITHMS, key=pooled.__getitem__)
    return {"folds": folds, "pooled_ndvi_rmse": pooled, "selected_nonlinear": winner}


def run(cache, baseline_path, work, workers):
    from joblib import dump, load
    import sklearn
    from threadpoolctl import threadpool_limits

    baseline = json.loads((baseline_path / "run.json").read_text(encoding="utf-8"))
    fingerprints = {str(year): sha256(cache / f"clayton_{year}.npz") for year in range(2005, 2026)}
    if fingerprints != baseline["input_sha256"]:
        raise ValueError("Use the exact annual input caches from the existing linear fit")
    dates, days, x, base, residual, weights = collect_examples(cache)
    models = {name: HistoryCorrection.load(baseline_path / f"{name}_spatial_model.npz") for name in HISTORIES}
    methods = BASELINES | {name: f"Ridge, {label}" for name, label in HISTORIES.items()}
    fallback, development, training, selected = {}, {}, {}, {}
    with threadpool_limits(limits=workers):
        for history, label in HISTORIES.items():
            long = history == "twenty_full"
            use = np.ones(len(days), dtype=bool) if long else days >= np.datetime64("2024-01-01")
            date_counts = {day: count for day, count in dates if long or day.startswith("2024-")}
            ridge = models[history]
            if tuple(date_counts) != ridge.training_dates or sum(date_counts.values()) != ridge.training_pixel_dates:
                raise ValueError(f"Training support differs from the original ridge for {history}")
            ridge.save(work / f"{history}_spatial_model.npz")
            development[history] = choose_family(history, days, x, base, residual, weights, workers)
            selected[history] = f"{history}_{development[history]['selected_nonlinear']}"
            training[history] = {"dates": date_counts, "pixels": int(use.sum()), "candidates": {}}
            x_train, y_train, w_train = x[use], residual[use], weights[use]
            for algorithm, title in ALGORITHMS.items():
                key = f"{history}_{algorithm}"
                start = perf_counter()
                model = fit_model(algorithm, x_train, y_train, w_train, workers)
                path = work / f"{key}.joblib"
                dump(model, path, compress=3)
                models[key] = load(path)
                fallback[key] = history
                methods[key] = f"{title}, {label}"
                seconds = perf_counter() - start
                training[history]["candidates"][algorithm] = {"seconds": seconds, "file": path.name, "bytes": path.stat().st_size}
                print(f"{history} final fit {algorithm}: {seconds:.1f}s", flush=True)
        rows, samples, skipped = evaluate(cache, work, models, linear_fallbacks=fallback)

    results = {}
    for pattern, entries in rows.items():
        for current, previous in zip(entries, baseline["patterns"][pattern]["dates"], strict=True):
            if (current["date"], current["reference"]) != (previous["date"], previous["reference"]):
                raise ValueError(f"The test date or its reference changed in the {pattern} comparison")
            for variable, score in current["scores"].items():
                expected = previous["scores"][variable]
                if score["common_pixels"] != expected["common_pixels"]:
                    raise ValueError(f"Common scoring support changed on {current['date']} for {pattern} {variable}")
                for history in HISTORIES:
                    if score["methods"][history]["rmse"] != expected["methods"][history]["rmse"]:
                        raise ValueError(f"Original ridge predictions changed on {current['date']}")
        # Main comparison uses the long-history validation winner against the corresponding ridge.
        scores = summarize(entries, samples[pattern], methods, "twenty_full", selected["twenty_full"])
        scores["test_best_nonlinear"] = {history: min((f"{history}_{name}" for name in ALGORITHMS),
            key=lambda key: scores["summary"]["NDVI"][key]["rmse"]) for history in HISTORIES}
        scores["linear_fallback_pixels"] = {name: sum(row["linear_fallback_pixels"][name] for row in entries) for name in fallback}
        results[pattern] = {"march_october": scores, "dates": entries}
        report_dir = work / pattern
        report_dir.mkdir()
        write_metrics(report_dir, entries)
        density_figure(work, pattern, samples[pattern],
                       ("twenty_full", selected["twenty_full"], selected["one_season"]), methods, "March–October")
        print(f"{pattern}: original 2025 scoring support verified on {len(entries)} dates", flush=True)
    report = {
        "grid": GRID, "methods": methods, "patterns": results, "training": training, "development": development,
        "sklearn_version": sklearn.__version__, "input_sha256": fingerprints,
        "baseline_manifest_sha256": sha256(baseline_path / "run.json"), "test_skipped": skipped,
        "settings": baseline["settings"] | {"primary_comparison": [selected["twenty_full"], "twenty_full"]},
        "nonlinear_comparison": {"algorithms": ALGORITHMS, "histories": HISTORIES, "selected": selected,
                                 "fit_target": "NIR and red residuals against adaptive correction, weighted equally by date",
                                 "selection": "Minimum pooled development NDVI RMSE among fixed nonlinear configurations",
                                 "fallbacks": fallback, "fallback_rule": "Use corresponding ridge for unsupported nonlinear NDVI"},
    }
    (work / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=SPRING_FALL_CACHE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/spatial_validation/clayton_nonlinear_2005_2025")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.workers < 1:
        parser.error("Choose a new output directory and a positive worker count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".nonlinear_", dir=args.output.parent) as temporary:
        staging = Path(temporary) / args.output.name
        staging.mkdir()
        run(args.cache, args.baseline, staging, args.workers)
        staging.rename(args.output)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
