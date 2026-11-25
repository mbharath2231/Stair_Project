#!/usr/bin/env python3
"""Tune Clayton's spatial CNN on forward-chaining Landsat acquisition folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import SPRING_FALL_CACHE
from scripts.compare_spatial_cnn import (
    DEFAULT_BENCHMARK, DEFAULT_OUTPUT, HISTORIES, LONG_BOOST, LONG_CNN, SEED,
    check_test_support, collect_scenes, development_predictions, development_targets,
    epoch_models, load_benchmarks, ndvi_diagnostics,
)
from scripts.compare_spatial_history import evaluate, summarize
from scripts.compare_training_history import sha256, write_metrics
from stair.cnn_spatial import SpatialCNN

FOLD_YEARS = (2022, 2023, 2024)
LEARNING_RATES = (0.001, 0.002)
EPOCHS = (4, 8, 13)
TUNED = "twenty_full_cnn_tuned"


def forward_folds(days):
    for year in FOLD_YEARS:
        start, end = np.datetime64(str(year)), np.datetime64(str(year + 1))
        train = np.flatnonzero(days < start)
        validation = np.flatnonzero((days >= start) & (days < end))
        if not len(train) or not len(validation):
            raise ValueError(f"Missing training or validation acquisitions for {year}")
        yield year, train, validation


def tune(scenes, device, batch_size):
    folds = []
    for year, train, validation in forward_folds(scenes.days):
        targets, ridge, common = development_targets(scenes, train, validation)
        pixels = sum(map(len, targets))
        if not pixels:
            raise ValueError(f"No common NDVI validation pixels in {year}")
        fold = {
            "year": year,
            "training_dates": scenes.days[train].astype(str).tolist(),
            "validation_dates": scenes.days[validation].astype(str).tolist(),
            "scored_pixels": pixels,
            "ridge_mae": sum(float(np.abs(guess - target).sum())
                             for guess, target in zip(ridge, targets, strict=True)) / pixels,
            "candidates": [],
        }
        for rate in LEARNING_RATES:
            for epoch, model in epoch_models(scenes, train, EPOCHS[-1], device, batch_size, learning_rate=rate):
                if epoch not in EPOCHS:
                    continue
                guesses = development_predictions(scenes, model, validation, ridge, common, batch_size)
                residual = np.concatenate([guess - target for guess, target in zip(guesses, targets, strict=True)])
                mae = float(np.abs(residual).mean())
                rmse = float(np.sqrt(np.square(residual).mean()))
                print(f"Validation {year}, rate {rate:g}, epochs {epoch}: MAE {mae:.6f}", flush=True)
                fold["candidates"].append({"learning_rate": rate, "epochs": epoch, "mae": mae, "rmse": rmse})
        folds.append(fold)
    results = []
    for candidates in zip(*(fold["candidates"] for fold in folds), strict=True):
        maes = [candidate["mae"] for candidate in candidates]
        results.append({"learning_rate": candidates[0]["learning_rate"], "epochs": candidates[0]["epochs"],
                        "mean_fold_mae": float(np.mean(maes)), "fold_mae_sd": float(np.std(maes, ddof=1))})
    chosen = min(results, key=lambda row: (row["mean_fold_mae"], row["epochs"], row["learning_rate"]))
    return {"folds": folds, "candidates": results, "selected": chosen}


def agreement_diagnostics(samples, methods):
    """Describe agreement with the identity line and bias at low/high reference NDVI."""
    reference = np.concatenate([sample["NDVI"]["truth"] for sample in samples])
    scores = {}
    for method in methods:
        prediction = np.concatenate([sample["NDVI"][method] for sample in samples])
        error = prediction - reference
        covariance = float(np.mean((reference - reference.mean()) * (prediction - prediction.mean())))
        scale = reference.var() + prediction.var() + error.mean() ** 2
        scores[method] = {
            "lin_concordance": float(2 * covariance / scale) if scale else None,
            "signed_error_p05_p95": np.quantile(error, [0.05, 0.95]).tolist(),
            "bias_by_reference": {
                label: {"pixels": int(mask.sum()), "mean_error": float(error[mask].mean()) if mask.any() else None}
                for label, mask in {
                    "ndvi_below_0_25": reference < 0.25,
                    "ndvi_at_least_0_65": reference >= 0.65,
                }.items()
            },
        }
    return scores


def paired_mae_interval(rows, samples, baseline):
    """Resample paired month totals so neighboring pixels aren't treated as independent."""
    monthly = {}
    for row, sample in zip(rows, samples, strict=True):
        ndvi = sample["NDVI"]
        target = ndvi["truth"]
        change = np.abs(ndvi[TUNED] - target) - np.abs(ndvi[baseline] - target)
        month = row["date"][:7]
        totals = monthly.setdefault(month, np.zeros(2))
        totals += (change.sum(), len(change))
    totals = np.stack(list(monthly.values()))
    draw = np.random.default_rng(20261118).integers(len(totals), size=(5000, len(totals)))
    bootstrap = totals[draw].sum(axis=1)
    difference, sample_count = totals.sum(axis=0)
    return {"mae_difference": float(difference / sample_count),
            "percentile_95_interval": np.quantile(bootstrap[:, 0] / bootstrap[:, 1], [0.025, 0.975]).tolist(),
            "month_blocks": len(totals)}


def run(args, work):
    benchmark = json.loads((args.benchmark / "run.json").read_text(encoding="utf-8"))
    cnn_baseline = json.loads((args.cnn_baseline / "run.json").read_text(encoding="utf-8"))
    hashes = {str(year): sha256(args.cache / f"clayton_{year}.npz") for year in range(2005, 2026)}
    if hashes != benchmark["input_sha256"] or hashes != cnn_baseline["input_sha256"]:
        raise ValueError("Use the identical annual inputs from the original CNN benchmark")
    torch.set_num_threads(args.workers)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    scenes = collect_scenes(args.cache)
    counts = dict(zip(scenes.days.astype(str), scenes.mask.sum(axis=(1, 2)).tolist(), strict=True))
    if counts != benchmark["training"]["twenty_full"]["dates"]:
        raise ValueError("Training acquisition counts differ from the fixed benchmark")
    selection = tune(scenes, device, args.batch_size)
    chosen = selection["selected"]
    selection["selection_metric"] = "Mean of the three validation seasons' pooled NDVI MAEs"
    (work / "cross_validation.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Locked settings before scoring 2025: rate {chosen['learning_rate']:g}, {chosen['epochs']} epochs", flush=True)
    for _, tuned in epoch_models(scenes, np.arange(len(scenes.days)), chosen["epochs"], device, args.batch_size,
                                 learning_rate=chosen["learning_rate"]):
        pass
    del scenes
    torch.save(tuned.state_dict(), work / f"{TUNED}.pt")
    models = load_benchmarks(args.benchmark, benchmark)
    fallback = dict(benchmark["nonlinear_comparison"]["fallbacks"])
    for history in HISTORIES:
        key = f"{history}_cnn"
        original = SpatialCNN()
        original.load_state_dict(torch.load(args.cnn_baseline / f"{key}.pt", map_location=device, weights_only=True))
        models[key] = original.eval()
        fallback[key] = history
    models[TUNED] = tuned
    fallback[TUNED] = "twenty_full"
    methods = cnn_baseline["methods"] | {TUNED: "CNN tuned by forward-chaining, 2005–2024"}
    rows, samples, skipped = evaluate(args.cache, work, models, linear_fallbacks=fallback, save_rasters=False)
    check_test_support(rows, cnn_baseline)
    comparisons = ("adaptive", "twenty_full", LONG_BOOST, LONG_CNN, TUNED)
    report = {
        "input_sha256": hashes, "prior_cnn_manifest_sha256": sha256(args.cnn_baseline / "run.json"),
        "torch_version": torch.__version__, "seed": SEED, "batch_size": args.batch_size, "months": "March–October",
        "test_year": 2025, "test_skipped": skipped, "methods": methods, "selected": chosen, "patterns": {},
    }
    for pattern, entries in rows.items():
        score = summarize(entries, samples[pattern], methods, LONG_BOOST, TUNED)
        score["ndvi_diagnostics"] = ndvi_diagnostics(samples[pattern], methods)
        score["ndvi_agreement"] = agreement_diagnostics(samples[pattern], comparisons)
        score["paired_mae_intervals"] = {name: paired_mae_interval(entries, samples[pattern], name)
                                         for name in comparisons if name != TUNED}
        report["patterns"][pattern] = score
        folder = work / pattern
        folder.mkdir()
        write_metrics(folder, entries)
        print(f"{pattern}: scored {len(entries)} acquisitions on the original 2025 support", flush=True)
    (work / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=SPRING_FALL_CACHE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--cnn-baseline", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT.with_name("clayton_cnn_rolling_2005_2025"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or min(args.workers, args.batch_size) < 1:
        parser.error("Choose a new output directory and positive worker and batch counts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cnn_rolling_", dir=args.output.parent) as temporary:
        staging = Path(temporary) / args.output.name
        staging.mkdir()
        run(args, staging)
        staging.rename(args.output)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
