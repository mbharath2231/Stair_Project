#!/usr/bin/env python3
"""Compare a convolutional residual correction with the existing Clayton holdouts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import GRID, SPRING_FALL_CACHE
from scripts.compare_nonlinear_spatial import HISTORIES, development_splits, ndvi_rows, validation_ridge
from scripts.compare_spatial_history import BASELINES, density_figure, evaluate, scenarios, summarize
from scripts.compare_training_history import sha256, write_metrics
from stair.cnn_spatial import SpatialCNN, cnn_features
from stair.history_model import HistoryCorrection
from stair.nonlinear_spatial import BASE_COLUMNS, TARGET_SCALE

DEFAULT_BENCHMARK = ROOT / "outputs/spatial_validation/clayton_nonlinear_2005_2025"
DEFAULT_OUTPUT = ROOT / "outputs/spatial_validation/clayton_cnn_2005_2025"
LONG_BOOST = "twenty_full_spatial_boosting"
LONG_CNN = "twenty_full_cnn"
SEED = 2024


@dataclass
class TrainingScenes:
    days: np.ndarray
    predictors: np.ndarray
    adaptive: np.ndarray
    truth: np.ndarray
    mask: np.ndarray


def collect_scenes(cache):
    """Mask the target before constructing CNN inputs from STAIR and its reference."""
    collected = []
    for year in range(2005, 2025):
        start = len(collected)
        for day, _, holdout, truth, _, reference, _, _, stages in scenarios(cache, year, ("cloud",), []):
            adaptive = stages["adaptive"]
            predictors, supported = cnn_features(adaptive, reference, day)
            mask = holdout & supported & np.isfinite(truth).all(axis=0)
            if mask.any():
                collected.append((day, predictors, adaptive.astype(np.float32), truth.astype(np.float32), mask))
        print(f"{year}: {len(collected) - start} training acquisitions", flush=True)
    if not collected:
        raise ValueError("No valid training examples")
    days, predictors, adaptive, truth, mask = zip(*collected)
    return TrainingScenes(np.array(days, dtype="datetime64[D]"), np.stack(predictors), np.stack(adaptive),
                          np.stack(truth), np.stack(mask))


def epoch_models(scenes, train, epochs, device, batch_size, *, learning_rate=0.001):
    """Shuffle scenes and give each acquisition equal weight in the band-level loss."""
    torch.manual_seed(SEED)
    generator = np.random.default_rng(SEED)
    model = SpatialCNN().to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    eligible = scenes.mask[train]
    predictors = scenes.predictors[train]
    residual = (scenes.truth[train] - scenes.adaptive[train]) / TARGET_SCALE[:, None, None]
    residual = np.where(eligible[:, None], residual, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(len(train))
        for start in range(0, len(train), batch_size):
            indices = order[start:start + batch_size]
            inputs = torch.from_numpy(predictors[indices]).to(device=device, memory_format=torch.channels_last)
            target = torch.from_numpy(residual[indices]).to(device)
            mask = torch.from_numpy(eligible[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            squared_errors = (model(inputs) - target).square().mean(dim=1)
            per_scene = (squared_errors * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
            (per_scene.sum() / batch_size).backward()
            optimizer.step()
        yield epoch, model.eval()


def development_targets(scenes, train, validation):
    """Lock which held-out development pixels can be scored and their ridge fallback."""
    xs, targets, weights = [], [], []
    for i in train:
        mask = scenes.mask[i]
        xs.append(scenes.predictors[i, :BASE_COLUMNS][:, mask].T)
        targets.append((scenes.truth[i] - scenes.adaptive[i])[:, mask].T)
        weights.append(np.full(mask.sum(), 1 / mask.sum()))
    beta = validation_ridge(np.concatenate(xs), np.concatenate(targets), np.concatenate(weights))
    truth_arrays, ridge_arrays, masks = [], [], []
    for i in validation:
        mask = scenes.mask[i]
        x = scenes.predictors[i, :BASE_COLUMNS][:, mask].T
        base = scenes.adaptive[i][:, mask].T
        observed = ndvi_rows(scenes.truth[i][:, mask].T)
        ridge = ndvi_rows(base + np.column_stack((np.ones(len(x)), x)) @ beta)
        common = np.isfinite(observed) & np.isfinite(ridge)
        masks.append(common)
        truth_arrays.append(observed[common])
        ridge_arrays.append(ridge[common])
    return truth_arrays, ridge_arrays, masks


def development_predictions(scenes, model, validation, ridge, common, batch_size):
    """Score on the same eligible validation pixels with the fixed ridge fallback."""
    for start in range(0, len(validation), batch_size):
        indices = validation[start:start + batch_size]
        predicted = model.corrected(scenes.predictors[indices], scenes.adaptive[indices])
        for position, (i, image) in enumerate(zip(indices, predicted, strict=True), start):
            guess = ndvi_rows(image[:, scenes.mask[i]].T)[common[position]]
            yield np.where(np.isfinite(guess), guess, ridge[position])


def select_epochs(scenes, history, epochs, device, batch_size):
    folds = []
    for label, train_mask, validation_mask in development_splits(history, scenes.days):
        train, validation = np.flatnonzero(train_mask), np.flatnonzero(validation_mask)
        if not len(train) or not len(validation) or (train_mask & validation_mask).any():
            raise ValueError(f"Invalid development split {history} {label}")
        targets, ridge, common = development_targets(scenes, train, validation)
        pixel_count = sum(map(len, targets))
        if not pixel_count:
            raise ValueError(f"No common NDVI support in {history} {label}")
        errors = []
        for epoch, model in epoch_models(scenes, train, epochs, device, batch_size):
            predictions = development_predictions(scenes, model, validation, ridge, common, batch_size)
            squared = sum(float(np.square(guess - target).sum())
                          for guess, target in zip(predictions, targets, strict=True))
            errors.append(squared)
            print(f"{history} {label} epoch {epoch:02}: NDVI RMSE {(squared / pixel_count) ** .5:.5f}", flush=True)
        folds.append({"block": label, "training_dates": scenes.days[train].astype(str).tolist(),
                      "validation_dates": scenes.days[validation].astype(str).tolist(),
                      "pixels": pixel_count, "squared_errors": errors})
    pixels = sum(fold["pixels"] for fold in folds)
    rmse = np.sqrt(np.sum([fold["squared_errors"] for fold in folds], axis=0) / pixels)
    return {"folds": folds, "ndvi_rmse_by_epoch": rmse.tolist(), "selected_epochs": int(rmse.argmin()) + 1}


def load_benchmarks(benchmark, manifest):
    from joblib import load
    import sklearn

    if sklearn.__version__ != manifest["sklearn_version"]:
        raise ValueError(f"Use scikit-learn {manifest['sklearn_version']} to reproduce the saved boosting models")
    models = {name: HistoryCorrection.load(benchmark / f"{name}_spatial_model.npz") for name in HISTORIES}
    for history, fit in manifest["training"].items():
        models.update({f"{history}_{algorithm}": load(benchmark / candidate["file"])
                       for algorithm, candidate in fit["candidates"].items()})
    return models


def check_test_support(rows, manifest):
    """Ensure CNN inclusion does not change any preceding comparison's scored pixels."""
    for pattern, dates in rows.items():
        for current, previous in zip(dates, manifest["patterns"][pattern]["dates"], strict=True):
            if (current["date"], current["reference"]) != (previous["date"], previous["reference"]):
                raise ValueError(f"Reference or date changed for {pattern}")
            for variable, score in current["scores"].items():
                expected = previous["scores"][variable]
                if score["common_pixels"] != expected["common_pixels"]:
                    raise ValueError(f"Scoring support changed for {pattern} {current['date']} {variable}")
                for name, values in expected["methods"].items():
                    np.testing.assert_allclose(score["methods"][name]["rmse"], values["rmse"], rtol=0, atol=1e-7)


def ndvi_diagnostics(samples, methods):
    truth = np.concatenate([sample["NDVI"]["truth"] for sample in samples])
    diagnostics = {}
    for method in methods:
        predicted = np.concatenate([sample["NDVI"][method] for sample in samples])
        r = float(np.corrcoef(truth, predicted)[0, 1])
        spread = float(predicted.std() / truth.std())
        ratio = float(predicted.mean() / truth.mean())
        diagnostics[method] = {
            "large_error_fraction_over_0_10": float((np.abs(predicted - truth) > 0.10).mean()),
            "pbias_percent_predicted_minus_observed": 100 * (ratio - 1),
            "nse": float(1 - np.mean((predicted - truth) ** 2) / truth.var()),
            "kge_2009": float(1 - np.linalg.norm(np.array([r, spread, ratio]) - 1)),
            "pearson_r": r, "std_ratio": spread, "mean_ratio": ratio,
        }
    return diagnostics


def run(cache, benchmark, work, args):
    manifest = json.loads((benchmark / "run.json").read_text(encoding="utf-8"))
    fingerprints = {str(year): sha256(cache / f"clayton_{year}.npz") for year in range(2005, 2026)}
    if fingerprints != manifest["input_sha256"]:
        raise ValueError("Use the same historical inputs as the nonlinear benchmark")
    models = load_benchmarks(benchmark, manifest)
    torch.set_num_threads(args.workers)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    scenes = collect_scenes(cache)
    methods = dict(manifest["methods"])
    fallback = dict(manifest["nonlinear_comparison"]["fallbacks"])
    development, fits = {}, {}
    for history, label in HISTORIES.items():
        train = np.flatnonzero(scenes.days >= np.datetime64("2024-01-01")) if history == "one_season" else np.arange(len(scenes.days))
        counts = dict(zip(scenes.days[train].astype(str), scenes.mask[train].sum(axis=(1, 2)).tolist(), strict=True))
        if counts != manifest["training"][history]["dates"]:
            raise ValueError(f"The training dates or pixel counts changed for {history}")
        development[history] = select_epochs(scenes, history, args.epochs, device, args.batch_size)
        epochs = development[history]["selected_epochs"]
        print(f"{history}: refitting with {len(train)} acquisitions for {epochs} epochs", flush=True)
        for _, model in epoch_models(scenes, train, epochs, device, args.batch_size):
            pass
        key = f"{history}_cnn"
        checkpoint = work / f"{key}.pt"
        torch.save(model.state_dict(), checkpoint)
        loaded = SpatialCNN().to(device)
        loaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        models[key] = loaded.eval()
        methods[key] = f"CNN, {label}"
        fallback[key] = history
        fits[history] = {"dates": counts, "epochs": epochs, "file": checkpoint.name}
    del scenes
    print("Scoring the 2025 cloud and stripe benchmarks", flush=True)
    rows, samples, skipped = evaluate(cache, work, models, linear_fallbacks=fallback)
    check_test_support(rows, manifest)
    results = {}
    for pattern, entries in rows.items():
        scores = summarize(entries, samples[pattern], methods, LONG_BOOST, LONG_CNN)
        scores["ndvi_diagnostics"] = ndvi_diagnostics(samples[pattern], methods)
        scores["cnn_fallback_pixels"] = {
            key: sum(row["linear_fallback_pixels"][key] for row in entries)
            for key in ("one_season_cnn", LONG_CNN)
        }
        results[pattern] = {"march_october": scores, "dates": entries}
        pattern_dir = work / pattern
        pattern_dir.mkdir()
        write_metrics(pattern_dir, entries)
        density_figure(work, pattern, samples[pattern], ("twenty_full", LONG_BOOST, LONG_CNN), methods, "March–October")
        print(f"{pattern}: matched the original scoring support on {len(entries)} dates", flush=True)
    report = {
        "grid": GRID, "methods": methods, "patterns": results, "input_sha256": fingerprints,
        "baseline_manifest_sha256": sha256(benchmark / "run.json"), "test_skipped": skipped,
        "settings": manifest["settings"] | {"primary_comparison": [LONG_CNN, LONG_BOOST]},
        "cnn_comparison": {"development": development, "fits": fits, "seed": SEED, "device": str(device),
                           "batch_size": args.batch_size, "max_epochs": args.epochs, "torch_version": torch.__version__},
    }
    (work / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=SPRING_FALL_CACHE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=16, help="Maximum epochs to compare on development dates")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists() or min(args.epochs, args.batch_size, args.workers) < 1:
        parser.error("Choose a new output directory and positive training settings")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cnn_", dir=args.output.parent) as temporary:
        staging = Path(temporary) / args.output.name
        staging.mkdir()
        run(args.cache, args.benchmark, staging, args)
        staging.rename(args.output)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
