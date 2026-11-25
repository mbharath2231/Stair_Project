#!/usr/bin/env python3
"""Test Landsat recurrent residual correction on single and consecutive gaps."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import GRID, SPRING_FALL_CACHE
from scripts.compare_spatial_cnn import (
    DEFAULT_BENCHMARK, DEFAULT_OUTPUT, HISTORIES, LONG_BOOST, SEED, TrainingScenes,
    check_test_support, development_targets, load_benchmarks, ndvi_diagnostics, ndvi_rows,
)
from scripts.compare_spatial_history import (
    MIN_TARGET_CLEAR, PATTERNS, evaluate, scenarios_for_images, summarize,
)
from scripts.compare_training_history import sha256, write_metrics
from scripts.export_stair_stages import WINDOW
from scripts.run_daily_ndvi import read_cache
from scripts.run_watershed_fusion import validity
from scripts.tune_spatial_cnn import TUNED, forward_folds
from stair.cnn_spatial import SpatialCNN, cnn_features
from stair.ndvi import ndvi
from stair.nonlinear_spatial import TARGET_SCALE
from stair.recurrent_spatial import KERNELS, RecurrentCorrection, sequence_frames
from stair.validation import make_holdout

EPOCHS = (2, 4, 6)
RECURRENT_LABELS = {"pixel_lstm": "Pixel LSTM, 2005–2024", "conv_lstm": "ConvLSTM, 2005–2024"}
MAP_LAYERS = ("truth", "masked", "temporal", "global", "adaptive", "twenty_full", LONG_BOOST, TUNED,
              *RECURRENT_LABELS)


def collect_training(cache):
    collected, sequences = [], []
    for year in range(2005, 2025):
        _, images, _ = read_cache(cache / f"clayton_{year}.npz", year)
        start = len(collected)
        for day, _, holdout, truth, _, reference, _, _, stages in scenarios_for_images(images, ("cloud",), []):
            adaptive = stages["adaptive"]
            static, valid = cnn_features(adaptive, reference, day)
            supported = holdout & valid & np.isfinite(truth).all(axis=0)
            if supported.any():
                collected.append((day, static, adaptive.astype(np.float32), truth.astype(np.float32), supported))
                sequence, _ = sequence_frames(images, day, window=WINDOW)
                sequences.append(sequence)
        print(f"{year}: {len(collected) - start} training acquisitions", flush=True)
    if not collected:
        raise ValueError("No available training acquisitions")
    dates, static, adaptive, truth, valid = zip(*collected)
    return (TrainingScenes(np.array(dates, dtype="datetime64[D]"), np.stack(static), np.stack(adaptive),
                           np.stack(truth), np.stack(valid)), sequences)


def network_epochs(kind, scenes, frames, indices, epochs, batch_size):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = RecurrentCorrection(kind).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    weights = TARGET_SCALE[:, None, None]
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(indices)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            mask = scenes.mask[batch]
            residual = np.where(mask[:, None], (scenes.truth[batch] - scenes.adaptive[batch]) / weights, 0)
            sequence = torch.from_numpy(np.stack([frames[i] for i in batch]))
            static = torch.from_numpy(scenes.predictors[batch]).contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            squared = (model(sequence, static) - torch.from_numpy(residual)).square().mean(dim=1)
            selected = torch.from_numpy(mask)
            acquisition_losses = (squared * selected).sum(dim=(1, 2)) / selected.sum(dim=(1, 2))
            (acquisition_losses.sum() / batch_size).backward()
            optimizer.step()
        yield epoch, model.eval()


def tune(kind, scenes, frames, batch_size, prepared):
    folds = []
    for year, train, valid, (targets, ridge, common) in prepared:
        n = sum(map(len, targets))
        scores = {}
        for epoch, model in network_epochs(kind, scenes, frames, train, EPOCHS[-1], batch_size):
            if epoch not in EPOCHS:
                continue
            abs_sum = 0.0
            for start in range(0, len(valid), batch_size):
                indices = valid[start:start + batch_size]
                predicted = model.corrected(np.stack([frames[i] for i in indices]),
                                            scenes.predictors[indices], scenes.adaptive[indices])
                for position, (i, image) in enumerate(zip(indices, predicted, strict=True), start):
                    prediction = ndvi_rows(image[:, scenes.mask[i]].T)[common[position]]
                    prediction = np.where(np.isfinite(prediction), prediction, ridge[position])
                    abs_sum += float(np.abs(prediction - targets[position]).sum())
            scores[epoch] = abs_sum / n
            print(f"{kind} validation {year}, epoch {epoch}: MAE {scores[epoch]:.6f}", flush=True)
        folds.append({"year": year, "training_dates": scenes.days[train].astype(str).tolist(),
                      "validation_dates": scenes.days[valid].astype(str).tolist(), "pixels": n, "mae_by_epoch": scores})
    means = {epoch: float(np.mean([row["mae_by_epoch"][epoch] for row in folds])) for epoch in EPOCHS}
    return {"folds": folds, "mean_fold_mae": means, "selected_epochs": min(EPOCHS, key=means.__getitem__)}


class SequencePredictor:
    """Provide the existing STAIR evaluator with a date-aware residual model."""

    def __init__(self, model, images, exclusions=None):
        self.model = model
        self.images = images
        self.exclusions = exclusions or {}

    def predict(self, adaptive, reference, day):
        frames, _ = sequence_frames(self.images, day, window=WINDOW, excluded=self.exclusions.get(day, ()))
        static, valid = cnn_features(adaptive, reference, day)
        corrected = self.model.corrected(frames[None], static[None], adaptive[None])[0]
        return np.where(valid, corrected, np.nan)


def consecutive_blocks(images, size):
    """Hold out successive season acquisitions in disjoint blocks."""
    dates = [day for day in sorted(images) if 3 <= int(day[5:7]) <= 10]
    iterator = iter(dates)
    blocks = []
    while len(block := tuple(islice(iterator, size))) == size:
        blocks.append(block)
    return blocks


def evaluate_run(args, work, images, baselines, networks, methods, fallback, *, size=None):
    skips = []
    blocks = consecutive_blocks(images, size) if size else []
    exclusions = {day: group for group in blocks for day in group} if size else None
    patterns = ("cloud",) if size else PATTERNS
    models = baselines | {name: SequencePredictor(model, images, exclusions) for name, model in networks.items()}
    cases = scenarios_for_images(images, patterns, skips, excluded_by_target=exclusions)
    rows, samples, _ = evaluate(args.cache, work, models, linear_fallbacks=fallback, save_rasters=not size,
                                cases=cases, export_methods=MAP_LAYERS)
    results = {}
    for pattern in patterns:
        dates = rows[pattern]
        if not dates:
            raise ValueError(f"No {pattern} dates can be evaluated with a gap of {size or 1} acquisitions")
        scores = summarize(dates, samples[pattern], methods, LONG_BOOST, "conv_lstm")
        scores["ndvi_diagnostics"] = ndvi_diagnostics(samples[pattern], methods)
        scores["recurrent_fallback_pixels"] = {
            name: sum(row["linear_fallback_pixels"][name] for row in dates) for name in networks
        }
        if exclusions is not None:
            mask = make_holdout(next(iter(images.values())).shape[1:], pattern)
            eligible = [day for day in exclusions if validity(images[day]).mean() >= MIN_TARGET_CLEAR]
            ndvi_eligible = sum(int((np.isfinite(ndvi(images[day])[0]) & mask).sum()) for day in eligible)
            scores["eligible_dates_before_reference_checks"] = len(eligible)
            scores["eligible_ndvi_pixels_before_reference_checks"] = ndvi_eligible
            scores["coverage_including_skipped_days"] = {
                name: sum(row["scores"]["NDVI"]["methods"][name]["covered_pixels"] for row in dates) / ndvi_eligible
                for name in methods
            }
        folder = work / (f"consecutive_{size}" if size else pattern)
        folder.mkdir()
        write_metrics(folder, dates)
        results[pattern] = {"march_october": scores, "dates": dates}
        print(f"{pattern}, gap length {size or 1}: {len(dates)} scenarios passed the reference checks", flush=True)
    return results, skips, blocks


def run(args, work):
    baseline = json.loads((args.benchmark / "run.json").read_text(encoding="utf-8"))
    cnn_run = json.loads((args.cnn / "run.json").read_text(encoding="utf-8"))
    tuned_run = json.loads((args.tuned_cnn / "run.json").read_text(encoding="utf-8"))
    hashes = {str(year): sha256(args.cache / f"clayton_{year}.npz") for year in range(2005, 2026)}
    if any(hashes != run["input_sha256"] for run in (baseline, cnn_run, tuned_run)):
        raise ValueError("Historical inputs differ from the established benchmarks")
    torch.set_num_threads(args.workers)
    torch.use_deterministic_algorithms(True)
    scenes, frames = collect_training(args.cache)
    date_counts = dict(zip(scenes.days.astype(str), scenes.mask.sum(axis=(1, 2)).tolist(), strict=True))
    if date_counts != baseline["training"]["twenty_full"]["dates"]:
        raise ValueError("Training dates or their pixel counts differ from the existing benchmark")
    prepared = [(year, train, valid, development_targets(scenes, train, valid))
                for year, train, valid in forward_folds(scenes.days)]
    indices = np.arange(len(scenes.days))
    networks, development = {}, {}
    for kind in KERNELS:
        info = tune(kind, scenes, frames, args.batch_size, prepared)
        development[kind] = info
        epochs = info["selected_epochs"]
        print(f"{kind}: locked {epochs} epochs and refitting with {len(indices)} acquisitions", flush=True)
        for _, model in network_epochs(kind, scenes, frames, indices, epochs, args.batch_size):
            pass
        path = work / f"{kind}.pt"
        torch.save(model.state_dict(), path)
        reloaded = RecurrentCorrection(kind)
        reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        networks[kind] = reloaded.eval()
    del scenes, frames, prepared
    (work / "cross_validation.json").write_text(json.dumps(development, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    models = load_benchmarks(args.benchmark, baseline)
    cnn_files = {f"{history}_cnn": args.cnn / f"{history}_cnn.pt" for history in HISTORIES}
    cnn_files[TUNED] = args.tuned_cnn / f"{TUNED}.pt"
    for key, path in cnn_files.items():
        model = SpatialCNN()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models[key] = model.eval()
    methods = cnn_run["methods"] | {TUNED: tuned_run["methods"][TUNED]} | RECURRENT_LABELS
    fallback = dict(baseline["nonlinear_comparison"]["fallbacks"])
    fallback.update({f"{history}_cnn": history for history in HISTORIES})
    fallback.update(dict.fromkeys((TUNED, *networks), "twenty_full"))
    _, images, _ = read_cache(args.cache / "clayton_2025.npz", 2025)
    regular, regular_skips, _ = evaluate_run(args, work, images, models, networks, methods, fallback)
    check_test_support({pattern: values["dates"] for pattern, values in regular.items()}, cnn_run)
    stress = {}
    for size in (2, 3):
        result, skips, blocks = evaluate_run(args, work, images, models, networks, methods, fallback, size=size)
        stress[str(size)] = {"blocks": blocks, "skipped": skips, **result["cloud"]}
    report = {
        "grid": GRID, "methods": methods, "input_sha256": hashes, "test_skipped": regular_skips,
        "patterns": regular, "consecutive_cloud_gaps": stress,
        "settings": cnn_run["settings"] | {"primary_comparison": ["conv_lstm", LONG_BOOST]},
        "recurrent_comparison": {"development": development, "training_dates": date_counts, "seed": SEED,
                                  "torch_version": torch.__version__, "batch_size": args.batch_size},
    }
    (work / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=SPRING_FALL_CACHE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--cnn", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tuned-cnn", type=Path, default=DEFAULT_OUTPUT.with_name("clayton_cnn_rolling_2005_2025"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT.with_name("clayton_recurrent_2005_2025"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or min(args.workers, args.batch_size) < 1:
        parser.error("Choose an unused output directory and positive worker and batch counts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".recurrent_", dir=args.output.parent) as temporary:
        staging = Path(temporary) / args.output.name
        staging.mkdir()
        run(args, staging)
        staging.rename(args.output)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
