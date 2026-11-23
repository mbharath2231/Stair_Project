#!/usr/bin/env python3
"""Refit the long spatial model with matched training support after Landsat harmonization."""
from __future__ import annotations

import argparse
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_landsat_provenance import DEFAULT_PROVENANCE, load_provenance
from scripts.collect_ndvi_inputs import GRID, SPRING_FALL_CACHE
from scripts.compare_spatial_history import BASELINES, density_figure, evaluate, scenarios_for_images, summarize
from scripts.compare_training_history import sha256, write_metrics
from scripts.run_daily_ndvi import read_cache
from scripts.run_watershed_fusion import validity
from stair.history_model import HistoryCorrection, RidgeFit
from stair.landsat_harmonization import COEFFICIENT_SOURCE, INTERCEPTS, SLOPES, harmonize_observations

DEFAULT_BASELINE = ROOT / "outputs/spatial_validation/clayton_full_season_history_2005_2025"
PREDICTIONS = {"one_season": "2024 season", "twenty_full": "20 years, native", "twenty_harmonized": "20 years, harmonized"}
METHODS = BASELINES | PREDICTIONS


def train_pair(cache, provenance):
    """Regenerate predictors on both scales and require identical per-date training masks."""
    fits = [RidgeFit(input_source="spatial"), RidgeFit(input_source="spatial")]
    masks = {}
    for year in range(2005, 2025):
        manifest, native, _ = read_cache(cache / f"clayton_{year}.npz", year)
        harmonized = harmonize_observations(manifest, native, provenance)
        for day, image in native.items():
            np.testing.assert_array_equal(np.isfinite(image), np.isfinite(harmonized[day]), err_msg=day)
        native_skips, harmonized_skips = [], []
        cases = scenarios_for_images(native, ("cloud",), native_skips)
        if all(harmonized[day] is image for day, image in native.items()):
            pairs = ((case, case) for case in cases)
        else:
            pairs = zip(cases, scenarios_for_images(harmonized, ("cloud",), harmonized_skips), strict=True)
        count = 0
        for original, adjusted in pairs:
            day, pattern, mask, _, _, _, reference_day, _, _ = original
            if (day, pattern, reference_day) != (adjusted[0], adjusted[1], adjusted[6]):
                raise ValueError(f"Harmonization changed target/reference selection on {day}")
            np.testing.assert_array_equal(mask, adjusted[2])
            samples = []
            for row in (original, adjusted):
                truth, reference, adaptive = row[3], row[5], row[-1]["adaptive"]
                support = mask & validity(truth) & validity(reference) & validity(adaptive)
                samples.append((support, adaptive, reference, truth))
            support = samples[0][0]
            np.testing.assert_array_equal(support, samples[1][0], err_msg=f"Training support differs on {day}")
            n = int(support.sum())
            if n:
                masks[day] = {"pixels": n, "sha256": hash_bytes(np.packbits(support).tobytes()).hexdigest(),
                              "reference": reference_day}
            for fit, (_, adaptive, reference, truth) in zip(fits, samples, strict=True):
                fit.add(day, adaptive, reference, np.where(support, truth, np.nan))
            count += bool(n)
        if harmonized_skips and native_skips != harmonized_skips:
            raise ValueError(f"Harmonization changed the exclusion list for {year}")
        print(f"{year}: {count} matched training dates", flush=True)
    return (fits[0].finish(), fits[1].finish()), masks


def run(cache, provenance_path, baseline_path, work):
    baseline = json.loads((baseline_path / "run.json").read_text(encoding="utf-8"))
    fingerprints = {str(year): sha256(cache / f"clayton_{year}.npz") for year in range(2005, 2026)}
    if fingerprints != baseline["input_sha256"]:
        raise ValueError("The archive does not match the original long-model run")
    provenance = load_provenance(provenance_path)
    (native, harmonized), masks = train_pair(cache, provenance)
    saved_native = HistoryCorrection.load(baseline_path / "twenty_full_spatial_model.npz")
    if (native.training_dates != saved_native.training_dates or native.training_pixel_dates != saved_native.training_pixel_dates
            or harmonized.training_dates != native.training_dates or harmonized.training_pixel_dates != native.training_pixel_dates):
        raise ValueError("The paired fits must reproduce the original training support")
    np.testing.assert_array_equal(native.coefficients, saved_native.coefficients, err_msg="Original long fit changed")
    models = {"one_season": HistoryCorrection.load(baseline_path / "one_season_spatial_model.npz"),
              "twenty_full": native, "twenty_harmonized": harmonized}
    for name, model in models.items():
        model.save(work / f"{name}_spatial_model.npz")
    rows, samples, skipped = evaluate(cache, work, models)
    results = {}
    for pattern, entries in rows.items():
        if not entries:
            raise ValueError(f"No eligible 2025 acquisitions for {pattern}")
        old_rows = baseline["patterns"][pattern]["dates"]
        for new, old in zip(entries, old_rows, strict=True):
            if (new["date"], new["reference"]) != (old["date"], old["reference"]):
                raise ValueError(f"Test date or reference changed for {pattern}")
            # Equal counts imply equal support: the new mask intersects the old mask with the new method.
            for variable, score in new["scores"].items():
                if score["common_pixels"] != old["scores"][variable]["common_pixels"]:
                    raise ValueError(f"Harmonization changed the evaluation support on {new['date']} for {pattern} {variable}")
                for key in (*BASELINES, "one_season", "twenty_full"):
                    if score["methods"][key]["rmse"] != old["scores"][variable]["methods"][key]["rmse"]:
                        raise ValueError(f"Original {key} predictions changed for {pattern} {variable}")
        results[pattern] = {
            "march_october": summarize(entries, samples[pattern], METHODS, "twenty_full", "twenty_harmonized"),
            "density": density_figure(work, pattern, samples[pattern], tuple(PREDICTIONS), METHODS, "March–October"),
            "dates": entries,
        }
        metrics = work / pattern
        metrics.mkdir()
        write_metrics(metrics, entries)
        print(f"{pattern}: {len(entries)} dates with original 2025 evaluation support verified", flush=True)
    report = {
        "grid": GRID, "methods": METHODS, "patterns": results, "settings": baseline["settings"] | {
            "primary_comparison": ["twenty_harmonized", "twenty_full"],
            "training": baseline["settings"]["training"] | {"twenty_harmonized": "March–October 2005–2024, OLI scale"},
        },
        "input_sha256": fingerprints, "baseline_run_sha256": sha256(baseline_path / "run.json"), "test_skipped": skipped,
        "training": {name: {"dates": model.training_dates, "pixels": model.training_pixel_dates,
                             "coefficients": model.coefficients.tolist()} for name, model in models.items()},
        "matched_training_masks": masks,
        "harmonization": {"source": COEFFICIENT_SOURCE, "bands": ["NIR", "Red"], "intercepts": INTERCEPTS,
                          "slopes": SLOPES, "scale": "OLI surface reflectance, Landsat Collection 2 Level 2",
                          "applies_to": ["Landsat 5 TM (ETM+ proxy)", "Landsat 7 ETM+"],
                          "mixed_dates": {day: int(legacy.sum()) for day, (legacy, _) in provenance.items()},
                          "provenance_sha256": sha256(provenance_path)},
    }
    (work / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=SPRING_FALL_CACHE)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/spatial_validation/clayton_harmonization_2005_2025")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".spatial_harmonization_", dir=args.output.parent) as temporary:
        staging = Path(temporary) / args.output.name
        staging.mkdir()
        run(args.cache, args.provenance, args.baseline, staging)
        staging.rename(args.output)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
