#!/usr/bin/env python3
"""Produce daily Clayton NDVI and source flags from cached Red/NIR inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from affine import Affine
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import BANDS, CACHE, GRID, SPRING_FALL_CACHE, calendar, season_bounds
from scripts.export_stair_stages import NODATA, WINDOW
from stair.ndvi import DailyNDVI, ndvi
from stair.validation import score_holdout

SOURCE = {
    0: "unresolved", 1: "observed_landsat", 2: "fused_landsat_modis",
    3: "single_pair_fusion", 4: "landsat_interpolation",
}
DEFAULT_OUTPUT = ROOT / "outputs" / "daily_ndvi"
MODEL = "Nearest observed QA-clear Landsat–MODIS residual pair on each side per pixel"
FALLBACK = "One matched pair within 16 days, then bracketed Landsat interpolation"


def read_cache(path, year):
    with np.load(path, allow_pickle=False) as cache:
        manifest = json.loads(cache["manifest"].item())
        if manifest["grid"] != GRID or manifest["bands"] != BANDS or manifest["year"] != year:
            raise ValueError(f"Unexpected grid, bands, or year in {path}")
        width, height = GRID["dimensions"]
        sources = []
        for name in ("landsat", "modis"):
            dates = cache[f"{name}_dates"].tolist()
            values = cache[name]
            if values.shape != (len(dates), len(BANDS), height, width) or dates != sorted(set(dates)):
                raise ValueError(f"Invalid {name} arrays in {path}")
            sources.append(dict(zip(dates, values)))
    return manifest, *sources


def validate(model, days, minimum_clear, fallback):
    results = []
    for day in days:
        if day not in model.landsat:
            continue
        truth = ndvi(model.landsat[day])
        valid_truth = np.isfinite(truth[0])
        if valid_truth.mean() < minimum_clear:
            continue
        methods = model.holdout(day)
        valid = {name: np.isfinite(image[0]) for name, image in methods.items()}
        common = np.logical_and.reduce(list(valid.values()))
        row = {
            "date": day,
            "qa_clear_pixels": int(valid_truth.sum()),
            "coverage": {name: float(mask[valid_truth].mean()) for name, mask in valid.items()},
            "common_support": {name: score_holdout(truth, image, common, ["NDVI"]) for name, image in methods.items()},
        }
        if fallback:
            estimated, flags = model.predict(day, observed=False, fallback=True)
            row["fallback_validation"] = {
                SOURCE[code]: score_holdout(truth, estimated[None], flags == code, ["NDVI"]) for code in (3, 4)
            }
        results.append(row)
    return results


def combined_scores(holdouts):
    names = ("fusion", "same_pairs_landsat", "landsat_only", "modis_only")
    counts = [row["common_support"]["fusion"]["scored_pixels"] for row in holdouts]
    total = sum(counts)
    summary = {"holdout_dates": len(holdouts), "dates_with_common_pixels": sum(n > 0 for n in counts),
               "common_pixel_dates": total, "rmse": {}}
    for name in names:
        squares = sum(count * row["common_support"][name]["bands"]["NDVI"]["rmse"] ** 2
                      for row, count in zip(holdouts, counts) if count)
        summary["rmse"][name] = (squares / total) ** 0.5 if total else None
    return summary


def write_input_ndvi(work, year, dates, profile, landsat, modis):
    """Write the two sensor baselines on the same grid and calendar as fusion."""
    missing = np.full((profile["height"], profile["width"]), NODATA, dtype=np.float32)
    for name, observations in (("Landsat", landsat), ("MODIS", modis)):
        with rasterio.open(work / f"{name}_NDVI_{year}.tif", "w", **profile,
                           dtype="float32", predictor=3, nodata=NODATA) as dst:
            dst.update_tags(variable="NDVI", source=name, date_order="UTC calendar days")
            for index, day in enumerate(dates, start=1):
                values = missing
                if day in observations:
                    ndvi_values = ndvi(observations[day])[0]
                    values = np.where(np.isfinite(ndvi_values), ndvi_values, NODATA)
                dst.write(values, index)
                dst.set_band_description(index, day)


def write_year(cache_dir, output, year, max_age, minimum_clear, fallback, *, march_october=False):
    cache_path = cache_dir / f"clayton_{year}.npz"
    start, end = season_bounds(year, march_october)
    dates = calendar(start, end)
    with cache_path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    settings = {
        "input_sha256": digest.hexdigest(), "model": MODEL, "fallback": FALLBACK if fallback else None,
        "minimum_anchor_clear_fraction": minimum_clear, "max_anchor_days_per_side": max_age,
    }
    target = output / str(year)
    if target.exists():
        report = json.loads((target / "run.json").read_text(encoding="utf-8"))
        files = [f"NDVI_{year}.tif", f"NDVI_source_{year}.tif"]
        if march_october:
            files.extend((f"Landsat_NDVI_{year}.tif", f"MODIS_NDVI_{year}.tif"))
        if (any(report.get(key) != value for key, value in settings.items())
                or [row["date"] for row in report["daily_coverage"]] != dates
                or not all((target / file).is_file() for file in files)):
            raise FileExistsError(f"Existing output uses different inputs or settings: {target}")
        print(f"{year}: reusing completed output", flush=True)
        return
    manifest, landsat, modis = read_cache(cache_path, year)
    if (manifest["start"] > start.isoformat() or manifest["end_exclusive"] < end.isoformat()
            or not landsat or not modis):
        raise ValueError(f"{cache_path} lacks input coverage for the requested period")
    model = DailyNDVI(landsat, modis, max_age=max_age, min_clear_fraction=minimum_clear)
    holdouts = validate(model, dates, minimum_clear, fallback)
    width, height = GRID["dimensions"]
    profile = {"driver": "GTiff", "width": width, "height": height, "count": len(dates),
               "crs": GRID["crs"], "transform": Affine(*GRID["transform"]),
               "compress": "deflate", "interleave": "band"}
    coverage = []
    with TemporaryDirectory(prefix=f"{year}_", dir=output) as temporary:
        work = Path(temporary) / str(year)
        work.mkdir()
        with (rasterio.open(work / f"NDVI_{year}.tif", "w", **profile, dtype="float32",
                            predictor=3, nodata=NODATA) as ndvi_tif,
              rasterio.open(work / f"NDVI_source_{year}.tif", "w", **profile, dtype="uint8",
                            nodata=0) as source_tif):
            ndvi_tif.update_tags(variable="NDVI", date_order="UTC calendar days")
            source_tif.update_tags(**{f"code_{code}": name for code, name in SOURCE.items()})
            for index, day in enumerate(dates, start=1):
                values, flags = model.predict(day, fallback=fallback)
                ndvi_tif.write(np.where(flags, values, NODATA).astype(np.float32), index)
                source_tif.write(flags, index)
                ndvi_tif.set_band_description(index, day)
                source_tif.set_band_description(index, day)
                counts = np.bincount(flags.ravel(), minlength=len(SOURCE))
                coverage.append({"date": day, "band": index,
                                 **{name: int(counts[code]) for code, name in SOURCE.items()}})
        if march_october:
            write_input_ndvi(work, year, dates, profile, landsat, modis)
        report = {
            **settings, "inputs": manifest, "sources": SOURCE,
            "registration": "Shared 30 m grid; independent sensor offset estimation pending",
            "anchor_dates_including_boundary_padding": model.pair_days,
            "daily_coverage": coverage, "validation": combined_scores(holdouts), "held_out": holdouts,
        }
        (work / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        work.replace(target)
    print(f"{year}: {len(dates)} daily bands, {report['validation']['dates_with_common_pixels']} scored Landsat dates",
          flush=True)


def write_dataset_summary(output, years):
    daily, holdouts, periods = [], [], []
    pixels = GRID["dimensions"][0] * GRID["dimensions"][1]
    for year in years:
        run = json.loads((output / str(year) / "run.json").read_text(encoding="utf-8"))
        season = run["daily_coverage"]
        daily.extend(season)
        holdouts.extend(run["held_out"])
        period = {
            "year": year, "first_date": season[0]["date"], "last_date": season[-1]["date"],
            "ndvi": f"{year}/NDVI_{year}.tif", "source": f"{year}/NDVI_source_{year}.tif",
            "report": f"{year}/run.json",
            "ndvi_pixel_day_coverage": 1 - sum(row["unresolved"] for row in season) / (len(season) * pixels),
            "days_without_estimates": sum(row["unresolved"] == pixels for row in season),
        }
        if (output / str(year) / f"Landsat_NDVI_{year}.tif").is_file():
            period.update(landsat_ndvi=f"{year}/Landsat_NDVI_{year}.tif", modis_ndvi=f"{year}/MODIS_NDVI_{year}.tif")
        periods.append(period)
    total = len(daily) * pixels
    summary = {
        "grid": GRID, "first_date": daily[0]["date"], "last_date": daily[-1]["date"], "days": len(daily),
        "years": periods, "source_codes": SOURCE,
        "days_without_estimates": sum(row["unresolved"] == pixels for row in daily),
        "pixel_day_fraction_by_source": {
            name: sum(row[name] for row in daily) / total for name in SOURCE.values()
        },
        "validation": combined_scores(holdouts),
    }
    fallback = [row["fallback_validation"] for row in holdouts if "fallback_validation" in row]
    if fallback:
        summary["fallback_validation"] = {}
        for code in (3, 4):
            name = SOURCE[code]
            scores = [row[name] for row in fallback if row[name]["scored_pixels"]]
            count = sum(score["scored_pixels"] for score in scores)
            squares = sum(score["scored_pixels"] * score["bands"]["NDVI"]["rmse"] ** 2 for score in scores)
            summary["fallback_validation"][name] = {"scored_pixel_dates": count,
                                                     "rmse": (squares / count) ** 0.5 if count else None}
    (output / "dataset.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-age", type=int, default=WINDOW)
    parser.add_argument("--minimum-clear", type=float, default=0.60)
    parser.add_argument("--fallback", action="store_true",
                        help="Label short single-pair fusion and Landsat-only interpolation in otherwise unresolved pixels")
    parser.add_argument("--march-october", action="store_true",
                        help="Write only March 1 to October 31 for each year")
    args = parser.parse_args()
    if args.start_year > args.end_year or not 1 <= args.max_age <= WINDOW or not 0 < args.minimum_clear <= 1:
        parser.error("Use ordered years, a supported anchor age, and a clear fraction in (0, 1]")
    base = DEFAULT_OUTPUT / "march_october" if args.march_october else DEFAULT_OUTPUT
    output = args.output or base / ("with_fallback" if args.fallback else "strict")
    cache = args.cache or (SPRING_FALL_CACHE if args.march_october else CACHE)
    output.mkdir(parents=True, exist_ok=True)
    years = range(args.start_year, args.end_year + 1)
    for year in years:
        write_year(cache, output, year, args.max_age, args.minimum_clear, args.fallback,
                   march_october=args.march_october)
    write_dataset_summary(output, years)


if __name__ == "__main__":
    main()
