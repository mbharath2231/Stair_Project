#!/usr/bin/env python3
"""Compare observed and STAIR-restored Landsat on every seasonal acquisition."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from affine import Affine
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import CACHE, GRID, calendar
from scripts.compare_spatial_history import SEGMENT_SETTINGS
from scripts.export_ndvi_timeseries import DISPLAY_LIMITS, statistics, write_csv, write_raster
from scripts.export_stair_stages import NODATA, WINDOW
from scripts.run_cloud_density_validation import choose_reference
from scripts.run_daily_ndvi import read_cache
from scripts.run_watershed_fusion import nearby, restore, validity
from stair.ndvi import ndvi
from stair.spatial import segment_reference
from stair.temporal import interpolate_temporal_numpy

STAGES = ("original", "temporal", "global", "adaptive")
SOURCES = {
    0: "unresolved_ndvi", 1: "observed_landsat", 2: "temporal_fallback",
    3: "global_correction", 4: "adaptive_correction",
}
PLOT_COLORS = {"original": "#315b9a", "adaptive": "#187963"}
PLOT_LABELS = {"original": "Original Landsat", "adaptive": "Restored Landsat"}


def restore_acquisition(images, day, segment_cache):
    observed = images[day]
    neighbors = nearby(images, day, WINDOW)
    result = interpolate_temporal_numpy(np.stack(list(neighbors.values())), list(neighbors), day,
                                        observed, max_distance_days=WINDOW)
    try:
        reference_day, reference_clear = choose_reference(images, day)
    except ValueError:
        reference_day, reference_clear = None, None
    clear = validity(observed)
    supported = ~clear & validity(result.filled)
    segmentation = None
    reference = images[reference_day] if reference_day else None
    if reference is not None and min(int(clear.sum()), int(supported.sum())) >= 30:
        if reference_day not in segment_cache:
            segment_cache[reference_day] = segment_reference(reference, **SEGMENT_SETTINGS)
        segmentation = segment_cache[reference_day]
    global_fill, adaptive, info = restore(observed, result.filled, reference, segmentation=segmentation)
    stages = {name: values.astype(np.float32) for name, values in zip(
        STAGES, (observed, result.filled, global_fill, adaptive))}
    for name, values in stages.items():
        if not np.array_equal(values[:, clear], observed[:, clear]):
            raise RuntimeError(f"{name} changed clear reflectance on {day}")
    indices = {name: ndvi(values)[0] for name, values in stages.items()}
    observed_valid = np.isfinite(indices["original"])
    if not np.array_equal(indices["adaptive"][observed_valid], indices["original"][observed_valid]):
        raise RuntimeError(f"Restoration changed observed NDVI on {day}")
    source = np.zeros(clear.shape, dtype=np.uint8)
    filled_valid = ~clear & np.isfinite(indices["adaptive"])
    source[filled_valid] = 2
    global_applied = "support" in info and not info["support"]["global"]["fallback_to_temporal"]
    if global_applied:
        source[filled_valid] = 3
        segments = segmentation[0]
        for label, support in info["support"]["classes"].items():
            if not support["fallback_to_global"]:
                source[filled_valid & (segments == int(label))] = 4
    source[observed_valid] = 1
    if not np.array_equal(source != 0, np.isfinite(indices["adaptive"])):
        raise RuntimeError(f"Source flags disagree with restored NDVI on {day}")
    ages = np.stack((result.before_age_days[0], result.after_age_days[0])).astype(np.float32)
    return stages, indices, source, ages, {
        "reference_date": reference_day, "reference_clear_fraction": reference_clear,
        "temporal_neighbor_dates": list(neighbors), "spatial_fit": info,
    }


def draw_curves(output, dates, rows, year):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    x = np.arange(len(dates))
    variants = (
        ("seasonal_curve_comparison.png", ("original", "adaptive"), "Original and restored Landsat"),
        ("original_landsat_curve.png", ("original",), "Original Landsat"),
        ("restored_landsat_curve.png", ("adaptive",), "Restored Landsat"),
    )
    for filename, names, title in variants:
        figure = Figure(figsize=(15, 8.6), facecolor="white", layout="constrained")
        FigureCanvasAgg(figure)
        top, bottom = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": (2.6, 1)})
        figure.suptitle(f"Clayton study area | {title}, March–October {year}",
                       fontsize=19, weight="bold", color="#1d2b36")
        for name in names:
            medians = np.array([np.nan if row[f"{name}_median"] is None else row[f"{name}_median"]
                                for row in rows])
            coverage = np.array([row[f"{name}_valid_fraction"] for row in rows])
            top.plot(x, medians, color=PLOT_COLORS[name], linewidth=1.5, marker="o", markersize=4.3,
                     label=PLOT_LABELS[name])
            low = (coverage < 0.8) & np.isfinite(medians)
            top.scatter(x[low], medians[low], s=23, facecolor="white", edgecolor=PLOT_COLORS[name],
                        linewidth=1, zorder=4)
            bottom.plot(x, coverage * 100, color=PLOT_COLORS[name], linewidth=1.2, marker="o", markersize=3)
        top.set_ylim(*DISPLAY_LIMITS)
        top.set_ylabel("Median NDVI of available pixels")
        top.set_title("Every acquisition is included | Hollow markers indicate less than 80% area coverage",
                      fontsize=10, loc="left", pad=10)
        top.legend(loc="upper left", frameon=False)
        bottom.set_ylim(-3, 105)
        bottom.set_ylabel("Available area (%)")
        bottom.set_xticks(x, [day[5:] for day in dates], rotation=90, fontsize=8)
        bottom.set_xlabel("Landsat acquisition date (month–day, equally spaced)", labelpad=9)
        bottom.set_xlim(-0.6, len(dates) - 0.4)
        for axis in (top, bottom):
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#e4e9ee", linewidth=0.6)
            axis.set_axisbelow(True)
        figure.savefig(output / filename, dpi=180)


def run(cache_path, output, year):
    manifest, images, _ = read_cache(cache_path, year)
    start, end = date(year, 3, 1), date(year, 11, 1)
    if manifest["start"] > start.isoformat() or manifest["end_exclusive"] < end.isoformat():
        raise ValueError("The input cache lacks the requested season")
    dates = [day for day in images if start.isoformat() <= day < end.isoformat()]
    if not dates:
        raise ValueError("No seasonal Landsat acquisition dates in the cache")
    width, height = GRID["dimensions"]
    profile = dict(driver="GTiff", width=width, height=height, count=1,
                   crs=GRID["crs"], transform=Affine(*GRID["transform"]),
                   dtype="float32", nodata=NODATA, compress="deflate", predictor=3)
    source_profile = {**profile, "dtype": "uint8", "nodata": 255, "predictor": 2}
    (output / "acquisitions").mkdir()
    (output / "annual").mkdir()
    segments, rows, details = {}, [], []
    indices_by_stage = {name: {} for name in STAGES}
    sources = {}
    for day in dates:
        stages, indices, source, ages, info = restore_acquisition(images, day, segments)
        folder = output / "acquisitions" / day
        folder.mkdir()
        scene_ids = [scene for scene, scene_day in manifest["landsat_scenes"] if scene_day == day]
        row = {"date": day, "reference_date": info["reference_date"], "total_pixels": width * height}
        for name in STAGES:
            indices_by_stage[name][day] = indices[name]
            write_raster(folder / f"{name}_reflectance.tif",
                         np.where(np.isfinite(stages[name]), stages[name], NODATA), profile, manifest["bands"],
                         acquisition_date=day, stage=name, units="surface_reflectance", scenes=json.dumps(scene_ids))
            write_raster(folder / f"{name}_NDVI.tif",
                         np.where(np.isfinite(indices[name][None]), indices[name][None], NODATA), profile, [day],
                         acquisition_date=day, stage=name, variable="NDVI")
            row.update({f"{name}_valid_pixels": int(np.isfinite(indices[name]).sum()),
                        f"{name}_valid_fraction": float(np.isfinite(indices[name]).mean()),
                        **{f"{name}_{key}": value for key, value in statistics(indices[name]).items()}})
        sources[day] = source
        write_raster(folder / "restoration_source.tif", source[None], source_profile, [day],
                     acquisition_date=day, **{f"code_{code}": name for code, name in SOURCES.items()})
        write_raster(folder / "temporal_anchor_age_days.tif", np.where(np.isfinite(ages), ages, NODATA),
                     profile, ["Prior clear Landsat age", "Later clear Landsat age"], units="days", acquisition_date=day)
        row.update({name: int(np.count_nonzero(source == code)) for code, name in SOURCES.items()})
        row["spatial_correction_applied_pixels"] = row["global_correction"] + row["adaptive_correction"]
        row["landsat_scene_ids"] = " | ".join(scene_ids)
        rows.append(row)
        details.append({"date": day, **info})
        print(f"{day} | original {row['original_valid_fraction']:.1%} | restored {row['adaptive_valid_fraction']:.1%} "
              f"| spatially corrected {row['spatial_correction_applied_pixels']:,} pixels", flush=True)
    write_csv(output / "all_acquisition_ndvi.csv", rows)
    season_days = calendar(start, end)
    for name, observations in indices_by_stage.items():
        stack = np.stack([observations.get(day, np.full((height, width), np.nan, dtype=np.float32))
                          for day in season_days])
        write_raster(output / f"annual/{name}_NDVI_{year}_Mar_Oct.tif",
                     np.where(np.isfinite(stack), stack, NODATA), profile, season_days,
                     variable="NDVI", stage=name, date_order="UTC calendar days")
    source_stack = np.stack([sources.get(day, np.full((height, width), 255, dtype=np.uint8)) for day in season_days])
    write_raster(output / f"annual/restoration_source_{year}_Mar_Oct.tif", source_stack, source_profile,
                 season_days, **{f"code_{code}": name for code, name in SOURCES.items()})
    draw_curves(output, dates, rows, year)
    report = {
        "year": year, "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "grid": GRID, "study_area": "5.01 km square Clayton study grid within Clayton Township",
        "acquisition_dates": dates, "calendar_dates": season_days, "ndvi_display_limits": DISPLAY_LIMITS,
        "stages": STAGES, "source_codes": SOURCES, "source_nodata": 255,
        "model": "Landsat temporal interpolation followed by global and adaptive spatial offsets",
        "max_temporal_days_per_side": WINDOW, "minimum_reference_clear_fraction": 0.95,
        "maximum_spatial_reference_days": 21, "segmentation_settings": SEGMENT_SETTINGS,
        "input_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(), "input_metadata": manifest,
        "original_dates_with_ndvi": sum(row["original_valid_pixels"] > 0 for row in rows),
        "restored_dates_with_ndvi": sum(row["adaptive_valid_pixels"] > 0 for row in rows),
        "restored_minimum_area_fraction": min(row["adaptive_valid_fraction"] for row in rows),
        "fully_masked_acquisition_dates": [row["date"] for row in rows if row["original_valid_pixels"] == 0],
        "dates_with_spatial_correction": sum(row["spatial_correction_applied_pixels"] > 0 for row in rows),
        "acquisition_details": details,
    }
    (output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        f"ORIGINAL AND RESTORED LANDSAT SEASONAL CURVES — MARCH–OCTOBER {year}\n\n"
        "Extract the complete ZIP and open landsat_restoration_time_series.qgz in QGIS 3.38 or later.\n"
        "This study covers the existing 167 x 167 Clayton grid at 30 m in EPSG:32616 (5.01 km square).\n\n"
        "EVERY LANDSAT ACQUISITION\n"
        f"All {len(dates)} Landsat acquisition dates from March 1 through October 31 are included.\n"
        "Original means QA-screened Landsat surface reflectance, with NDVI calculated from NIR and red.\n"
        f"The original series has usable NDVI on {report['original_dates_with_ndvi']} dates.\n"
        "Fully masked targets retain a missing original NDVI value.\n"
        "Curves use the median of the available pixels over the study grid at each acquisition date.\n"
        "Every acquisition date is labelled on the x axis at equal spacing. Connections between markers guide reading.\n"
        "Hollow markers indicate dates with less than 80% valid area. Coverage is plotted underneath.\n"
        "seasonal_curve_comparison.png overlays the two curves. Separate original and restored curve PNGs are also supplied.\n"
        "all_acquisition_ndvi.csv contains every date, coverage, median, mean, spatial quartiles, reference and source counts.\n\n"
        "RESTORATION\n"
        "Each target is restored from other Landsat observations, with the target date excluded from temporal references.\n"
        "The nearest valid earlier and later observations may be up to 32 days away on each side.\n"
        "Boundary inputs from outside March–October support targets near the season edges.\n"
        "The spatial reference is the nearest independent acquisition within 21 days with at least 95% clear reflectance.\n"
        "Global correction adjusts the filled gap mean using clear target pixels and the independent reference contrast.\n"
        "Adaptive correction applies this adjustment within spectral classes of the reference.\n"
        "Segmentation uses the same 320 sampled pixels, four null draws and seed 42 as the matched-season benchmark.\n"
        "Sparse classes inherit the global offset. Insufficient support retains the temporal estimate.\n"
        "Fully masked targets use temporal interpolation because same-day clear target support is absent.\n"
        "All original clear reflectance and valid original NDVI pixels are preserved exactly.\n"
        "NDVI is calculated after each reflectance stage, using finite nonnegative NIR/red and a positive sum.\n"
        "Source codes are 0 unresolved NDVI, 1 observed Landsat, 2 temporal fallback,\n"
        "3 global correction and 4 adaptive correction. Value 255 is reserved for calendar dates without acquisitions.\n\n"
        "INTERPRETING THE TWO CURVES\n"
        "Original NDVI summarizes the visible pixels. Restored NDVI adds estimates in supported gaps.\n"
        "The sampled locations therefore vary with cloud coverage in the original curve.\n"
        "Differences between the curves reflect added gap estimates and changes in spatial support.\n"
        "The two products agree exactly on the same clear pixels. This preservation is verified for every date.\n"
        "Accuracy in natural cloud gaps requires controlled withholding of known clear pixels,\n"
        "as in the repository's cloud-filling validation. Seasonal curve comparisons describe behavior and coverage.\n\n"
        "QGIS\n"
        "Choose a date under Acquisition comparison, then Original NDVI, Restored NDVI or Restoration source.\n"
        "Project > Layouts contains the combined curve, both individual curves and example acquisition maps.\n"
        "Temporal, global and adaptive stage TIFFs are in each acquisition folder for closer inspection.\n"
        "For calendar animation, turn off Acquisition comparison and enable Calendar comparison.\n"
        "Open Temporal Controller, select Animated temporal navigation and use a 1-day step from\n"
        f"March 1 {year} 00:00 through November 1 {year} 00:00. Dates without Landsat remain empty.\n"
        "Use NDVI_shared.qml for the same NDVI display scale on individual stage TIFFs.\n\n"
        "ALL-ACQUISITION MAP SHEETS\n"
        "original_all_acquisition_maps.png and restored_all_acquisition_maps.png show every acquisition in chronological order.\n"
        "Read left to right, then continue on the next row. The same positions correspond to the same dates in both series.\n"
        "map_sheets/original/ and map_sheets/restored/ contain numbered pages with up to twelve maps per page.\n"
        "QGIS Project > Layouts includes Original NDVI and Restored NDVI overview and numbered-page layouts.\n"
        "All maps share the same NDVI scale, spatial extent and orientation. Each panel shows its available area.\n"
        "Follow the same field location across successive maps and use Identify in QGIS to read its NDVI values.\n"
        "The maps describe vegetation greenness. Crop records support interpretation of crop identity and growth stages.\n\n"
        "REPRODUCE\n"
        f"python scripts/compare_landsat_season.py --year {year}\n"
        f"Use QGIS Python for scripts/build_landsat_restoration_qgis.py outputs/daily_ndvi/clayton_{year}_acquisition_restoration\n"
        "The input cache and scene provenance are recorded in review.json.\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"outputs/daily_ndvi/clayton_{args.year}_acquisition_restoration"
    if output.exists():
        parser.error(f"Choose a fresh output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".acquisition_restoration_", dir=output.parent) as temporary:
        staging = Path(temporary) / "review"
        staging.mkdir()
        run(args.cache / f"clayton_{args.year}.npz", staging, args.year)
        staging.replace(output)
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
