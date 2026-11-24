#!/usr/bin/env python3
"""Export actual March–October Landsat observations for a Clayton NDVI review."""
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
from scripts.export_ndvi_timeseries import DISPLAY_LIMITS, statistics, write_csv, write_raster
from scripts.export_stair_stages import NODATA
from scripts.run_daily_ndvi import read_cache
from stair.ndvi import ndvi


def plot_observations(path, dates, values, selected, common, year):
    from matplotlib import dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    times = np.array([date.fromisoformat(day) for day in dates])
    figure = Figure(figsize=(12, 7.6), facecolor="white", layout="constrained")
    FigureCanvasAgg(figure)
    top, bottom = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": (2.3, 1)})
    figure.suptitle(f"Clayton study area | Observed Landsat NDVI, {year}",
                   fontsize=19, weight="bold", color="#1d2b36")
    q25, median, q75 = np.quantile(values[selected][:, common], (0.25, 0.5, 0.75), axis=1)
    top.errorbar(times[selected], median, yerr=(median - q25, q75 - median), fmt="o",
                 color="#315b9a", ecolor="#a7bdcf", markersize=5, capsize=3,
                 label="Observed median and spatial middle 50%")
    top.set_title(f"Same {common.sum():,} pixels on all {selected.sum()} plotted acquisitions "
                  f"({common.mean():.1%} of the study grid)", loc="left", fontsize=10, pad=10)
    top.set_ylim(*DISPLAY_LIMITS)
    top.set_ylabel("NDVI")
    top.legend(loc="upper left", frameon=False, fontsize=9)
    fractions = np.isfinite(values).mean(axis=(1, 2)) * 100
    bottom.bar(times, fractions, width=2.5, color=np.where(selected, "#315b9a", "#b2bdc8"))
    bottom.set_ylim(0, 105)
    bottom.set_ylabel("Clear area (%)")
    bottom.set_title("Coverage on each acquisition date | Blue dates enter the curve", loc="left", fontsize=10)
    bottom.xaxis.set_major_locator(mdates.MonthLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    bottom.set_xlim(date(year, 3, 1), date(year, 11, 1))
    for axis in (top, bottom):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e4e9ee", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.savefig(path, dpi=180)


def export(cache_path, output, year, minimum_observations, chart_coverage):
    manifest, landsat, _ = read_cache(cache_path, year)
    start, end = date(year, 3, 1), date(year, 11, 1)
    if manifest["start"] > start.isoformat() or manifest["end_exclusive"] < end.isoformat():
        raise ValueError("The input cache does not cover the requested season")
    dates = [day for day in landsat if start.isoformat() <= day < end.isoformat()]
    if not dates:
        raise ValueError("The cache has no Landsat acquisitions in this season")
    values = np.stack([ndvi(landsat[day])[0] for day in dates])
    selected = np.isfinite(values).mean(axis=(1, 2)) >= chart_coverage
    if not selected.any():
        raise ValueError("No acquisition meets the requested clear-area fraction for the curve")
    common = np.isfinite(values[selected]).all(axis=0)
    if not common.any():
        raise ValueError("The plotted acquisitions have no shared clear pixels")
    width, height = GRID["dimensions"]
    profile = dict(driver="GTiff", width=width, height=height, count=1,
                   crs=GRID["crs"], transform=Affine(*GRID["transform"]),
                   dtype="float32", nodata=NODATA, compress="deflate", predictor=3)
    count_profile = {**profile, "dtype": "uint16", "nodata": 65535, "predictor": 2}
    for name in ("acquisitions", "monthly", "annual"):
        (output / name).mkdir()
    rows = []
    scenes = {day: [scene for scene, scene_day in manifest["landsat_scenes"] if scene_day == day]
              for day in dates}
    for index, day in enumerate(dates):
        data = values[index]
        write_raster(output / f"acquisitions/Landsat_NDVI_{day}.tif",
                     np.where(np.isfinite(data[None]), data[None], NODATA), profile, [day],
                     variable="observed_Landsat_NDVI", acquisition_date=day,
                     scenes=json.dumps(scenes[day]), processing="QA-clear reflectance to NDVI")
        reflectance = landsat[day]
        write_raster(output / f"acquisitions/Landsat_NIR_Red_{day}.tif",
                     np.where(np.isfinite(reflectance), reflectance, NODATA), profile,
                     manifest["bands"], acquisition_date=day, units="surface_reflectance",
                     scenes=json.dumps(scenes[day]), band_order="NIR,Red")
        common_stats = statistics(data[common]) if selected[index] else {key: None for key in statistics(np.array([]))}
        rows.append({"date": day, "clear_pixels": int(np.isfinite(data).sum()),
                     "clear_area_fraction": float(np.isfinite(data).mean()),
                     **{f"available_pixel_{key}": value for key, value in statistics(data).items()},
                     "in_fixed_footprint_curve": bool(selected[index]),
                     **{f"fixed_footprint_{key}": value for key, value in common_stats.items()},
                     "landsat_scene_ids": " | ".join(scenes[day])})
    write_csv(output / "landsat_acquisition_summary.csv", rows)
    season_days = calendar(start, end)
    daily = np.full((len(season_days), height, width), np.nan, dtype=np.float32)
    by_date = dict(zip(dates, values))
    for index, day in enumerate(season_days):
        if day in by_date:
            daily[index] = by_date[day]
    write_raster(output / f"annual/Landsat_observed_NDVI_{year}_Mar_Oct.tif",
                 np.where(np.isfinite(daily), daily, NODATA), profile, season_days,
                 variable="observed_Landsat_NDVI", date_order="UTC calendar days",
                 temporal_support="Actual acquisition dates with QA-clear NDVI")
    write_raster(output / "curve_common_pixels.tif", common[None].astype(np.uint16), count_profile,
                 ["Same clear pixels on all plotted acquisitions"], units="mask_0_or_1")
    months = []
    for month in range(3, 11):
        prefix = f"{year}-{month:02d}"
        indexes = [index for index, day in enumerate(dates) if day.startswith(prefix)]
        monthly = values[indexes]
        counts = np.isfinite(monthly).sum(axis=0)
        means = np.divide(np.nansum(monthly, axis=0, dtype=np.float64), counts,
                          out=np.full(counts.shape, np.nan), where=counts >= minimum_observations).astype(np.float32)
        write_raster(output / f"monthly/Landsat_observed_mean_{prefix}.tif",
                     np.where(np.isfinite(means[None]), means[None], NODATA), profile,
                     [f"{prefix} observed Landsat mean NDVI"], variable="mean_observed_NDVI",
                     minimum_clear_observations=str(minimum_observations))
        write_raster(output / f"monthly/Landsat_clear_count_{prefix}.tif", counts[None].astype(np.uint16),
                     count_profile, [f"{prefix} clear Landsat observation count"], units="observations")
        months.append({"month": prefix, "acquisition_dates": len(indexes),
                       "dates_with_any_clear_pixels": int(np.isfinite(monthly).any(axis=(1, 2)).sum()),
                       "supported_area_fraction": float(np.isfinite(means).mean()),
                       "minimum_clear_observations": minimum_observations,
                       **{f"monthly_observed_mean_{key}": value for key, value in statistics(means).items()}})
    write_csv(output / "landsat_monthly_summary.csv", months)
    plot_observations(output / "landsat_observed_curve.png", dates, values, selected, common, year)
    report = {
        "year": year, "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "grid": GRID, "study_area": "5.01 km square Clayton study grid within Clayton Township",
        "acquisition_dates": dates, "calendar_dates": season_days,
        "acquisitions_with_clear_ndvi": int(np.isfinite(values).any(axis=(1, 2)).sum()),
        "acquisition_scenes": scenes, "ndvi_display_limits": DISPLAY_LIMITS,
        "monthly_minimum_clear_observations": minimum_observations, "months": months,
        "curve_minimum_acquisition_clear_fraction": chart_coverage,
        "curve_dates": [day for day, include in zip(dates, selected) if include],
        "curve_common_pixels": int(common.sum()), "curve_common_area_fraction": float(common.mean()),
        "input_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "input_metadata": manifest,
        "method": "Direct NDVI from observed QA-screened Landsat surface reflectance",
    }
    (output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        f"OBSERVED LANDSAT NDVI — CLAYTON STUDY AREA, MARCH–OCTOBER {year}\n\n"
        "Open landsat_observed_time_series.qgz in QGIS 3.38 or later. Keep the extracted files together.\n"
        "The established study grid covers 5.01 x 5.01 km within Clayton Township, at 30 m in EPSG:32616.\n\n"
        "OBSERVATIONS\n"
        "NDVI is calculated directly from Landsat Collection 2 Tier 1 Level 2 surface reflectance.\n"
        "QA_PIXEL bits 0–5 and QA_RADSAT screen fill, clouds, cirrus, shadows, snow and saturation.\n"
        "Same-day source scenes are mosaicked in source-ID order. Their IDs are recorded in the report and CSV.\n"
        "NDVI = (NIR - red) / (NIR + red), after reflectance scaling and offset.\n"
        "Landsat 8/9 use SR_B5 and SR_B4, and Landsat 5/7 use SR_B4 and SR_B3.\n"
        "Cached output names are standardized to SR_B5 (NIR) and SR_B4 (red).\n"
        "The implementation requires finite, nonnegative NIR and red and a positive sum.\n"
        "Cloud gaps and other excluded pixels remain NoData (-9999). Every value comes from a Landsat observation.\n"
        f"There are {len(dates)} acquisition dates and {report['acquisitions_with_clear_ndvi']} dates with usable NDVI pixels.\n"
        "acquisitions/ contains one NDVI TIFF and one NIR/red surface-reflectance TIFF per acquisition date.\n"
        "The NIR/red TIFF has two spectral bands. Open it as singleband gray and select band 1 or band 2.\n"
        "The annual NDVI stack has 245 date-labelled bands, from March 1 through October 31.\n"
        "Dates without acquisitions and excluded pixels have NoData. The QGIS temporal view preserves these gaps.\n\n"
        "MONTHLY MAPS\n"
        f"A monthly mean requires at least {minimum_observations} clear observations at each pixel.\n"
        "Monthly counts are saved alongside the NDVI maps. Zero is a valid observation count.\n"
        "These means describe the sampled clear acquisition dates. Their timing and number vary by month.\n"
        "Check the count maps when comparing months or years. Late-season cloud cover can favor earlier observations.\n\n"
        "SEASONAL CURVE\n"
        f"Acquisitions with at least {chart_coverage:.0%} clear area enter the curve.\n"
        f"The {len(report['curve_dates'])} plotted dates share exactly {common.sum():,} clear pixels "
        f"({common.mean():.1%} of the grid).\n"
        "Each point is the median of that fixed footprint. Vertical bars show its spatial middle 50% of NDVI.\n"
        "The footprint is saved as curve_common_pixels.tif. All acquisitions and their coverage are in the CSV.\n"
        "The bottom panel shows clear area on every acquisition date. Gaps between points reflect sampling.\n"
        "The curve describes vegetation greenness. Field or land-cover data are needed to attribute changes to crop management.\n\n"
        "QGIS\n"
        "Project > Layouts > Observed monthly NDVI shows the eight months on one NDVI scale.\n"
        "Project > Layouts > Observed NDVI curve shows the fixed-footprint curve and coverage.\n"
        "Acquisition NDVI lets you select an actual acquisition date and inspect cloud gaps.\n"
        "To animate, turn off the monthly and acquisition groups and enable Calendar observations.\n"
        "Open View > Panels > Temporal Controller, choose Animated temporal navigation and step 1 day.\n"
        f"Use March 1 {year} 00:00:00 through November 1 {year} 00:00:00 as the time range.\n"
        "Use NDVI_shared.qml to style individual NDVI TIFFs consistently.\n\n"
        "EXTEND THE ANALYSIS\n"
        f"python scripts/export_landsat_timeseries.py --year {year}\n"
        f"Use QGIS Python for scripts/build_landsat_timeseries_qgis.py outputs/daily_ndvi/clayton_{year}_landsat_observed_march_october\n"
        "Collect another year with collect_ndvi_inputs.py, then change the export year.\n"
        "Across-year comparisons should match months, locations, cloud screening and sensor conventions.\n"
        "For pre-2013 years use the historical March–October cache and pass its directory through --cache.\n"
        "The source-band names are standardized in that cache and retain mission-native reflectance.\n\n"
        "NDVI reference https://www.usgs.gov/landsat-missions/landsat-normalized-difference-vegetation-index\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-observations", type=int, default=2)
    parser.add_argument("--chart-coverage", type=float, default=0.80)
    args = parser.parse_args()
    if args.minimum_observations < 1 or not 0 < args.chart_coverage <= 1:
        parser.error("Use a positive observation count and a chart coverage in (0, 1]")
    output = args.output or ROOT / f"outputs/daily_ndvi/clayton_{args.year}_landsat_observed_march_october"
    if output.exists():
        parser.error(f"Choose a fresh output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".landsat_season_", dir=output.parent) as temporary:
        staging = Path(temporary) / "review"
        staging.mkdir()
        export(args.cache / f"clayton_{args.year}.npz", staging, args.year,
               args.minimum_observations, args.chart_coverage)
        staging.replace(output)
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
