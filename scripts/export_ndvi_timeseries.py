#!/usr/bin/env python3
"""Export one completed Clayton NDVI year for temporal and monthly QGIS review."""
from __future__ import annotations

import argparse
import calendar
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_ndvi_inputs import CACHE
from scripts.export_stair_stages import NODATA
from scripts.run_daily_ndvi import SOURCE, read_cache, write_input_ndvi

DISPLAY_LIMITS = (-0.2, 1.0)
COLORS = ("#427baa", "#187963", "#f5b547", "#8e639c", "#e0e4e9")
SOURCE_LABELS = (
    "Observed Landsat", "Two-pair fusion", "Single-pair fusion",
    "Landsat interpolation", "Unresolved",
)


def write_raster(path, values, profile, descriptions, **tags):
    values = np.asarray(values)
    with rasterio.open(path, "w", **{**profile, "count": len(values)}) as dst:
        dst.write(values)
        dst.descriptions = tuple(descriptions)
        dst.update_tags(**tags)


def statistics(values):
    valid = values[np.isfinite(values)]
    if not valid.size:
        return {name: None for name in ("mean", "median", "q25", "q75")}
    q25, median, q75 = np.quantile(valid, (0.25, 0.50, 0.75))
    return {"mean": float(valid.mean()), "median": float(median),
            "q25": float(q25), "q75": float(q75)}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_curve(path, dates, modeled, observed, flags, coverage_limit):
    from matplotlib import dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    times = np.array([date.fromisoformat(day) for day in dates])
    figure = Figure(figsize=(12, 7.6), facecolor="white", layout="constrained")
    FigureCanvasAgg(figure)
    top, bottom = figure.subplots(2, 1, sharex=True,
                                 gridspec_kw={"height_ratios": (2.2, 1)})
    figure.suptitle(f"Clayton study area | NDVI through {times[0].year}",
                   fontsize=19, weight="bold", color="#1d2b36")
    enough = (np.isfinite(modeled).mean(axis=(1, 2)) >= coverage_limit)
    quantiles = np.full((3, len(dates)), np.nan)
    for index in np.flatnonzero(enough):
        quantiles[:, index] = np.nanquantile(modeled[index], (0.25, 0.5, 0.75))
    top.fill_between(times, quantiles[0], quantiles[2], color="#187963", alpha=0.14,
                     label="Middle 50% of pixel values")
    top.plot(times, quantiles[1], color="#187963", linewidth=1.5,
             label="Daily product median")
    observed_enough = np.isfinite(observed).mean(axis=(1, 2)) >= coverage_limit
    medians = [np.nanmedian(observed[index]) for index in np.flatnonzero(observed_enough)]
    top.scatter(times[observed_enough], medians, color="#315b9a", s=22,
                edgecolors="white", linewidths=0.45, zorder=3,
                label="QA-clear Landsat median")
    top.set_ylabel("NDVI")
    top.set_ylim(*DISPLAY_LIMITS)
    top.set_title(f"Area summaries shown when at least {coverage_limit:.0%} of pixels are available",
                  fontsize=10, loc="left", pad=10)
    top.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    fractions = [np.mean(flags == code, axis=(1, 2)) * 100 for code in (1, 2, 3, 4, 0)]
    bottom.stackplot(times, fractions, colors=COLORS, labels=SOURCE_LABELS, linewidth=0)
    bottom.set_ylabel("Study area (%)")
    bottom.set_ylim(0, 100)
    bottom.set_yticks((0, 50, 100))
    bottom.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=5,
                  frameon=False, fontsize=8.5)
    bottom.xaxis.set_major_locator(mdates.MonthLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    bottom.set_xlim(date(times[0].year, 1, 1), date(times[0].year, 12, 31))
    for axis in (top, bottom):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e4e9ee", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.savefig(path, dpi=180)


def export(source, cache, year, output, monthly_fraction, coverage_limit):
    annual = output / "annual"
    daily = output / "daily"
    monthly = output / "monthly"
    for folder in (annual, daily, monthly):
        folder.mkdir()
    run = json.loads((source / str(year) / "run.json").read_text(encoding="utf-8"))
    dates = [row["date"] for row in run["daily_coverage"]]
    expected = calendar.isleap(year) + 365
    if len(dates) != expected or dates[0] != f"{year}-01-01" or dates[-1] != f"{year}-12-31":
        raise ValueError("This review requires a completed full calendar year")
    for stem in ("NDVI", "NDVI_source"):
        shutil.copy2(source / str(year) / f"{stem}_{year}.tif", annual)
    shutil.copy2(source / str(year) / "run.json", annual / "run.json")
    with rasterio.open(annual / f"NDVI_{year}.tif") as src:
        modeled = src.read(masked=True).filled(np.nan)
        profile = src.profile
        if list(src.descriptions) != dates:
            raise ValueError("Annual NDVI bands do not match the run calendar")
        bounds = list(src.bounds)
    with rasterio.open(annual / f"NDVI_source_{year}.tif") as src:
        flags = src.read()
        if (src.transform != profile["transform"] or src.crs != profile["crs"]
                or list(src.descriptions) != dates or flags.shape != modeled.shape):
            raise ValueError("Source flags do not match the NDVI grid and calendar")
    if not np.array_equal(np.isfinite(modeled), flags != 0):
        raise ValueError("Source flags disagree with the modeled NDVI validity mask")
    manifest, landsat, modis = read_cache(cache / f"clayton_{year}.npz", year)
    if hashlib.sha256((cache / f"clayton_{year}.npz").read_bytes()).hexdigest() != run["input_sha256"]:
        raise ValueError("The cache differs from the inputs used for the annual product")
    write_input_ndvi(annual, year, dates,
                     {key: value for key, value in profile.items() if key not in ("dtype", "nodata", "predictor")},
                     landsat, modis)
    with rasterio.open(annual / f"Landsat_NDVI_{year}.tif") as src:
        observed = src.read(masked=True).filled(np.nan)
    with rasterio.open(annual / f"MODIS_NDVI_{year}.tif") as src:
        coarse = src.read(masked=True).filled(np.nan)
    if not np.array_equal(modeled[flags == 1], observed[flags == 1]):
        raise ValueError("The daily product changed a clear Landsat observation")
    rows = []
    pixels = flags.shape[1] * flags.shape[2]
    single_profile = {**profile, "count": 1}
    for index, day in enumerate(dates):
        row = {"date": day, "day_of_year": index + 1,
               "valid_pixels": int(np.isfinite(modeled[index]).sum()), "total_pixels": pixels,
               "valid_fraction": float(np.isfinite(modeled[index]).mean()),
               **{f"ndvi_{key}": value for key, value in statistics(modeled[index]).items()},
               "landsat_valid_fraction": float(np.isfinite(observed[index]).mean()),
               "landsat_median": statistics(observed[index])["median"],
               "modis_valid_fraction": float(np.isfinite(coarse[index]).mean()),
               "modis_median": statistics(coarse[index])["median"],
               **{name: int(np.count_nonzero(flags[index] == code)) for code, name in SOURCE.items()}}
        rows.append(row)
        write_raster(daily / f"NDVI_{day}.tif",
                     np.where(np.isfinite(modeled[index:index + 1]), modeled[index:index + 1], NODATA),
                     single_profile, [day], variable="NDVI", date=day,
                     source_flags=f"../annual/NDVI_source_{year}.tif", source_band=str(index + 1))
    write_csv(output / "daily_ndvi_summary.csv", rows)
    monthly_rows = []
    monthly_arrays = {}
    count_profile = {**profile, "dtype": "uint16", "nodata": 65535, "predictor": 2}
    for month in range(1, 13):
        prefix = f"{year}-{month:02d}"
        indexes = [i for i, day in enumerate(dates) if day.startswith(prefix)]
        values = modeled[indexes]
        counts = np.isfinite(values).sum(axis=0)
        required = int(np.ceil(len(indexes) * monthly_fraction))
        means = np.divide(np.nansum(values, axis=0, dtype=np.float64), counts,
                          out=np.full(counts.shape, np.nan), where=counts >= required).astype(np.float32)
        monthly_arrays[month] = means
        write_raster(monthly / f"NDVI_mean_{prefix}.tif",
                     np.where(np.isfinite(means[None]), means[None], NODATA), single_profile,
                     [f"{prefix} mean NDVI"], variable="monthly_mean_NDVI",
                     minimum_supported_days=str(required), calendar_days=str(len(indexes)))
        write_raster(monthly / f"NDVI_valid_days_{prefix}.tif", counts[None].astype(np.uint16),
                     count_profile, ["Supported days"], units="days", calendar_days=str(len(indexes)))
        counts_by_source = np.stack([np.count_nonzero(flags[indexes] == code, axis=0)
                                     for code in SOURCE]).astype(np.uint16)
        write_raster(monthly / f"NDVI_source_days_{prefix}.tif", counts_by_source, count_profile,
                     list(SOURCE.values()), units="days")
        monthly_rows.append({"month": prefix, "calendar_days": len(indexes),
                             "minimum_supported_days": required,
                             "valid_area_fraction": float(np.isfinite(means).mean()),
                             **{f"monthly_mean_ndvi_{key}": value for key, value in statistics(means).items()},
                             "pixel_day_coverage": float(np.isfinite(values).mean())})
    write_csv(output / "monthly_ndvi_summary.csv", monthly_rows)
    changes = []
    (output / "seasonal").mkdir()
    for before, after in ((4, 7), (7, 10)):
        earlier = f"{year}-{before:02d}"
        later = f"{year}-{after:02d}"
        values = monthly_arrays[after] - monthly_arrays[before]
        filename = f"seasonal/NDVI_change_{later}_minus_{earlier}.tif"
        write_raster(output / filename, np.where(np.isfinite(values[None]), values[None], NODATA),
                     single_profile, [f"{later} minus {earlier}"], variable="change_in_monthly_mean_NDVI",
                     earlier_month=earlier, later_month=later)
        changes.append({"earlier_month": earlier, "later_month": later, "raster": filename,
                        "shared_valid_area_fraction": float(np.isfinite(values).mean()),
                        **statistics(values)})
    draw_curve(output / "ndvi_time_series.png", dates, modeled, observed, flags, coverage_limit)
    report = {
        "year": year, "dates": dates, "grid": manifest["grid"], "bounds": bounds,
        "study_area": "Clayton study grid, 5.01 km by 5.01 km within Clayton Township",
        "area_km2": pixels * 30 ** 2 / 1e6, "source_codes": SOURCE,
        "ndvi_display_limits": DISPLAY_LIMITS,
        "chart_minimum_daily_area_fraction": coverage_limit,
        "monthly_minimum_supported_day_fraction": monthly_fraction,
        "months": monthly_rows, "validation": run["validation"],
        "seasonal_changes": changes, "change_display_limits": (-0.5, 0.5),
        "input_sha256": run["input_sha256"],
        "source_fractions": {name: float(np.mean(flags == code)) for code, name in SOURCE.items()},
        "days_without_estimates": sum(row["valid_pixels"] == 0 for row in rows),
        "references": {
            "NDVI": "https://www.usgs.gov/landsat-missions/landsat-normalized-difference-vegetation-index",
            "MODIS": "https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD43A4",
            "QGIS": "https://docs.qgis.org/3.44/en/docs/user_manual/map_views/map_view.html#time-based-control-on-the-map-canvas",
        },
    }
    (output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    validation = run["validation"]
    validation_rmse = {name: f"{value:.5f}" if value is not None else "unavailable"
                       for name, value in validation["rmse"].items()}
    (output / "README.txt").write_text(
        f"CLAYTON STUDY AREA NDVI TIME SERIES — {year}\n\n"
        "Open clayton_ndvi_time_series.qgz in QGIS 3.38 or later after extracting the full ZIP.\n"
        "The package covers the established 167 x 167 pixel study grid within Clayton Township.\n"
        "Pixels are 30 m in EPSG:32616. The square is 5.01 x 5.01 km (25.1001 km2).\n\n"
        "QGIS REVIEW\n"
        "The opening view contains monthly mean NDVI maps. Select one month in the Layers panel.\n"
        "For animation, turn off Monthly mean NDVI and turn on Daily time series.\n"
        "Open View > Panels > Temporal Controller. Choose Animated temporal navigation.\n"
        f"Set the range to {year}-01-01 00:00:00 through {year + 1}-01-01 00:00:00 and step to 1 day.\n"
        "Choose one daily layer (NDVI product, Landsat, MODIS or source flags), then press Play.\n"
        "Use Project > Layouts for Monthly NDVI and NDVI curve.\n"
        "Load NDVI_shared.qml to apply the same display scale to an individual daily TIFF.\n"
        "Identify a pixel in the annual stack to see its date-labelled daily values.\n\n"
        "FILES\n"
        "annual/ contains the daily NDVI product, QA-clear Landsat and MODIS baselines, source flags, and run.json.\n"
        "Every annual raster has one ISO-date-labelled band per calendar day, including February 29.\n"
        "daily/ contains an individual single-band product GeoTIFF for every calendar date.\n"
        "monthly/ contains monthly mean NDVI, supported-day counts and counts by each source code.\n"
        "daily_ndvi_summary.csv and monthly_ndvi_summary.csv contain area summaries and coverage.\n"
        "ndvi_time_series.png shows the seasonal curve with daily source coverage.\n"
        "monthly_ndvi_maps.png and seasonal_change.png show shared-scale maps and seasonal differences.\n\n"
        "METHOD\n"
        "NDVI = (NIR - red) / (NIR + red), from QA-screened surface reflectance.\n"
        "Clear same-day Landsat observations take priority. The daily model interpolates the\n"
        "Landsat-minus-MODIS reflectance residual between independent clear paired dates.\n"
        "The maximum anchor age is 32 days on each side. Enabled fallbacks use one pair\n"
        "within 16 days, then two-sided Landsat interpolation.\n"
        "MODIS MCD43A4.061 is a nominal 500 m product resampled onto the 30 m grid.\n"
        "Its daily retrievals use a moving 16-day window and share underlying observations.\n"
        "The grid specifies output spacing. Fine detail is supported by Landsat anchors.\n"
        "Negative NIR or red and nonpositive NIR+red are excluded from this implementation's NDVI.\n"
        "Source codes are 0 unresolved, 1 observed Landsat, 2 two-pair fusion,\n"
        "3 single-pair fusion and 4 Landsat interpolation. The NDVI NoData value is -9999.\n\n"
        "INTERPRETING THE TIME SERIES\n"
        f"Monthly means require valid estimates on at least {monthly_fraction:.0%} of the month's calendar days per pixel.\n"
        "The monthly supported-day rasters include zero where all days are unresolved.\n"
        f"The curve shows area medians and the spatial middle 50% on dates with at least {coverage_limit:.0%} coverage.\n"
        "The shaded range describes spatial variation among pixels.\n"
        "The CSV retains summaries for every date, with its coverage fraction.\n"
        "Blue Landsat points are input observations retained in the daily product.\n"
        "Independent accuracy is measured by separately withholding whole Landsat dates.\n"
        "Spatial support can vary between dates and months. Compare the coverage and source flags with the NDVI.\n"
        f"This run has {report['days_without_estimates']} dates with no supported estimates.\n"
        "Unsupported pixels remain NoData and plot gaps. Seasonal changes compare monthly means on shared valid pixels.\n"
        "Greenness changes can reflect vegetation development and management. Crop identity and causes\n"
        "require field or land-cover information. A single year describes a seasonal cycle.\n"
        "Changes across years need matching seasons, consistent spatial support and sensor checks.\n\n"
        "INDEPENDENT VALIDATION\n"
        f"Whole target Landsat dates were withheld. {validation['dates_with_common_pixels']} dates have common valid support.\n"
        f"Common scored pixel-dates = {validation['common_pixel_dates']}.\n"
        f"NDVI RMSE for two-pair fusion = {validation_rmse['fusion']}.\n"
        f"NDVI RMSE for same-pair Landsat interpolation = {validation_rmse['same_pairs_landsat']}.\n"
        f"NDVI RMSE for nearest-clear Landsat interpolation = {validation_rmse['landsat_only']}.\n"
        f"NDVI RMSE for MODIS alone = {validation_rmse['modis_only']}.\n"
        "These scores describe common valid pixels in the held-out scenes. The blended product also\n"
        "includes copied observations and flagged fallback estimates. Its daily values share source observations.\n\n"
        "REPRODUCE AND EXTEND\n"
        f"python scripts/collect_ndvi_inputs.py --start-year {year} --end-year {year}\n"
        f"python scripts/run_daily_ndvi.py --start-year {year} --end-year {year} --fallback --output outputs/daily_ndvi/clayton_{year}_timeseries\n"
        f"python scripts/export_ndvi_timeseries.py outputs/daily_ndvi/clayton_{year}_timeseries --year {year}\n"
        f"Use QGIS Python to run scripts/build_ndvi_timeseries_qgis.py outputs/daily_ndvi/clayton_{year}_timeseries/review\n"
        "Change the year and output directory to add another year on the same grid.\n\n"
        + "\n".join(report["references"].values()) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Output directory from run_daily_ndvi.py")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--monthly-fraction", type=float, default=0.50)
    parser.add_argument("--chart-coverage", type=float, default=0.80)
    args = parser.parse_args()
    if not 0 < args.monthly_fraction <= 1 or not 0 < args.chart_coverage <= 1:
        parser.error("Coverage fractions must be in (0, 1]")
    output = args.output or args.source / "review"
    if output.exists():
        parser.error(f"Choose a fresh output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".timeseries_", dir=output.parent) as directory:
        staging = Path(directory) / "review"
        staging.mkdir()
        export(args.source, args.cache, args.year, staging, args.monthly_fraction, args.chart_coverage)
        staging.replace(output)
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
