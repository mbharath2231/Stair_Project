#!/usr/bin/env python3
"""
CLI Runner for Multi-Date Time Series Generation
Supports:
  1. fusion_season: Multi-month MODIS-Landsat NDVI synthesis across growing season
  2. crop_chip: USDA CDL crop pixel sampling and 5km chip time-lapse exports
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import config
from stair.core import init_earth_engine, load_aoi, get_landsat_collection, get_modis_collection
from stair.fusion import learn_spatial_rules, synthesize_missing_day
from stair.temporal import compute_temporal_regression
from stair.diagnostics import sample_agricultural_pixel, create_chip_bounds, build_time_series_frame


def run_fusion_season(aoi, args):
    print(f"\nRunning 6-month Fusion Time Series ({args.start_date} to {args.end_date}, every {args.step_days} days)...")
    golden_start = ee.Date(args.golden_date)
    golden_landsat = get_landsat_collection(aoi, golden_start, golden_start.advance(1, "day"), apply_mask=False).first()
    golden_modis = get_modis_collection(aoi, golden_start, golden_start.advance(1, "day")).first()

    print("Locking spatial rules from golden day...")
    rules = learn_spatial_rules(golden_landsat, golden_modis, kernel_radius=args.kernel_radius)

    date_range = pd.date_range(start=args.start_date, end=args.end_date, freq=f"{args.step_days}D")
    print(f"Submitting {len(date_range)} STAIR synthesis tasks...")

    for i, target_dt in enumerate(date_range):
        date_str = target_dt.strftime("%Y-%m-%d")
        dt_ee = ee.Date(date_str)
        target_modis = get_modis_collection(aoi, dt_ee, dt_ee.advance(1, "day")).first()

        fused = synthesize_missing_day(target_modis, rules, aoi)
        synth_ndvi = fused.select("synthetic_ndvi")
        file_name = f"stair_ndvi_{target_dt.strftime('%Y%m%d')}"

        if not args.dry_run:
            task = ee.batch.Export.image.toDrive(
                image=synth_ndvi,
                description=file_name,
                folder=args.export_folder,
                fileNamePrefix=file_name,
                scale=config.DEFAULT_SCALE,
                region=aoi,
                crs="EPSG:4326",
                maxPixels=1e10
            )
            task.start()
            print(f"[{i+1}/{len(date_range)}] Export submitted for {date_str}")
            time.sleep(0.5)
        else:
            print(f"[DRY RUN] [{i+1}/{len(date_range)}] Prepared {date_str}")


def run_crop_chip(aoi, args):
    print("\nRunning Agricultural Crop Chip Time-Series Pipeline...")
    point, lon, lat = sample_agricultural_pixel(aoi, year=2021)
    chip_bounds = create_chip_bounds(point, radius_meters=2500)
    print(f"Target crop pixel at Lon: {lon:.4f}, Lat: {lat:.4f} (5km bounding chip)")

    raw_col = get_landsat_collection(chip_bounds, args.start_date, args.end_date, apply_mask=False)
    masked_col = get_landsat_collection(chip_bounds, args.start_date, args.end_date, apply_mask=True)

    print("Fitting STAIR temporal regression for the 5km chip...")
    slope, intercept = compute_temporal_regression(masked_col, band_name=args.band)

    img_list = raw_col.toList(raw_col.size())
    num_images = img_list.size().getInfo()
    print(f"Discovered {num_images} total Landsat passes in season.")

    for i in range(num_images):
        raw_img = ee.Image(img_list.get(i))
        date_str = raw_img.date().format("YYYY-MM-dd").getInfo()
        stack = build_time_series_frame(raw_img, slope, intercept, band_name=args.band, bounds=chip_bounds)
        file_name = f"STAIR_5km_{date_str.replace('-', '')}"

        if not args.dry_run:
            task = ee.batch.Export.image.toDrive(
                image=stack,
                description=file_name,
                folder=args.export_folder,
                scale=config.DEFAULT_SCALE,
                region=chip_bounds,
                maxPixels=1e8
            )
            task.start()
            print(f"[{i+1}/{num_images}] Submitted task for {date_str}")
            time.sleep(0.5)
        else:
            print(f"[DRY RUN] [{i+1}/{num_images}] Prepared {date_str}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Date Time Series Generator")
    parser.add_argument("--mode", choices=["fusion_season", "crop_chip"], default="crop_chip", help="Pipeline mode")
    parser.add_argument("--start-date", default=config.DEFAULT_SEASON_START, help="Season start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=config.DEFAULT_SEASON_END, help="Season end date (YYYY-MM-DD)")
    parser.add_argument("--step-days", type=int, default=8, help="Interval in days for fusion_season")
    parser.add_argument("--golden-date", default=config.DEFAULT_REFERENCE_DATE, help="Clear golden day")
    parser.add_argument("--kernel-radius", type=int, default=config.DEFAULT_KERNEL_RADIUS, help="Moving window size")
    parser.add_argument("--band", default="SR_B5", help="Target spectral band")
    parser.add_argument("--project-id", default=config.EE_PROJECT_ID, help="EE project ID")
    parser.add_argument("--export-folder", default="STAIR_Time_Series", help="Google Drive folder")
    parser.add_argument("--dry-run", action="store_true", help="Skip Google Drive export")
    args = parser.parse_args()

    init_earth_engine(project_id=args.project_id)
    aoi, _ = load_aoi()

    if args.mode == "fusion_season":
        run_fusion_season(aoi, args)
    else:
        run_crop_chip(aoi, args)

    print("\nTime series processing complete!")


if __name__ == "__main__":
    main()
