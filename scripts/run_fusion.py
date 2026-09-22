#!/usr/bin/env python3
"""
CLI Runner for Landsat 8 & MODIS Spatiotemporal Data Fusion
Learns spatial regression rules on a golden day and synthesizes high-res NDVI for missing dates.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import config
from stair.core import init_earth_engine, load_aoi, get_landsat_collection, get_modis_collection
from stair.fusion import learn_spatial_rules, synthesize_missing_day


def main():
    parser = argparse.ArgumentParser(description="Landsat 8 & MODIS Spatiotemporal Fusion")
    parser.add_argument("--prediction-date", default=config.DEFAULT_PREDICTION_DATE, help="Date to synthesize (YYYY-MM-DD)")
    parser.add_argument("--golden-date", default=config.DEFAULT_REFERENCE_DATE, help="Clear golden reference date")
    parser.add_argument("--kernel-radius", type=int, default=config.DEFAULT_KERNEL_RADIUS, help="Moving window radius in pixels")
    parser.add_argument("--project-id", default=config.EE_PROJECT_ID, help="EE project ID")
    parser.add_argument("--export-folder", default="STAIR_Exports", help="Google Drive folder")
    parser.add_argument("--validate-red", action="store_true", help="Generate Red band validation stack against real Landsat")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without Drive export")
    args = parser.parse_args()

    print(f"\n{'='*60}\nLANDSAT-MODIS SPATIOTEMPORAL FUSION PIPELINE\n{'='*60}")
    init_earth_engine(project_id=args.project_id)
    aoi, _ = load_aoi()

    print(f"Golden Date: {args.golden_date}")
    print(f"Prediction Date: {args.prediction_date}")
    print(f"Moving Window Kernel Radius: {args.kernel_radius} px")

    # 1. Fetch Golden Day Images
    g_start = ee.Date(args.golden_date)
    g_end = g_start.advance(1, "day")

    golden_landsat = get_landsat_collection(aoi, g_start, g_end, apply_mask=False).first()
    golden_modis = get_modis_collection(aoi, g_start, g_end).first()

    print("Executing server-side spatial linear regression on Red and NIR bands...")
    rules = learn_spatial_rules(golden_landsat, golden_modis, kernel_radius=args.kernel_radius)

    # 2. Fetch Prediction Day MODIS
    p_start = ee.Date(args.prediction_date)
    p_end = p_start.advance(1, "day")
    target_modis = get_modis_collection(aoi, p_start, p_end).first()

    print(f"Synthesizing 30m Red, NIR, and NDVI for {args.prediction_date}...")
    fused_image = synthesize_missing_day(target_modis, rules, aoi)
    synthetic_ndvi = fused_image.select("synthetic_ndvi")

    if args.validate_red:
        print("Building Red Band validation stack [MODIS_Blurry, STAIR_Sharp, REAL_Landsat]...")
        real_landsat = get_landsat_collection(aoi, p_start, p_end, apply_mask=False).first()
        modis_red = target_modis.select(config.MODIS_BANDS["red"]).rename("MODIS_Red_Blurry")
        stair_red = fused_image.select("synthetic_red").rename("STAIR_Red_Sharp")
        real_red = real_landsat.select(config.LANDSAT_BANDS["red"]).rename("REAL_Landsat_Red")
        validation_stack = ee.Image([modis_red, stair_red, real_red]).clip(aoi)

    if args.dry_run:
        print("[DRY RUN] Fusion pipeline executed in EE graph successfully. Skipping Drive export.")
        return

    # Export Tasks
    file_name = f"synthetic_ndvi_{args.prediction_date.replace('-', '')}"
    print(f"Submitting export for {file_name} to Drive folder '{args.export_folder}'...")
    task = ee.batch.Export.image.toDrive(
        image=synthetic_ndvi,
        description=file_name,
        folder=args.export_folder,
        fileNamePrefix=file_name,
        scale=config.DEFAULT_SCALE,
        region=aoi,
        crs="EPSG:4326",
        maxPixels=1e10
    )
    task.start()
    print(f"Submitted task: {file_name}")

    if args.validate_red:
        val_name = f"red_band_validation_{args.prediction_date.replace('-', '')}"
        val_task = ee.batch.Export.image.toDrive(
            image=validation_stack,
            description=val_name,
            folder=f"{args.export_folder}_Red",
            fileNamePrefix=val_name,
            scale=config.DEFAULT_SCALE,
            region=aoi,
            crs="EPSG:4326",
            maxPixels=1e10
        )
        val_task.start()
        print(f"Submitted task: {val_name}")

    print("\nFusion pipeline complete!")


if __name__ == "__main__":
    main()
