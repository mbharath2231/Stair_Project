#!/usr/bin/env python3
"""
CLI Runner for STAIR Section 2.2.1 (Temporal Linear Interpolation)
Supports testing different scenes, real cloud cover levels, and simulated cloud gaps.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import config
from stair.core import init_earth_engine, load_aoi, get_landsat_collection, mask_landsat_clouds
from stair.temporal import (
    compute_temporal_regression,
    compute_regression_metrics,
    compute_temporal_support,
    impute_temporal,
)


def list_available_scenes(aoi):
    """List available Landsat 8 scenes over AOI with cloud cover percentages."""
    print(f"\n{'='*70}\nAVAILABLE LANDSAT 8 SCENES OVER AOI (2021 GROWING SEASON)\n{'='*70}")
    col = (
        ee.ImageCollection(config.LANDSAT8_COLLECTION)
        .filterBounds(aoi)
        .filterDate("2021-04-01", "2021-11-01")
        .sort("system:time_start")
    )
    info = col.getInfo()
    print(f"{'Date':<12} | {'Scene ID':<25} | {'Scene Clouds':<14} | {'AOI Cloud Cover':<15}")
    print("-" * 70)
    for f in info["features"]:
        props = f["properties"]
        date_str = ee.Date(props["system:time_start"]).format("YYYY-MM-dd").getInfo()
        cc_scene = props.get("CLOUD_COVER", -1)
        scene_id = props.get("system:index", "")
        
        # Calculate AOI specific cloud cover
        raw_img = ee.Image(f["id"])
        qa = raw_img.select("QA_PIXEL")
        clear_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0)).clip(aoi)
        stats = clear_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=120,
            maxPixels=1e9
        ).getInfo()
        clear_ratio = stats.get("QA_PIXEL", 0)
        aoi_cc = (1.0 - clear_ratio) * 100.0 if clear_ratio is not None else -1
        
        print(f"{date_str:<12} | {scene_id:<25} | {cc_scene:5.1f}%        | {aoi_cc:5.1f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="STAIR Section 2.2.1: Temporal Linear Imputation")
    parser.add_argument("--target-date", default=config.DEFAULT_TARGET_DATE, help="Date to restore (YYYY-MM-DD)")
    parser.add_argument("--start-date", default="2021-05-15", help="Reference window start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2021-08-15", help="Reference window end date (YYYY-MM-DD)")
    parser.add_argument("--band", default="SR_B5", help="Band name to interpolate (default: SR_B5)")
    parser.add_argument("--project-id", default=config.EE_PROJECT_ID, help="Earth Engine project ID")
    parser.add_argument("--export-folder", default="STAIR_Temporal_Interpolation", help="Google Drive folder name")
    parser.add_argument("--dry-run", action="store_true", help="Build graph without submitting Drive export tasks")
    parser.add_argument("--wait", action="store_true", help="Wait and monitor Google Drive export tasks until completion")
    parser.add_argument("--list-scenes", action="store_true", help="Scan and list all available Landsat scenes and cloud %")
    parser.add_argument("--simulated-gap", action="store_true", help="Create an artificial circular cloud hole on a clear day to test ground truth accuracy")
    args = parser.parse_args()

    init_earth_engine(project_id=args.project_id)
    aoi, _ = load_aoi()

    if args.list_scenes:
        list_available_scenes(aoi)
        return

    print(f"\n{'='*60}\nSTAIR 2.2.1 TEMPORAL INTERPOLATION PIPELINE\n{'='*60}")
    print(f"Target Date: {args.target_date}")
    print(f"Reference Window: {args.start_date} to {args.end_date}")
    print(f"Band: {args.band}")
    print(f"Google Drive Destination: '{args.export_folder}'")

    # 1. Build reference training collection with cloud masking
    landsat_col = get_landsat_collection(
        aoi=aoi,
        start_date=args.start_date,
        end_date=args.end_date,
        apply_mask=True
    )

    print("Computing pixel-wise temporal regression, evaluation metrics, and temporal support (N_Before / N_After)...")
    slope, intercept = compute_temporal_regression(landsat_col, band_name=args.band)
    eval_matrix = compute_regression_metrics(landsat_col, band_name=args.band, aoi=aoi)
    support_stack = compute_temporal_support(landsat_col, target_date=args.target_date, band_name=args.band, aoi=aoi)

    # 2. Extract target date raw image
    target_start = ee.Date(args.target_date)
    target_end = target_start.advance(1, "day")
    raw_col = get_landsat_collection(aoi=aoi, start_date=target_start, end_date=target_end, apply_mask=False)
    
    if raw_col.size().getInfo() == 0:
        raise ValueError(f"No Landsat 8 scenes found on {args.target_date}. Use --list-scenes to see available dates.")
        
    raw_unmasked = raw_col.first()
    
    # 3. Apply mask (Real clouds or Simulated gap)
    if args.simulated_gap:
        print("[SIMULATED GAP] Punching an artificial 4km circular cloud hole around watershed center...")
        centroid = aoi.centroid(maxError=1).getInfo()["coordinates"]
        hole_geom = ee.Geometry.Point([centroid[0], centroid[1]]).buffer(4000)
        # 0 inside hole, 1 outside
        hole_mask = ee.Image(1).clip(aoi).paint(hole_geom, 0)
        # Also apply standard cloud mask
        target_img = mask_landsat_clouds(raw_unmasked).updateMask(hole_mask)
    else:
        target_img = mask_landsat_clouds(raw_unmasked)

    # Calculate actual clear vs missing percentage on target scene
    clear_mask = target_img.select(args.band).mask().clip(aoi)
    stats = clear_mask.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=120, maxPixels=1e9).getInfo()
    clear_pct = stats.get(args.band, 0) * 100.0
    missing_pct = 100.0 - clear_pct
    print(f"Target Scene: {clear_pct:.1f}% valid data | {missing_pct:.1f}% cloud/missing pixels to restore.")

    # 4. Temporal Imputation
    predicted, patched = impute_temporal(
        raw_image=target_img,
        slope=slope,
        intercept=intercept,
        target_date=args.target_date,
        band_name=args.band
    )

    date_tag = args.target_date.replace("-", "")
    stage1_img = target_img.select(args.band).toDouble().clip(aoi)
    stage2_img = patched.toDouble().clip(aoi)
    eval_matrix_img = eval_matrix.toDouble().clip(aoi)
    support_stack_img = support_stack.toInt().clip(aoi)

    if args.dry_run:
        print("[DRY RUN] Pipeline graph constructed successfully. Skipping Google Drive export.")
        return

    # 5. Build export task list
    tasks = [
        ee.batch.Export.image.toDrive(
            image=stage1_img,
            description=f"STAIR_Raw_{date_tag}",
            folder=args.export_folder,
            fileNamePrefix=f"STAIR_Raw_{date_tag}",
            scale=config.DEFAULT_SCALE,
            region=aoi,
            maxPixels=1e10,
        ),
        ee.batch.Export.image.toDrive(
            image=stage2_img,
            description=f"STAIR_Imputed_{date_tag}",
            folder=args.export_folder,
            fileNamePrefix=f"STAIR_Imputed_{date_tag}",
            scale=config.DEFAULT_SCALE,
            region=aoi,
            maxPixels=1e10,
        ),
        ee.batch.Export.image.toDrive(
            image=eval_matrix_img,
            description=f"STAIR_Eval_Matrix_{date_tag}",
            folder=args.export_folder,
            fileNamePrefix=f"STAIR_Eval_Matrix_{date_tag}",
            scale=config.DEFAULT_SCALE,
            region=aoi,
            maxPixels=1e10,
        ),
        ee.batch.Export.image.toDrive(
            image=support_stack_img,
            description=f"STAIR_Temporal_Support_{date_tag}",
            folder=args.export_folder,
            fileNamePrefix=f"STAIR_Temporal_Support_{date_tag}",
            scale=config.DEFAULT_SCALE,
            region=aoi,
            maxPixels=1e10,
        ),
    ]

    # If simulated gap, also export Ground Truth and Residual Error for accuracy measurement
    if args.simulated_gap:
        ground_truth = raw_unmasked.select(args.band).toDouble().clip(aoi)
        residual_error = stage2_img.subtract(ground_truth).abs().rename("Residual_Error")
        tasks.extend([
            ee.batch.Export.image.toDrive(
                image=ground_truth,
                description=f"STAIR_Truth_{date_tag}",
                folder=args.export_folder,
                fileNamePrefix=f"STAIR_Truth_{date_tag}",
                scale=config.DEFAULT_SCALE,
                region=aoi,
                maxPixels=1e10,
            ),
            ee.batch.Export.image.toDrive(
                image=residual_error,
                description=f"STAIR_Residual_Error_{date_tag}",
                folder=args.export_folder,
                fileNamePrefix=f"STAIR_Residual_Error_{date_tag}",
                scale=config.DEFAULT_SCALE,
                region=aoi,
                maxPixels=1e10,
            ),
        ])

    print(f"Submitting {len(tasks)} export tasks to Google Drive folder: '{args.export_folder}'...")
    for t in tasks:
        t.start()
        print(f"  -> Submitted: {t.status().get('description')}")

    print("\nTasks submitted! Check Google Earth Engine Tasks tab or Google Drive folder.")

    if args.wait:
        print("\n[--wait enabled] Monitoring export tasks on Google Cloud...")
        while any(t.active() for t in tasks):
            statuses = [f"{t.status().get('description')}: {t.status().get('state')}" for t in tasks]
            print(f"Status update: {statuses}")
            time.sleep(15)
        print("\nAll export tasks completed! Download the GeoTIFFs from Google Drive for QGIS.")


if __name__ == "__main__":
    main()
