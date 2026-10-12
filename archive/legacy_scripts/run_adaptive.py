#!/usr/bin/env python3
"""
CLI Runner for STAIR Section 2.2.2 (Spatial & Adaptive K-Means Correction)
Supports localized 5km demo patch and full-watershed execution.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import config
from stair.core import init_earth_engine, load_aoi, get_landsat_collection
from stair.temporal import compute_temporal_regression, impute_temporal
from stair.adaptive import stair_adaptive_pipeline
from stair.diagnostics import create_chip_bounds


def main():
    parser = argparse.ArgumentParser(description="STAIR Section 2.2.2: Spatial & Adaptive Correction")
    parser.add_argument("--mode", choices=["5km", "full_watershed"], default="5km", help="Region mode")
    parser.add_argument("--target-date", default="2021-07-03", help="Broken target date (YYYY-MM-DD)")
    parser.add_argument("--reference-date", default=config.DEFAULT_REFERENCE_DATE, help="Clear reference date")
    parser.add_argument("--start-date", default="2021-06-01", help="Regression window start")
    parser.add_argument("--end-date", default="2021-07-05", help="Regression window end")
    parser.add_argument("--clusters", type=int, default=config.DEFAULT_KMEANS_CLUSTERS, help="K-Means clusters")
    parser.add_argument("--band", default="SR_B5", help="Band name (default: SR_B5)")
    parser.add_argument("--project-id", default=config.EE_PROJECT_ID, help="EE project ID")
    parser.add_argument("--export-folder", default="STAIR_Adaptive_Restoration", help="Drive folder")
    parser.add_argument("--dry-run", action="store_true", help="Skip Google Drive export")
    parser.add_argument("--wait", action="store_true", help="Wait and monitor Google Drive export tasks until completion")
    args = parser.parse_args()

    print(f"\n{'='*60}\nSTAIR 2.2.2 ADAPTIVE CORRECTION PIPELINE ({args.mode.upper()})\n{'='*60}")
    init_earth_engine(project_id=args.project_id)
    aoi, _ = load_aoi()

    # Determine region boundary
    if args.mode == "5km":
        centroid = aoi.centroid(maxError=1).getInfo()["coordinates"]
        target_point = ee.Geometry.Point([float(centroid[0]), float(centroid[1])])
        region = create_chip_bounds(target_point, radius_meters=2500)
        scale = 30.0
        tile_scale = 1
        print("Using 5km x 5km localized chip around watershed geographic center.")
    else:
        region = aoi
        scale = 120.0  # Coarser scale for memory-efficient full watershed computation
        tile_scale = 8
        print("Using full watershed AOI with tileScale=8.")

    # 1. Target broken image
    t0_start = ee.Date(args.target_date)
    target_col = get_landsat_collection(region, t0_start, t0_start.advance(1, "day"), apply_mask=True)
    raw_image_t0 = target_col.first()

    # 2. Reference clear image
    t1_start = ee.Date(args.reference_date)
    ref_col = get_landsat_collection(region, t1_start, t1_start.advance(1, "day"), apply_mask=True)
    image_t1 = ref_col.first()

    # 3. Training collection for temporal regression
    training_col = get_landsat_collection(
        region,
        args.start_date,
        args.end_date,
        max_cloud_cover=config.DEFAULT_CLOUD_THRESHOLD,
        apply_mask=True
    )

    print("Fitting temporal regression baseline...")
    slope, intercept = compute_temporal_regression(training_col, band_name=args.band)
    pred_linear, _ = impute_temporal(raw_image_t0, slope, intercept, args.target_date, band_name=args.band)

    print("Running adaptive stages (Global offset, K-Means clustering, Edge blending)...")
    results = stair_adaptive_pipeline(
        raw_image_t0=raw_image_t0,
        image_t1=image_t1,
        linear_prediction=pred_linear,
        region=region,
        n_clusters=args.clusters,
        band_name=args.band,
        scale=scale,
        tile_scale=tile_scale
    )

    # Diagnostic Staircase Tracker (0 = Original Valid, 1 = Temporal Patch, 2 = Spatial Patch)
    from stair.diagnostics import create_staircase_tracker
    raw_band_mask = raw_image_t0.select(args.band).mask()
    temporal_mask = pred_linear.mask()
    diagnostic_tracker = create_staircase_tracker(raw_mask=raw_band_mask, temporal_mask=temporal_mask, aoi=region)

    # Calculate Difference Raster: Adaptive (Homogeneous) - Global (Heterogeneous)
    diff_adaptive_vs_global = (
        results["stage3b_adaptive"]
        .subtract(results["stage3a_global"])
        .rename("Diff_Adaptive_vs_Global")
    )

    if args.dry_run:
        print("[DRY RUN] All stages assembled in EE memory. Skipping Google Drive export.")
        return

    date_tag = args.target_date.replace("-", "")
    print(f"Submitting 8 export tasks to Google Drive folder: '{args.export_folder}'...")
    export_layers = [
        (results["stage1_broken"].toDouble(), "Stage1_Broken"),
        (results["stage2_temporal"].toDouble(), "Stage2_Temporal"),
        (results["stage3a_global"].toDouble(), "Stage3a_Global"),
        (results["stage3b_adaptive"].toDouble(), "Stage3b_Adaptive"),
        (results["stage3c_blended"].toDouble(), "Stage3c_Flawless"),
        (results["segments"], "Stage4_KMeans_Clusters"),
        (diff_adaptive_vs_global.toDouble(), "Diff_Adaptive_vs_Global"),
        (diagnostic_tracker.toInt(), "Diagnostic_Fill_Tracker"),
    ]

    tasks = []
    for img, name in export_layers:
        desc = f"STAIR_{args.mode}_{date_tag}_{name}"
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=desc,
            folder=args.export_folder,
            fileNamePrefix=desc,
            scale=scale,
            region=region,
            maxPixels=1e10
        )
        task.start()
        tasks.append(task)
        print(f"Submitted task: {desc}")

    print("\nTasks submitted! Check Google Earth Engine Tasks manager or Google Drive folder.")

    if args.wait:
        print("\n[--wait enabled] Monitoring export tasks on Google Cloud...")
        import time
        while any(t.active() for t in tasks):
            statuses = [f"{t.status().get('description')}: {t.status().get('state')}" for t in tasks]
            print(f"Status update: {statuses}")
            time.sleep(15)
        print("\nAll export tasks completed! Download the GeoTIFFs from Google Drive for QGIS.")


if __name__ == "__main__":
    main()
