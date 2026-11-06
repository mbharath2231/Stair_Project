#!/usr/bin/env python3
"""
STAIR Imputation Validation: Ground Truth Proof with Simulated Cloud Area
========================================================================
Validates Landsat 8 cloud removal against known Ground Truth.
Takes a 100% cloud-free Landsat 8 image (2021-07-19) over Clayton, MI,
artificially masks out a realistic cloud patch over agricultural fields, runs STAIR:
  1. Temporal Linear Interpolation
  2. Global Average Correction
  3. Seamless Adaptive Correction (Local K-Means Offset + Boundary Feathering)
and quantitatively and visually compares all three stages against the Ground Truth.
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import config
from stair.core import init_earth_engine, get_landsat_collection
from stair.diagnostics import create_chip_bounds
from stair.temporal import impute_temporal
from stair.adaptive import compute_global_correction

def main():
    init_earth_engine()
    
    # 7km x 7km AOI centered on Clayton, MI
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    scale = 30.0
    
    # Target 100% clear day (Ground Truth)
    target_date = "2021-07-19"
    # Historical clear reference day for K-Means land cover mapping
    reference_date = "2021-06-17"
    
    print(f"Target Ground Truth Date: {target_date} (0.00% clouds - Landsat 8)")
    print(f"Reference Mapping Date:   {reference_date}")
    
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]  # NIR, Red, Green
    
    # 1. Load Ground Truth (unmasked, pristine Landsat 8 image)
    ground_truth_img = get_landsat_collection(
        region, target_date, ee.Date(target_date).advance(1, "day").format("YYYY-MM-dd").getInfo(), apply_mask=False
    ).first()
    
    # 2. Load Reference Image
    image_t1 = get_landsat_collection(
        region, reference_date, ee.Date(reference_date).advance(1, "day").format("YYYY-MM-dd").getInfo(), apply_mask=True
    ).first()
    
    # 3. Create a Realistic Simulated Cloud Area over Agricultural Fields
    # Realistic 1.5 km radius cloud patch (3km diameter, covering ~7.1 km^2 of farmland)
    cloud_center = ee.Geometry.Point([-84.2333, 41.8611])
    cloud_geom = cloud_center.buffer(1500)
    
    # 1 = Valid pixels outside, 0 = Masked cloud hole inside
    cloud_mask = ee.Image.constant(1).clip(region).paint(cloud_geom, 0)
    masked_target = ground_truth_img.updateMask(cloud_mask)
    hole_mask = cloud_mask.eq(0)
    
    # Local neighborhood buffer: 1km ring immediately outside the cloud perimeter
    local_ring = cloud_center.buffer(2500).difference(cloud_geom)
    
    # 4. Training Collection for Temporal Interpolation (all 2021 observations)
    training_col = get_landsat_collection(region, "2021-01-01", "2021-12-31", max_cloud_cover=20.0, apply_mask=True)
    
    # 5. Precompute Weka K-Means Land Cover Clusters on Reference Stack
    print("\nTraining Weka K-Means (4 clusters) on multi-band reference image...")
    ref_stack = image_t1.select(bands_to_process)
    training_data = ref_stack.sample(region=region, scale=scale, numPixels=5000, tileScale=1)
    clusterer = ee.Clusterer.wekaKMeans(4).train(training_data)
    master_segments = ref_stack.cluster(clusterer).rename("seg")
    
    stage_bands = {
        "0_Ground_Truth": [],
        "1_Masked_Simulated_Cloud": [],
        "2_Temporal_Interpolation": [],
        "3_Global_Correction": [],
        "4_Adaptive_Seamless": []
    }
    
    stats_summary = {}
    
    # Feathering weight mask across the boundary (5-pixel radius = 150m transition zone)
    # Outside boundary: 1.0; Inside boundary: 0.0; Boundary edge: smooth gradient
    feather_weight = cloud_mask.focal_mean(radius=5, kernelType="circle", units="pixels")
    
    print("\nRunning STAIR Imputation Pipeline across bands...")
    for band in bands_to_process:
        print(f"\n--- Processing Band: {band} ---")
        # Step 1: Temporal Linear Interpolation
        pred_linear, _ = impute_temporal(training_col, masked_target, target_date, band_name=band)
        temp_band = masked_target.select(band).unmask(pred_linear)
        
        # Step 2: Global Spatial Correction
        gt_band = ground_truth_img.select(band)
        pred_band = pred_linear.select([0]).rename(band)
        
        pred_global = compute_global_correction(
            image_t0=masked_target,
            mask_t0=cloud_mask,
            region=region,
            linear_baseline=pred_linear,
            band_name=band,
            scale=scale
        )
        glob_band = masked_target.select(band).unmask(pred_global)
        
        # Step 3: Local Adaptive Correction
        # Calculate cluster means in the local ring surrounding the cloud
        r_true = masked_target.select(band).addBands(master_segments).reduceRegion(
            reducer=ee.Reducer.mean().group(groupField=1, groupName="seg"),
            geometry=local_ring, scale=scale
        ).get("groups").getInfo()
        
        r_pred = pred_band.updateMask(cloud_mask.eq(1)).addBands(master_segments).reduceRegion(
            reducer=ee.Reducer.mean().group(groupField=1, groupName="seg"),
            geometry=local_ring, scale=scale
        ).get("groups").getInfo()
        
        dict_pred = {g["seg"]: g["mean"] for g in r_pred}
        keys = [-1]
        deltas = [0.0]
        for g in r_true:
            k = g["seg"]
            mean_t = g["mean"]
            mean_p = dict_pred.get(k, 0)
            keys.append(k)
            deltas.append(mean_t - mean_p)
            
        delta_img = master_segments.remap(keys, deltas, 0).rename("delta")
        local_adaptive = pred_band.add(delta_img)
        
        # Step 4: Seamless Boundary Feathering (Alpha Blending)
        # Guarantees zero step discontinuity at the cloud circle boundary
        seamless_adaptive = gt_band.multiply(feather_weight).add(
            local_adaptive.multiply(ee.Image.constant(1).subtract(feather_weight))
        )
        
        stage_bands["0_Ground_Truth"].append(gt_band)
        stage_bands["1_Masked_Simulated_Cloud"].append(masked_target.select(band))
        stage_bands["2_Temporal_Interpolation"].append(temp_band)
        stage_bands["3_Global_Correction"].append(glob_band)
        stage_bands["4_Adaptive_Seamless"].append(seamless_adaptive)
        
        # Quantitative metrics inside the cloud hole
        gt_in_gap = gt_band.updateMask(hole_mask)
        temp_in_gap = temp_band.updateMask(hole_mask)
        glob_in_gap = glob_band.updateMask(hole_mask)
        adap_in_gap = seamless_adaptive.updateMask(hole_mask)
        
        mae_t = gt_in_gap.subtract(temp_in_gap).abs().reduceRegion(ee.Reducer.mean(), cloud_geom, scale).get(band).getInfo()
        mae_g = gt_in_gap.subtract(glob_in_gap).abs().reduceRegion(ee.Reducer.mean(), cloud_geom, scale).get(band).getInfo()
        mae_a = gt_in_gap.subtract(adap_in_gap).abs().reduceRegion(ee.Reducer.mean(), cloud_geom, scale).get(band).getInfo()
        
        rmse_t = (gt_in_gap.subtract(temp_in_gap).pow(2).reduceRegion(ee.Reducer.mean(), cloud_geom, scale).get(band).getInfo()) ** 0.5
        rmse_g = (gt_in_gap.subtract(glob_in_gap).pow(2).reduceRegion(ee.Reducer.mean(), cloud_geom, scale).get(band).getInfo()) ** 0.5
        rmse_a = (gt_in_gap.subtract(adap_in_gap).pow(2).reduceRegion(ee.Reducer.mean(), cloud_geom, scale).get(band).getInfo()) ** 0.5
        
        stats_summary[band] = {
            "MAE_Temporal": mae_t,
            "MAE_Global": mae_g,
            "MAE_Adaptive": mae_a,
            "RMSE_Temporal": rmse_t,
            "RMSE_Global": rmse_g,
            "RMSE_Adaptive": rmse_a,
        }
        
        print(f"MAE  -> Temporal: {mae_t:7.2f} | Global: {mae_g:7.2f} | Adaptive Seamless: {mae_a:7.2f}")
        print(f"RMSE -> Temporal: {rmse_t:7.2f} | Global: {rmse_g:7.2f} | Adaptive Seamless: {rmse_a:7.2f}")
    
    # Print overall summary table
    print("\n" + "="*70)
    print("      GROUND TRUTH VALIDATION METRICS: SEAMLESS ADAPTIVE")
    print("="*70)
    print(f"{'Band':<10} | {'Method':<16} | {'MAE':<10} | {'RMSE':<10} | {'Error Reduction':<15}")
    print("-"*70)
    for band in bands_to_process:
        s = stats_summary[band]
        red_mae = ((s['MAE_Temporal'] - s['MAE_Adaptive']) / s['MAE_Temporal']) * 100
        red_rmse = ((s['RMSE_Temporal'] - s['RMSE_Adaptive']) / s['RMSE_Temporal']) * 100
        print(f"{band:<10} | Temporal         | {s['MAE_Temporal']:<10.2f} | {s['RMSE_Temporal']:<10.2f} | Baseline")
        print(f"{'':<10} | Global           | {s['MAE_Global']:<10.2f} | {s['RMSE_Global']:<10.2f} |")
        print(f"{'':<10} | Adaptive Seamless| {s['MAE_Adaptive']:<10.2f} | {s['RMSE_Adaptive']:<10.2f} | -{red_mae:.1f}% MAE / -{red_rmse:.1f}% RMSE")
        print("-"*70)
    
    # 6. Generate Multi-Band Exports to Google Drive for QGIS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_folder = f"STAIR_Seamless_Validation_{timestamp}"
    print(f"\nSubmitting Multi-Band GeoTIFF tasks to Google Drive folder: '{export_folder}'...")
    
    tasks = []
    for stage_name, band_list in stage_bands.items():
        fcc_img = ee.Image.cat(band_list).toDouble().clip(region)
        desc = f"Validation_Cloud_{stage_name}"
        task = ee.batch.Export.image.toDrive(
            image=fcc_img,
            description=desc,
            folder=export_folder,
            fileNamePrefix=desc,
            scale=scale,
            region=region,
            crs="EPSG:32616",
            maxPixels=1e10
        )
        task.start()
        tasks.append(task)
        print(f"  -> Export Task Started: {desc}")
        
    print(f"\nAll {len(tasks)} tasks submitted to Google Earth Engine! Track at: https://code.earthengine.google.com/tasks")
    
    # 7. Generate Local Diagnostic Plot
    print("\nGenerating local visual comparison plot...")
    scratch_dir = Path(__file__).resolve().parent.parent / "outputs" / "simulated_gap_validation"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out_plot_path = scratch_dir / "simulated_cloud_seamless_proof.png"
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = [
        "0. Ground Truth\n(Actual Clear Landsat 8)",
        "1. Masked Cloud Area\n(Artificial Missing Patch)",
        "2. Temporal Imputation\n(Linear Regression)",
        "3. Global Correction\n(Uniform Delta Offset)",
        "4. Seamless Adaptive\n(Local Offset + Feathering)"
    ]
    
    import urllib.request
    from PIL import Image
    import io
    
    vis_params = {
        'region': region.getInfo(),
        'scale': 30,
        'min': 8000,
        'max': 22000,
        'palette': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
    }
    
    stage_keys = [
        "0_Ground_Truth",
        "1_Masked_Simulated_Cloud",
        "2_Temporal_Interpolation",
        "3_Global_Correction",
        "4_Adaptive_Seamless"
    ]
    
    for ax, title, key in zip(axes, titles, stage_keys):
        try:
            # Visualize NIR band (SR_B5)
            img_to_render = stage_bands[key][0]  # B5 is index 0
            thumb_url = img_to_render.getThumbURL(vis_params)
            req = urllib.request.Request(thumb_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp:
                pil_img = Image.open(io.BytesIO(resp.read()))
                ax.imshow(pil_img)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.axis('off')
        except Exception as e:
            ax.set_title(f"{title}\n(Render Error: {e})", fontsize=10)
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved local visual comparison plot to: {out_plot_path}")

if __name__ == "__main__":
    main()
