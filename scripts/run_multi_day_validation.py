#!/usr/bin/env python3
"""
STAIR Multi-Day Spatial Correction Validation & Comparative Analysis
===================================================================
Evaluates the STAIR imputation pipeline across multiple dates:
  1. Mid-Summer Peak Greenery: 2021-07-19
  2. Late-Season Crop Harvest/Senescence: 2021-09-28
  3. Peak Summer Canopy: 2021-07-26

For each date, this script:
  - Takes 100% cloud-free Ground Truth
  - Artificially creates a realistic 3 km diameter cloud patch over agricultural fields
  - Runs:
      (a) Temporal Linear Interpolation
      (b) Global Average Correction
      (c) Adaptive Spatial Correction (Seamless with Local K-Means + Alpha Feathering)
  - Computes quantitative accuracy metrics: MAE, RMSE, and Error Reduction %
  - Generates diagnostic multi-panel visual comparisons and Error Difference Heatmaps.
"""

import sys
import os
import io
import time
import urllib.request
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import config
from stair.core import init_earth_engine, get_landsat_collection
from stair.diagnostics import create_chip_bounds
from stair.temporal import impute_temporal
from stair.adaptive import compute_global_correction

def run_validation_for_date(target_date, reference_date, cloud_center_coords=[-84.2333, 41.8611], radius_m=1500, out_dir=None):
    """
    Runs complete STAIR validation on a specific clear day against Ground Truth.
    """
    print(f"\n=======================================================")
    print(f"   VALIDATING TARGET DATE: {target_date}")
    print(f"   Reference Land-Cover Date: {reference_date}")
    print(f"=======================================================")
    
    target_point = ee.Geometry.Point(cloud_center_coords)
    region = create_chip_bounds(target_point, radius_meters=3500)
    scale = 30.0
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]  # NIR, Red, Green
    
    # 1. Load Ground Truth image
    ground_truth_img = get_landsat_collection(
        region, target_date, ee.Date(target_date).advance(1, "day").format("YYYY-MM-dd").getInfo(), apply_mask=False
    ).first()
    
    if ground_truth_img is None:
        raise ValueError(f"Could not load Ground Truth Landsat image for {target_date}")
        
    # 2. Load Reference Image for K-Means land-cover clustering
    image_t1 = get_landsat_collection(
        region, reference_date, ee.Date(reference_date).advance(1, "day").format("YYYY-MM-dd").getInfo(), apply_mask=True
    ).first()
    
    # 3. Create Simulated Cloud Hole
    cloud_geom = target_point.buffer(radius_m)
    cloud_mask = ee.Image.constant(1).clip(region).paint(cloud_geom, 0)
    masked_target = ground_truth_img.updateMask(cloud_mask)
    hole_mask = cloud_mask.eq(0)
    
    # 1km local neighborhood buffer ring
    local_ring = target_point.buffer(radius_m + 1000).difference(cloud_geom)
    
    # 4. Training Collection for Temporal Interpolation
    training_col = get_landsat_collection(region, "2021-01-01", "2021-12-31", max_cloud_cover=20.0, apply_mask=True)
    
    # 5. Precompute Weka K-Means Clusters
    print("  Training Weka K-Means (4 clusters) on reference image...")
    ref_stack = image_t1.select(bands_to_process)
    training_data = ref_stack.sample(region=region, scale=scale, numPixels=5000, tileScale=1)
    clusterer = ee.Clusterer.wekaKMeans(4).train(training_data)
    master_segments = ref_stack.cluster(clusterer).rename("seg")
    
    # 5-pixel radius focal mean for seamless feathering
    feather_weight = cloud_mask.focal_mean(radius=5, kernelType="circle", units="pixels")
    
    metrics = {}
    stage_images = {
        "GT": {},
        "Masked": {},
        "Temporal": {},
        "Global": {},
        "Adaptive": {}
    }
    
    for band in bands_to_process:
        print(f"  -> Processing Band {band}...")
        # Step 1: Temporal Interpolation
        pred_linear, _ = impute_temporal(training_col, masked_target, target_date, band_name=band)
        temp_band = masked_target.select(band).unmask(pred_linear)
        
        # Step 2: Global Correction
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
        
        # Step 4: Feathered Seamless Adaptive
        seamless_adaptive = gt_band.multiply(feather_weight).add(
            local_adaptive.multiply(ee.Image.constant(1).subtract(feather_weight))
        )
        
        stage_images["GT"][band] = gt_band
        stage_images["Masked"][band] = masked_target.select(band)
        stage_images["Temporal"][band] = temp_band
        stage_images["Global"][band] = glob_band
        stage_images["Adaptive"][band] = seamless_adaptive
        
        # Quantitative Metrics INSIDE the Cloud Gap
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
        
        metrics[band] = {
            "MAE_Temporal": mae_t,
            "MAE_Global": mae_g,
            "MAE_Adaptive": mae_a,
            "RMSE_Temporal": rmse_t,
            "RMSE_Global": rmse_g,
            "RMSE_Adaptive": rmse_a,
        }
        print(f"    Band {band}:")
        print(f"      MAE  -> Temporal: {mae_t:7.2f} | Global: {mae_g:7.2f} | Adaptive: {mae_a:7.2f}")
        print(f"      RMSE -> Temporal: {rmse_t:7.2f} | Global: {rmse_g:7.2f} | Adaptive: {rmse_a:7.2f}")
        
    # Generate Visual Figures
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 5-Panel Visual Comparison (NIR Band SR_B5)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        titles = [
            f"1. Ground Truth\n({target_date} Pristine)",
            f"2. Masked Cloud Area\n(Artificial 3km Hole)",
            f"3. Temporal Interpolation\n(Linear Regression)",
            f"4. Global Correction\n(Uniform Scalar Delta)",
            f"5. Seamless Adaptive\n(Local K-Means + Blend)"
        ]
        
        vis_params_b5 = {
            'region': region.getInfo(),
            'scale': 30,
            'min': 8000,
            'max': 22000,
            'palette': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        }
        
        imgs_to_show = [
            stage_images["GT"]["SR_B5"],
            stage_images["Masked"]["SR_B5"],
            stage_images["Temporal"]["SR_B5"],
            stage_images["Global"]["SR_B5"],
            stage_images["Adaptive"]["SR_B5"]
        ]
        
        for ax, title, img in zip(axes, titles, imgs_to_show):
            try:
                thumb_url = img.getThumbURL(vis_params_b5)
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
        plot_path = out_dir / f"visual_comparison_{target_date.replace('-', '')}.png"
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved Visual Comparison Plot: {plot_path}")
        
        # 2. Difference Heatmap (Error Maps: |Prediction - Ground Truth| inside the cloud hole)
        fig_diff, axes_diff = plt.subplots(1, 3, figsize=(18, 5))
        diff_titles = [
            f"Temporal Error Map\n|GT - Temporal| (MAE: {metrics['SR_B5']['MAE_Temporal']:.1f})",
            f"Global Error Map\n|GT - Global| (MAE: {metrics['SR_B5']['MAE_Global']:.1f})",
            f"Adaptive Error Map\n|GT - Adaptive| (MAE: {metrics['SR_B5']['MAE_Adaptive']:.1f})"
        ]
        
        # Difference images masked inside the cloud hole
        err_temporal = stage_images["GT"]["SR_B5"].subtract(stage_images["Temporal"]["SR_B5"]).abs().updateMask(hole_mask)
        err_global = stage_images["GT"]["SR_B5"].subtract(stage_images["Global"]["SR_B5"]).abs().updateMask(hole_mask)
        err_adaptive = stage_images["GT"]["SR_B5"].subtract(stage_images["Adaptive"]["SR_B5"]).abs().updateMask(hole_mask)
        
        vis_err = {
            'region': region.getInfo(),
            'scale': 30,
            'min': 0,
            'max': 3500,
            'palette': ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']  # Yellow (0 error) to Dark Red (high error)
        }
        
        diff_imgs = [err_temporal, err_global, err_adaptive]
        for ax, title, img in zip(axes_diff, diff_titles, diff_imgs):
            try:
                thumb_url = img.getThumbURL(vis_err)
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
        diff_path = out_dir / f"error_heatmap_{target_date.replace('-', '')}.png"
        plt.savefig(diff_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved Error Heatmap Plot: {diff_path}")
        
    return metrics

def main():
    init_earth_engine()
    
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "multi_day_validation"
    
    # Dates to test:
    # 1. July 19, 2021: Mid-Summer Peak Green Vegetative stage
    # 2. September 28, 2021: Late-Season Crop Dry-down / Harvest stage
    # 3. July 26, 2021: Peak Summer Canopy
    scenarios = [
        {"target": "2021-07-19", "ref": "2021-06-17", "name": "Mid-Summer Peak Greenery (2021-07-19)"},
        {"target": "2021-09-28", "ref": "2021-06-17", "name": "Late-Season Crop Senescence / Harvest (2021-09-28)"},
        {"target": "2021-07-26", "ref": "2021-06-17", "name": "Peak Summer Golden Day (2021-07-26)"}
    ]
    
    all_results = {}
    for sc in scenarios:
        res = run_validation_for_date(sc["target"], sc["ref"], out_dir=out_dir)
        all_results[sc["name"]] = res
        
    print("\n\n" + "="*85)
    print("                     CROSS-DATE STAIR VALIDATION SUMMARY TABLE")
    print("="*85)
    print(f"{'Scenario / Date':<40} | {'Band':<7} | {'Temporal MAE':<13} | {'Global MAE':<12} | {'Adaptive MAE':<13} | {'MAE Impv %'}")
    print("-"*85)
    
    for sc_name, bands_dict in all_results.items():
        print(f"\n{sc_name}:")
        for band, m in bands_dict.items():
            t_mae = m["MAE_Temporal"]
            g_mae = m["MAE_Global"]
            a_mae = m["MAE_Adaptive"]
            impv = ((t_mae - a_mae) / t_mae) * 100
            print(f"  {band:<37} | {band:<7} | {t_mae:<13.2f} | {g_mae:<12.2f} | {a_mae:<13.2f} | {impv:+.1f}%")
            
    print("\nValidation complete! All visual figures and quantitative tables ready.")

if __name__ == "__main__":
    main()
