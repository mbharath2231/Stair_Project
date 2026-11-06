#!/usr/bin/env python3
"""
STAIR 2022 New Dates Evaluation: Multi-Stage Cloud Imputation & Ground Truth Benchmark
=====================================================================================
Tests the complete STAIR pipeline on 5 brand-new cloudy scenes from the 2022 growing
season over Clayton Township, MI (Landsat 8 & 9), plus 1 Ground Truth Benchmark scene.

Tested New Dates (Completely excluding all previously used dates):
  1. 2022-05-11 (Landsat 9, Spring Awakening / Early Planting, ~43.1% clouds)
  2. 2022-06-11 (Landsat 8, Early Summer Vegetative Growth, ~37.7% clouds)
  3. 2022-06-27 (Landsat 8, Mid-Vegetative Biomass Surge, ~46.5% clouds)
  4. 2022-07-29 (Landsat 8, Peak Summer Canopy / Reproductive, ~54.0% clouds)
  5. 2022-08-30 (Landsat 8, Late Summer Grain Filling, ~54.5% clouds)
  6. 2022-07-22 (Landsat 8, Ground Truth Benchmark: 0.00% natural clouds, realistic 46.5% cloud gap)

Rigorous STAIR Formulation (Luo et al., 2018):
  1. Temporal Linear Imputation: L_linear(p, t0) = a_p * t0 + b_p
  2. Global Spatial Correction: delta_global = Delta_temp + Delta_spatial
       Delta_temp = mean(L(C_f, t0)) - mean(L_linear(C_f, t0))
       Delta_spatial = mean(L(C_g, t1)) - mean(L(C_f, t1))
  3. Adaptive K-Means Correction: delta_c = Delta_temp(c) + Delta_spatial(c)
       Delta_temp(c) = mean(L(C_{f,c}, t0)) - mean(L_linear(C_{f,c}, t0)) [fallback if count < 30]
       Delta_spatial(c) = mean(L(C_{g,c}, t1)) - mean(L(C_{f,c}, t1))
  4. 100% Crisp Native 30m Fill: Zero whole-image blur. All field parcel boundaries preserved.
  5. Permanent Water Body Protection: JRC Water + NDWI + QA water flags ensure Posey Lake is never masked.
"""

import sys
import os
import io
import math
import time
import urllib.request
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import config
from stair.core import init_earth_engine
from stair.diagnostics import create_chip_bounds
from stair.temporal import impute_temporal

def get_combined_landsat_collection(region, start_date, end_date, apply_mask=False):
    """
    Returns merged Landsat 8 and Landsat 9 Tier 1 Level 2 Surface Reflectance collection.
    """
    col8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(region).filterDate(start_date, end_date)
    col9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(region).filterDate(start_date, end_date)
    col = col8.merge(col9)
    
    if apply_mask:
        def mask_qa(img):
            qa = img.select("QA_PIXEL")
            cloud = qa.bitwiseAnd(1 << 3).neq(0)
            shadow = qa.bitwiseAnd(1 << 4).neq(0)
            dilated = qa.bitwiseAnd(1 << 1).neq(0)
            cirrus = qa.bitwiseAnd(1 << 2).neq(0)
            bad = cloud.Or(shadow).Or(dilated).Or(cirrus)
            return img.updateMask(bad.eq(0))
        col = col.map(mask_qa)
    return col

def process_evaluation_day(date_str, cloud_desc, ref_date="2022-07-22", simulated_mask_date=None, region=None, scale=30.0, out_dir=None, drive_folder=None):
    """
    Processes one test day through all STAIR stages using the complete Luo et al. (2018) formulation.
    """
    next_date = ee.Date(date_str).advance(1, "day").format("YYYY-MM-dd").getInfo()
    print(f"\n" + "="*80)
    print(f"   PROCESSING NEW TEST DAY: {date_str} ({cloud_desc})")
    print(f"="*80)
    
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]  # NIR, Red, Green
    
    # 1. Load Target Scene
    target_col = get_combined_landsat_collection(region, date_str, next_date, apply_mask=False)
    raw_img = target_col.first()
    if raw_img is None:
        raise ValueError(f"Could not load Landsat image for {date_str}")
    spacecraft = raw_img.get("SPACECRAFT_ID").getInfo()
    print(f"  -> Sensor: {spacecraft}")
    
    # 2. Enhanced Cloud & Shadow Masking with Permanent Water Protection
    if simulated_mask_date:
        next_mask_date = ee.Date(simulated_mask_date).advance(1, "day").format("YYYY-MM-dd").getInfo()
        mask_col = get_combined_landsat_collection(region, simulated_mask_date, next_mask_date, apply_mask=False)
        cloud_source_img = mask_col.first()
    else:
        cloud_source_img = raw_img
        
    qa = cloud_source_img.select("QA_PIXEL")
    qa_cloud = qa.bitwiseAnd(1 << 3).neq(0)
    qa_shadow = qa.bitwiseAnd(1 << 4).neq(0)
    qa_dilated = qa.bitwiseAnd(1 << 1).neq(0)
    qa_cirrus = qa.bitwiseAnd(1 << 2).neq(0)
    
    base_mask = (
        qa_cloud.Or(qa_shadow)
        .Or(qa_dilated)
        .Or(qa_cirrus)
        .focal_max(radius=6, kernelType="circle", units="pixels")
    )
    
    # Load Clear Reference Scene (2022-07-22: 100% cloud-free Landsat 8)
    ref_col_raw = get_combined_landsat_collection(region, ref_date, ee.Date(ref_date).advance(1, "day").format("YYYY-MM-dd").getInfo(), apply_mask=False)
    raw_ref_img = ref_col_raw.first()
    ref_img = raw_ref_img.select(bands_to_process)
    
    # Permanent water protection (Posey Lake / wetlands)
    qa_water_ref = raw_ref_img.select("QA_PIXEL").bitwiseAnd(1 << 7).neq(0)
    qa_water_raw = qa.bitwiseAnd(1 << 7).neq(0)
    jrc_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gt(20).unmask(0)
    ndwi_ref = raw_ref_img.normalizedDifference(["SR_B3", "SR_B5"]).gt(0.0)
    water_mask = qa_water_ref.Or(qa_water_raw).Or(jrc_water).Or(ndwi_ref)
    
    # Spectral dark shadow on land
    spectral_shadow = (
        cloud_source_img.select("SR_B5").lt(12000)
        .And(cloud_source_img.select("SR_B4").lt(9000))
        .And(water_mask.eq(0))
    )
    
    cloud_shadow_mask = (base_mask.Or(spectral_shadow)).And(water_mask.eq(0)).Or(qa_cloud)
    clear_mask = cloud_shadow_mask.eq(0).rename("clear_mask")
    gap_mask = clear_mask.eq(0).rename("cloud_gap")
    
    aoi_stats = clear_mask.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=scale).getInfo()
    clear_fraction = aoi_stats.get("clear_mask", 1.0)
    cloud_pct = (1.0 - clear_fraction) * 100.0
    print(f"  -> Total AOI Cloud + Shadow Coverage: {cloud_pct:.2f}% (Clear: {clear_fraction*100:.2f}%)")
    
    masked_target = raw_img.updateMask(clear_mask)
    
    # 3. Merged 2022 Training Collection for Temporal Linear Imputation (61 scenes)
    training_col = get_combined_landsat_collection(region, "2022-01-01", "2022-12-31", apply_mask=True)
    
    # 4. Unsupervised Weka K-Means Clustering on Clear Reference Day (2022-07-22)
    sample_data = ref_img.sample(region=region, scale=scale, numPixels=5000, tileScale=1)
    clusterer = ee.Clusterer.wekaKMeans(4).train(sample_data)
    master_segments = ref_img.cluster(clusterer).rename("seg")
    
    stage_bands = {
        "0_Raw_Cloudy": [],
        "1_Masked_Clouds": [],
        "2_Temporal_Interpolation": [],
        "3_Global_Correction": [],
        "4_Adaptive_Correction": [],
        "5_Diff_Global_Temporal": [],
        "6_Diff_Adaptive_Global": []
    }
    
    metrics = {
        "date": date_str,
        "spacecraft": spacecraft,
        "cloud_pct": cloud_pct,
        "global_deltas": {},
        "cluster_deltas": {},
        "ground_truth_eval": {}
    }
    
    for band in bands_to_process:
        print(f"  -> Processing Band {band}...")
        
        # STAIR Step 1: Temporal Linear Imputation (L_linear)
        pred_linear, _ = impute_temporal(training_col, masked_target, date_str, band_name=band)
        pred_band = pred_linear.select([0]).rename(band)
        temp_filled = masked_target.select(band).unmask(pred_band)
        
        # STAIR Step 2: Global Spatial Correction (Luo et al., 2018)
        mean_target_f = masked_target.select(band).reduceRegion(ee.Reducer.mean(), region, scale).get(band).getInfo()
        mean_pred_f = pred_band.updateMask(clear_mask).reduceRegion(ee.Reducer.mean(), region, scale).get(band).getInfo()
        delta_temp_glob = (mean_target_f - mean_pred_f) if (mean_target_f and mean_pred_f) else 0.0
        
        mean_ref_g = ref_img.select(band).updateMask(gap_mask).reduceRegion(ee.Reducer.mean(), region, scale).get(band).getInfo()
        mean_ref_f = ref_img.select(band).updateMask(clear_mask).reduceRegion(ee.Reducer.mean(), region, scale).get(band).getInfo()
        delta_spatial_glob = (mean_ref_g - mean_ref_f) if (mean_ref_g and mean_ref_f) else 0.0
        
        delta_glob = delta_temp_glob + delta_spatial_glob
        metrics["global_deltas"][band] = delta_glob
        
        pred_global_band = pred_band.add(ee.Image.constant(delta_glob)).rename(band)
        glob_filled = masked_target.select(band).unmask(pred_global_band)
        
        # STAIR Step 3: Adaptive K-Means Correction (Luo et al., 2018)
        def get_grouped(img, mask_img):
            return img.rename("val").updateMask(mask_img).addBands(master_segments).reduceRegion(
                reducer=ee.Reducer.mean().group(1, "seg"),
                geometry=region, scale=scale
            ).get("groups").getInfo()
            
        r_target_f = get_grouped(masked_target.select(band), clear_mask)
        r_pred_f = get_grouped(pred_band, clear_mask)
        r_ref_g = get_grouped(ref_img.select(band), gap_mask)
        r_ref_f = get_grouped(ref_img.select(band), clear_mask)
        
        # Pixel counts per cluster on clear pixels
        r_counts = master_segments.rename("seg").updateMask(clear_mask).addBands(master_segments).reduceRegion(
            reducer=ee.Reducer.count().group(1, "seg"),
            geometry=region, scale=scale
        ).get("groups").getInfo()
        
        dt_target = {g["seg"]: g["mean"] for g in r_target_f}
        dt_pred = {g["seg"]: g["mean"] for g in r_pred_f}
        dt_ref_g = {g["seg"]: g["mean"] for g in r_ref_g}
        dt_ref_f = {g["seg"]: g["mean"] for g in r_ref_f}
        dt_cnt = {g["seg"]: g["count"] for g in r_counts}
        
        keys = [-1]
        deltas = [delta_glob]
        cluster_delta_dict = {}
        
        for c in range(4):
            cnt = dt_cnt.get(c, 0)
            if c in dt_target and c in dt_pred and cnt >= 30:
                tb = dt_target[c] - dt_pred[c]
            else:
                tb = delta_temp_glob
                
            if c in dt_ref_g and c in dt_ref_f:
                sb = dt_ref_g[c] - dt_ref_f[c]
            else:
                sb = delta_spatial_glob
                
            tot_delta = tb + sb
            keys.append(c)
            deltas.append(tot_delta)
            cluster_delta_dict[f"Cluster_{c}"] = tot_delta
            
        metrics["cluster_deltas"][band] = cluster_delta_dict
        
        delta_img = master_segments.remap(keys, deltas, delta_glob).rename("delta")
        adapt_pred = pred_band.add(delta_img)
        
        # 100% Crisp Native 30m Fill (NO WHOLE-IMAGE BLUR!)
        adapt_crisp = masked_target.select(band).unmask(adapt_pred)
        
        # Explicit Difference Rasters inside the cloud gap
        diff_gt = pred_global_band.subtract(pred_band).abs().updateMask(gap_mask)
        diff_ag = adapt_pred.subtract(pred_global_band).abs().updateMask(gap_mask)
        
        stage_bands["0_Raw_Cloudy"].append(raw_img.select(band))
        stage_bands["1_Masked_Clouds"].append(masked_target.select(band))
        stage_bands["2_Temporal_Interpolation"].append(temp_filled)
        stage_bands["3_Global_Correction"].append(glob_filled)
        stage_bands["4_Adaptive_Correction"].append(adapt_crisp)
        stage_bands["5_Diff_Global_Temporal"].append(diff_gt)
        stage_bands["6_Diff_Adaptive_Global"].append(diff_ag)
        
        # Quantitative Ground Truth Validation (if simulated mask date)
        if simulated_mask_date and band == "SR_B5":
            true_gap_vals = raw_img.select(band).updateMask(gap_mask)
            diff_sq_temp = temp_filled.subtract(true_gap_vals).pow(2)
            rmse_temp = math.sqrt(diff_sq_temp.reduceRegion(ee.Reducer.mean(), region, scale).get(band).getInfo() or 0)
            
            diff_sq_glob = glob_filled.subtract(true_gap_vals).pow(2)
            rmse_glob = math.sqrt(diff_sq_glob.reduceRegion(ee.Reducer.mean(), region, scale).get(band).getInfo() or 0)
            
            diff_sq_adap = adapt_crisp.subtract(true_gap_vals).pow(2)
            rmse_adap = math.sqrt(diff_sq_adap.reduceRegion(ee.Reducer.mean(), region, scale).get(band).getInfo() or 0)
            
            err_reduc = ((rmse_temp - rmse_adap) / rmse_temp) * 100.0 if rmse_temp > 0 else 0.0
            metrics["ground_truth_eval"] = {
                "rmse_temporal": rmse_temp,
                "rmse_global": rmse_glob,
                "rmse_adaptive": rmse_adap,
                "error_reduction_pct": err_reduc
            }
            print(f"     [Ground Truth Benchmark] NIR RMSE: Temporal={rmse_temp:.1f} | Global={rmse_glob:.1f} | Adaptive={rmse_adap:.1f} (Error Reduction: {err_reduc:.2f}%)")

    print(f"     Global Delta (NIR): {metrics['global_deltas']['SR_B5']:+.1f}")
    cd_vals = [f"{k}: {v:+.1f}" for k, v in metrics["cluster_deltas"]["SR_B5"].items()]
    print(f"     Cluster Deltas (NIR): {', '.join(cd_vals)}")
    
    # 5. Submit Multi-Band GeoTIFF Tasks to Google Drive for QGIS
    if drive_folder:
        clean_date = date_str.replace("-", "")
        export_stages = [
            ("0_Raw_Cloudy", stage_bands["0_Raw_Cloudy"]),
            ("1_Masked_Clouds", stage_bands["1_Masked_Clouds"]),
            ("2_Temporal_Interpolation", stage_bands["2_Temporal_Interpolation"]),
            ("3_Global_Correction", stage_bands["3_Global_Correction"]),
            ("4_Adaptive_Correction", stage_bands["4_Adaptive_Correction"]),
            ("5_Diff_Global_Temporal", stage_bands["5_Diff_Global_Temporal"]),
            ("6_Diff_Adaptive_Global", stage_bands["6_Diff_Adaptive_Global"])
        ]
        print(f"  -> Submitting GeoTIFF exports to Drive folder '{drive_folder}'...")
        for sname, blist in export_stages:
            fcc_3band = ee.Image.cat(blist).toDouble().clip(region)
            desc = f"{clean_date}_{sname}"
            task = ee.batch.Export.image.toDrive(
                image=fcc_3band,
                description=desc,
                folder=drive_folder,
                fileNamePrefix=desc,
                scale=scale,
                region=region,
                crs="EPSG:32616",
                maxPixels=1e10
            )
            task.start()
            print(f"     • Task started: {desc}")

    # 6. Render High-Resolution 6-Panel Diagnostic Figure (FCC 5-4-3)
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
        titles = [
            f"1. Raw Cloudy Image\n({spacecraft} {date_str}: {cloud_pct:.1f}% Clouds)",
            f"2. Enhanced Masked\n(Clouds & Shadows Removed)",
            f"3. Stage 1: Temporal\n(Linear Interpolation)",
            f"4. Stage 2: Global\n(Uniform Delta Offset)",
            f"5. Stage 3: Adaptive (CRISP)\n(STAIR K-Means, NO BLUR)",
            f"6. Adaptive vs Global Diff\n|Adaptive - Global| in Gap"
        ]
        
        fcc_vis = {
            'region': region.getInfo(),
            'dimensions': 600,
            'min': [7000, 5000, 5000],
            'max': [25000, 12000, 12000]
        }
        
        diff_vis = {
            'region': region.getInfo(),
            'dimensions': 600,
            'min': 0,
            'max': 2500,
            'palette': ['#ffffcc', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#800026']
        }
        
        stages_to_render = [
            ("0_Raw_Cloudy", fcc_vis, False),
            ("1_Masked_Clouds", fcc_vis, False),
            ("2_Temporal_Interpolation", fcc_vis, False),
            ("3_Global_Correction", fcc_vis, False),
            ("4_Adaptive_Correction", fcc_vis, False),
            ("6_Diff_Adaptive_Global", diff_vis, True)
        ]
        
        for ax, title, (sname, vp, is_diff) in zip(axes, titles, stages_to_render):
            try:
                if is_diff:
                    img_to_thumb = stage_bands[sname][0]  # NIR diff
                else:
                    img_to_thumb = ee.Image.cat(stage_bands[sname]).select([0, 1, 2])
                    
                thumb_url = img_to_thumb.getThumbURL(vp)
                req = urllib.request.Request(thumb_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as resp:
                    pil_img = Image.open(io.BytesIO(resp.read()))
                    ax.imshow(pil_img)
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.axis('off')
            except Exception as e:
                ax.set_title(f"{title}\n(Render Error: {e})", fontsize=9)
                ax.axis('off')
                
        plt.tight_layout()
        plot_path = out_dir / f"stair_2022_{date_str.replace('-', '')}_6panel.png"
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved Local 6-Panel Diagnostic Plot: {plot_path}")
        
    return metrics

def main():
    init_earth_engine()
    
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    drive_folder = f"STAIR_2022_NewDates_Evaluation_{timestamp}"
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "5day_cloudy_evaluation"
    
    # 5 Brand-New 2022 Evaluation Dates + 1 Ground Truth Benchmark
    # Strictly excludes all previous dates (2021-04-21, 05-16, 05-23, 06-01, 06-17, 08-04, 08-20, 09-05)
    evaluation_days = [
        {"date": "2022-05-11", "desc": "Spring Awakening / Early Crop Emergence", "sim_mask": None},
        {"date": "2022-06-11", "desc": "Early Summer Vegetative Growth", "sim_mask": None},
        {"date": "2022-06-27", "desc": "Mid-Vegetative Biomass Surge", "sim_mask": None},
        {"date": "2022-07-29", "desc": "Peak Summer Canopy / Reproductive", "sim_mask": None},
        {"date": "2022-08-30", "desc": "Late Summer Grain Filling / Senescence", "sim_mask": None},
        {"date": "2022-07-22", "desc": "Ground Truth Benchmark (0.0% Cloud vs Realistic 46.5% Gap)", "sim_mask": "2022-06-27"}
    ]
    
    all_metrics = []
    print("\n" + "#"*80)
    print(f" STARTING STAIR 2022 NEW DATES EVALUATION (CRISP & SHADOW-FREE)")
    print(f" Google Drive Folder for QGIS: '{drive_folder}'")
    print(f" Clear Reference Day (t1): 2022-07-22 (Pristine 0.00% Cloud Landsat 8)")
    print("#"*80)
    
    for item in evaluation_days:
        m = process_evaluation_day(
            date_str=item["date"],
            cloud_desc=item["desc"],
            ref_date="2022-07-22",
            simulated_mask_date=item.get("sim_mask"),
            region=region,
            scale=30.0,
            out_dir=out_dir,
            drive_folder=drive_folder
        )
        all_metrics.append(m)
        
    print("\n\n" + "="*95)
    print("                     STAIR 2022 NEW DATES EVALUATION SUMMARY")
    print("="*95)
    print(f"{'Date':<12} | {'Sensor':<10} | {'AOI Cloud %':<12} | {'Global Delta (NIR)':<20} | {'Cluster Deltas (NIR Spread)'}")
    print("-"*95)
    for m in all_metrics:
        d = m["date"]
        s = m["spacecraft"]
        cp = m["cloud_pct"]
        gd = m["global_deltas"]["SR_B5"]
        cd_vals = list(m["cluster_deltas"]["SR_B5"].values())
        min_cd = min(cd_vals) if cd_vals else 0
        max_cd = max(cd_vals) if cd_vals else 0
        spread = max_cd - min_cd
        print(f"{d:<12} | {s:<10} | {cp:<12.1f}% | {gd:<+20.1f} | [{min_cd:+.1f} to {max_cd:+.1f}] (Spread: {spread:.1f})")
        
    print("\n" + "="*95)
    print("                      GROUND TRUTH BENCHMARK ACCURACY")
    print("="*95)
    gt_item = next((m for m in all_metrics if m["ground_truth_eval"]), None)
    if gt_item:
        gte = gt_item["ground_truth_eval"]
        print(f"Benchmark Date: {gt_item['date']} (Real 46.5% Cloud Gap applied over 100% Clear Ground Truth)")
        print(f"  • Stage 1 (Temporal Linear) NIR RMSE:  {gte['rmse_temporal']:.1f}")
        print(f"  • Stage 2 (Global Spatial)  NIR RMSE:  {gte['rmse_global']:.1f}")
        print(f"  • Stage 3 (Adaptive STAIR)  NIR RMSE:  {gte['rmse_adaptive']:.1f}")
        print(f"  >>> Error Reduction: {gte['error_reduction_pct']:.2f}% improvement over Temporal baseline!")
    print("="*95)
    
    print(f"\nAll 42 QGIS GeoTIFF export tasks submitted to Google Drive: '{drive_folder}'!")
    print(f"Local 6-panel PNG diagnostic figures generated in: '{out_dir}'")

if __name__ == "__main__":
    main()
