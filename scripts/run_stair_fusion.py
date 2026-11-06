#!/usr/bin/env python3
"""
STAIR Spatiotemporal Fusion Pipeline & Ground Truth Benchmark
=============================================================
Implements the rigorous Landsat-MODIS fusion algorithm from Luo et al. (2018):
  1. Input 1: Reference Landsat & MODIS pair on T1 (2022-07-14)
  2. Input 2: Reference Landsat & MODIS pair on T2 (2022-07-29)
              (Landsat on T2 is 49.6% cloudy and reconstructed via STAIR Adaptive Imputation)
  3. Input 3: Daily Target MODIS on tp (2022-07-22)
  4. Difference Calculation:
       epsilon(T1) = L(T1) - M(T1)
       epsilon(T2) = L_imputed(T2) - M(T2)
  5. Difference Interpolation to target date tp:
       Delta(tp) = w1 * epsilon(T1) + w2 * epsilon(T2)
  6. High-Resolution Synthesis:
       I(tp) = M(tp) + Delta(tp)
  7. Validation against True Ground Truth Landsat 8 image on 2022-07-22 (0.0% clouds).
"""

import sys
import os
import time
import io
import urllib.request
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
from stair.core import init_earth_engine
from stair.diagnostics import create_chip_bounds

def fetch_multiband_geotiff(ee_img, region, crs="EPSG:32616", scale=30.0, max_retries=4):
    url = ee_img.getDownloadURL({
        'region': region,
        'scale': scale,
        'crs': crs,
        'format': 'GEO_TIFF'
    })
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=35) as resp:
                with rasterio.open(io.BytesIO(resp.read())) as src:
                    return src.read(), src.profile
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)

def run_fusion_evaluation():
    init_earth_engine()
    
    # 7km x 7km Clayton Township, MI AOI
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    
    out_dir = Path("/Users/bharathmikkilineni/Stair_Project/outputs/STAIR_Fusion_Evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    qgis_dir = out_dir / "QGIS_Layers"
    qgis_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STAIR SPATIOTEMPORAL FUSION PIPELINE: LANDSAT 8/9 & MODIS MCD43A4")
    print("Target Prediction Date: 2022-07-22 (Ground Truth Benchmark)")
    print("Reference Prior Date:   2022-07-14 (Landsat 9 + MODIS)")
    print("Reference Post Date:    2022-07-29 (Landsat 8 STAIR-Imputed + MODIS)")
    print("=" * 70)
    
    l8_bands = ['SR_B5', 'SR_B4', 'SR_B3'] # NIR, Red, Green
    mcd_bands = ['Nadir_Reflectance_Band2', 'Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band4']
    
    # 1. Fetch Target Ground Truth Landsat (2022-07-22)
    print("\n[1/5] Fetching True Ground Truth Landsat (2022-07-22)...")
    l8_target = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(region).filterDate('2022-07-22', '2022-07-23').first()
    l8_proj = l8_target.select('SR_B5').projection()
    gt_ee = l8_target.select(l8_bands).multiply(0.0000275).subtract(0.2)
    gt_arr, prof = fetch_multiband_geotiff(gt_ee, region)
    
    # 2. Fetch Target MODIS (2022-07-22)
    print("[2/5] Fetching Target Daily MODIS MCD43A4 (2022-07-22)...")
    mcd_target = ee.ImageCollection('MODIS/061/MCD43A4').filterBounds(region).filterDate('2022-07-22', '2022-07-23').first()
    mcd_target_ee = mcd_target.select(mcd_bands).multiply(0.0001).resample('bilinear').reproject(crs=l8_proj, scale=30.0)
    target_modis, _ = fetch_multiband_geotiff(mcd_target_ee, region)
    
    # 3. Fetch Prior Reference T1: 2022-07-14 (Landsat 9 + MODIS)
    print("[3/5] Fetching Reference Pair T1 (2022-07-14)...")
    l9_ref1 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(region).filterDate('2022-07-14', '2022-07-15').first()
    l9_ref1_ee = l9_ref1.select(l8_bands).multiply(0.0000275).subtract(0.2)
    ref1_landsat, _ = fetch_multiband_geotiff(l9_ref1_ee, region)
    
    mcd_ref1 = ee.ImageCollection('MODIS/061/MCD43A4').filterBounds(region).filterDate('2022-07-14', '2022-07-15').first()
    mcd_ref1_ee = mcd_ref1.select(mcd_bands).multiply(0.0001).resample('bilinear').reproject(crs=l8_proj, scale=30.0)
    ref1_modis, _ = fetch_multiband_geotiff(mcd_ref1_ee, region)
    
    # 4. Load Post Reference T2: 2022-07-29 (STAIR Adaptive Imputed Landsat 8 + MODIS)
    print("[4/5] Loading Reference Pair T2 (2022-07-29) with STAIR Adaptive Imputation...")
    p_ref2 = '/Users/bharathmikkilineni/Downloads/STAIR_2022_NewDates_Evaluation_20261018_234144/20220729_4_Adaptive_Correction.tif'
    with rasterio.open(p_ref2) as src:
        # Band 1: NIR, Band 2: Red, Band 3: Green
        ref2_landsat = src.read() * 0.0000275 - 0.2
        
    mcd_ref2 = ee.ImageCollection('MODIS/061/MCD43A4').filterBounds(region).filterDate('2022-07-29', '2022-07-30').first()
    mcd_ref2_ee = mcd_ref2.select(mcd_bands).multiply(0.0001).resample('bilinear').reproject(crs=l8_proj, scale=30.0)
    ref2_modis, _ = fetch_multiband_geotiff(mcd_ref2_ee, region)
    
    # 5. Execute STAIR Spatiotemporal Fusion Math (Section 2.3)
    print("\n[5/5] Executing STAIR Spatiotemporal Fusion Synthesis...")
    # Time weights: target is July 22 (DOY 203) between July 14 (DOY 195) and July 29 (DOY 210)
    t1_doy = 195
    tp_doy = 203
    t2_doy = 210
    w1 = (t2_doy - tp_doy) / (t2_doy - t1_doy) # 7 / 15
    w2 = (tp_doy - t1_doy) / (t2_doy - t1_doy) # 8 / 15
    print(f"     Time Interpolation Weights: w1(Jul 14) = {w1:.4f}, w2(Jul 29) = {w2:.4f}")
    
    # Spatial differences on matching pairs
    eps_t1 = ref1_landsat - ref1_modis
    eps_t2 = ref2_landsat - ref2_modis
    
    # Interpolated difference field
    delta_interp = w1 * eps_t1 + w2 * eps_t2
    
    # Synthesized 30m prediction
    fused_prediction = np.clip(target_modis + delta_interp, 0.0, 1.0)
    
    # Single-reference baseline (using T1 only)
    fused_single = np.clip(target_modis + eps_t1, 0.0, 1.0)
    
    # Calculate NDVI
    ndvi_gt = (gt_arr[0] - gt_arr[1]) / np.maximum(gt_arr[0] + gt_arr[1], 1e-4)
    ndvi_fused = (fused_prediction[0] - fused_prediction[1]) / np.maximum(fused_prediction[0] + fused_prediction[1], 1e-4)
    ndvi_ref1 = (ref1_landsat[0] - ref1_landsat[1]) / np.maximum(ref1_landsat[0] + ref1_landsat[1], 1e-4)
    ndvi_modis = (target_modis[0] - target_modis[1]) / np.maximum(target_modis[0] + target_modis[1], 1e-4)
    
    # Quantitative Validation
    band_names = ['NIR (Band 5)', 'Red (Band 4)', 'Green (Band 3)']
    metrics = {}
    
    print("\n" + "=" * 70)
    print("QUANTITATIVE BENCHMARK: STAIR FUSION VS GROUND TRUTH (2022-07-22)")
    print("=" * 70)
    
    for b_idx, b_name in enumerate(band_names):
        gt_b = gt_arr[b_idx]
        f_b = fused_prediction[b_idx]
        r1_b = ref1_landsat[b_idx]
        r2_b = ref2_landsat[b_idx]
        m_b = target_modis[b_idx]
        
        valid = ~np.isnan(gt_b) & ~np.isnan(f_b) & (gt_b > 0)
        
        rmse_f = np.sqrt(np.mean((f_b[valid] - gt_b[valid])**2))
        mae_f = np.mean(np.abs(f_b[valid] - gt_b[valid]))
        corr_f = np.corrcoef(f_b[valid], gt_b[valid])[0, 1]
        r2_f = corr_f ** 2
        
        rmse_base = np.sqrt(np.mean((r1_b[valid] - gt_b[valid])**2))
        corr_base = np.corrcoef(r1_b[valid], gt_b[valid])[0, 1]
        
        rmse_mod = np.sqrt(np.mean((m_b[valid] - gt_b[valid])**2))
        corr_mod = np.corrcoef(m_b[valid], gt_b[valid])[0, 1]
        
        err_red = (rmse_base - rmse_f) / rmse_base * 100.0
        
        metrics[b_name] = {
            'rmse': rmse_f, 'mae': mae_f, 'corr': corr_f, 'r2': r2_f,
            'rmse_base': rmse_base, 'corr_base': corr_base,
            'rmse_mod': rmse_mod, 'corr_mod': corr_mod,
            'error_reduction': err_red
        }
        
        print(f"\n{b_name}:")
        print(f"  • Raw MODIS 500m (No Fusion):     RMSE = {rmse_mod:.4f},  Corr = {corr_mod:.4f}")
        print(f"  • Reference Landsat 1 (Jul 14):   RMSE = {rmse_base:.4f},  Corr = {corr_base:.4f}")
        print(f"  • STAIR Fused 30m (Ours):         RMSE = {rmse_f:.4f},  Corr = {corr_f:.4f},  R² = {r2_f:.4f}")
        print(f"  >>> TOTAL ERROR REDUCTION: {err_red:.1f}%")
        
    # NDVI Metrics
    val_ndvi = ~np.isnan(ndvi_gt) & ~np.isnan(ndvi_fused) & (gt_arr[0] > 0)
    rmse_ndvi = np.sqrt(np.mean((ndvi_fused[val_ndvi] - ndvi_gt[val_ndvi])**2))
    corr_ndvi = np.corrcoef(ndvi_fused[val_ndvi], ndvi_gt[val_ndvi])[0, 1]
    rmse_base_ndvi = np.sqrt(np.mean((ndvi_ref1[val_ndvi] - ndvi_gt[val_ndvi])**2))
    corr_base_ndvi = np.corrcoef(ndvi_ref1[val_ndvi], ndvi_gt[val_ndvi])[0, 1]
    ndvi_err_red = (rmse_base_ndvi - rmse_ndvi) / rmse_base_ndvi * 100.0
    
    print(f"\nNDVI (Vegetation Phenology):")
    print(f"  • Reference Landsat 1 (Jul 14):   RMSE = {rmse_base_ndvi:.4f},  Corr = {corr_base_ndvi:.4f}")
    print(f"  • STAIR Fused 30m NDVI (Ours):    RMSE = {rmse_ndvi:.4f},  Corr = {corr_ndvi:.4f}")
    print(f"  >>> NDVI ERROR REDUCTION: {ndvi_err_red:.1f}%")
    
    # -------------------------------------------------------------
    # 6. Save GeoTIFF Rasters for QGIS
    # -------------------------------------------------------------
    print("\nSaving Multi-Band GeoTIFFs to outputs directory...")
    p = prof.copy()
    p.update({'count': 3, 'dtype': 'float32', 'nodata': None})
    
    with rasterio.open(out_dir / "1_Prior_Ref_Landsat_20220714.tif", "w", **p) as dst:
        for b in range(3): dst.write(ref1_landsat[b].astype(np.float32), b + 1)
        
    with rasterio.open(out_dir / "2_Post_Ref_Landsat_Imputed_20220729.tif", "w", **p) as dst:
        for b in range(3): dst.write(ref2_landsat[b].astype(np.float32), b + 1)
        
    with rasterio.open(out_dir / "3_Target_MODIS_Resampled_20220722.tif", "w", **p) as dst:
        for b in range(3): dst.write(target_modis[b].astype(np.float32), b + 1)
        
    with rasterio.open(out_dir / "4_STAIR_Fused_Prediction_20220722.tif", "w", **p) as dst:
        for b in range(3): dst.write(fused_prediction[b].astype(np.float32), b + 1)
        
    with rasterio.open(out_dir / "5_True_GroundTruth_Landsat_20220722.tif", "w", **p) as dst:
        for b in range(3): dst.write(gt_arr[b].astype(np.float32), b + 1)
        
    p_single = prof.copy()
    p_single.update({'count': 1, 'dtype': 'float32', 'nodata': None})
    diff_nir = np.abs(fused_prediction[0] - gt_arr[0]).astype(np.float32)
    with rasterio.open(out_dir / "6_Error_Heatmap_NIR_20220722.tif", "w", **p_single) as dst:
        dst.write(diff_nir, 1)
        
    with rasterio.open(out_dir / "7_NDVI_STAIR_Fused_20220722.tif", "w", **p_single) as dst:
        dst.write(ndvi_fused.astype(np.float32), 1)
        
    with rasterio.open(out_dir / "8_NDVI_GroundTruth_20220722.tif", "w", **p_single) as dst:
        dst.write(ndvi_gt.astype(np.float32), 1)
        
    # Pre-Colored RGB GeoTIFFs for instant QGIS visual toggle
    def to_fcc_rgb(arr):
        # Bands: 0=NIR, 1=Red, 2=Green. FCC maps R=NIR, G=Red, B=Green
        r = np.clip((arr[0] - 0.15) / (0.60 - 0.15), 0, 1)
        g = np.clip((arr[1] - 0.02) / (0.20 - 0.02), 0, 1)
        b = np.clip((arr[2] - 0.03) / (0.20 - 0.03), 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255).astype(np.uint8)
        
    def save_rgb_tif(rgb_arr, path):
        p_rgb = prof.copy()
        p_rgb.update({'count': 3, 'dtype': 'uint8', 'nodata': None})
        with rasterio.open(path, "w", **p_rgb) as dst:
            for b in range(3): dst.write(rgb_arr[:, :, b], b + 1)
            
    save_rgb_tif(to_fcc_rgb(ref1_landsat), qgis_dir / "1_Prior_Ref_Landsat_20220714_FCC.tif")
    save_rgb_tif(to_fcc_rgb(ref2_landsat), qgis_dir / "2_Post_Ref_Landsat_Imputed_20220729_FCC.tif")
    save_rgb_tif(to_fcc_rgb(target_modis), qgis_dir / "3_Target_MODIS_Resampled_20220722_FCC.tif")
    save_rgb_tif(to_fcc_rgb(fused_prediction), qgis_dir / "4_STAIR_Fused_Prediction_20220722_FCC.tif")
    save_rgb_tif(to_fcc_rgb(gt_arr), qgis_dir / "5_True_GroundTruth_Landsat_20220722_FCC.tif")
    print(f"Pre-colored QGIS layers saved to: {qgis_dir}")
    
    # -------------------------------------------------------------
    # 7. Render Comprehensive 6-Panel Diagnostic Figure
    # -------------------------------------------------------------
    print("\nGenerating Diagnostic Figures...")
    fig, axes = plt.subplots(1, 6, figsize=(32, 5.5))
    
    panels = [
        ("(a) Prior Ref Landsat (T1)\n2022-07-14 (30m)", to_fcc_rgb(ref1_landsat)),
        ("(b) Post Ref Landsat (T2)\n2022-07-29 (STAIR Imputed)", to_fcc_rgb(ref2_landsat)),
        ("(c) Target Daily MODIS (tp)\n2022-07-22 (Coarse 500m)", to_fcc_rgb(target_modis)),
        ("(d) STAIR Fused Prediction (tp)\n2022-07-22 (30m Synthetic)", to_fcc_rgb(fused_prediction)),
        ("(e) True Ground Truth (tp)\n2022-07-22 (Real Landsat 8)", to_fcc_rgb(gt_arr)),
    ]
    
    for ax, (title, img) in zip(axes[:5], panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis('off')
        
    im_err = axes[5].imshow(diff_nir, cmap='inferno', vmin=0.0, vmax=0.15)
    axes[5].set_title(f"(f) Absolute Error |Fused - True|\nRMSE: {metrics['NIR (Band 5)']['rmse']:.4f} (NIR)", fontsize=11, fontweight="bold", color="darkred")
    axes[5].axis('off')
    plt.colorbar(im_err, ax=axes[5], fraction=0.046, pad=0.04)
    
    plt.suptitle("STAIR Spatiotemporal Data Fusion (Landsat 8/9 & MODIS MCD43A4) - 2022-07-22 Ground Truth Benchmark", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    fig6_path = out_dir / "STAIR_Fusion_Benchmark_6panel.png"
    plt.savefig(fig6_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved 6-Panel Benchmark Figure: {fig6_path}")
    
    # -------------------------------------------------------------
    # 8. Render Zoomed Farmland Parcel Comparison
    # -------------------------------------------------------------
    cy, cx, cs = 40, 60, 70 # Crop parcel focus
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    
    crop_panels = [
        ("(a) Daily MODIS 500m\n(Blurred Crop Parcels)", to_fcc_rgb(target_modis)[cy:cy+cs, cx:cx+cs]),
        ("(b) Prior Landsat (T1)\n(Outdated Phenology)", to_fcc_rgb(ref1_landsat)[cy:cy+cs, cx:cx+cs]),
        ("(c) STAIR Fused 30m\n(Sharp Synthesized Parcels)", to_fcc_rgb(fused_prediction)[cy:cy+cs, cx:cx+cs]),
        ("(d) True Ground Truth 30m\n(Real Landsat 8 Benchmark)", to_fcc_rgb(gt_arr)[cy:cy+cs, cx:cx+cs]),
    ]
    
    for ax, (title, img) in zip(axes, crop_panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis('off')
        # Add highlight box around single field
        rect = patches.Rectangle((15, 15), 35, 35, linewidth=2.5, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)
        
    plt.suptitle("STAIR Spatiotemporal Fusion on Farmland Parcels: Preserving 30m Boundaries with Daily MODIS Phenology", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    zoom_path = out_dir / "STAIR_Fusion_Farmland_Zoom_4panel.png"
    plt.savefig(zoom_path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved Farmland Zoom Comparison: {zoom_path}")
    
    # -------------------------------------------------------------
    # 9. Render Scatter Plot Validation (1:1 Line & R²)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    scatter_data = [
        ("NIR (Band 5)", gt_arr[0], fused_prediction[0], 0.0, 0.7),
        ("Red (Band 4)", gt_arr[1], fused_prediction[1], 0.0, 0.25),
        ("Green (Band 3)", gt_arr[2], fused_prediction[2], 0.0, 0.25),
        ("NDVI", ndvi_gt, ndvi_fused, -0.2, 0.95),
    ]
    
    for ax, (name, actual, pred, min_v, max_v) in zip(axes, scatter_data):
        mask = ~np.isnan(actual) & ~np.isnan(pred) & (gt_arr[0] > 0)
        a_sub = actual[mask][::5] # Subsample for fast plotting
        p_sub = pred[mask][::5]
        
        r = np.corrcoef(a_sub, p_sub)[0, 1]
        rmse = np.sqrt(np.mean((a_sub - p_sub)**2))
        
        ax.scatter(a_sub, p_sub, alpha=0.15, s=6, color='navy')
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label="1:1 Line")
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        ax.set_xlabel("True Landsat Ground Truth", fontsize=10, fontweight="bold")
        ax.set_ylabel("STAIR Fused Prediction", fontsize=10, fontweight="bold")
        ax.set_title(f"{name}\nRMSE: {rmse:.4f} | R²: {r**2:.3f} | R: {r:.3f}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")
        
    plt.suptitle("STAIR Fusion Quantitative Scatter Plots vs Ground Truth (Clayton Township, MI - 2022-07-22)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    scatter_path = out_dir / "STAIR_Fusion_Scatter_Validation.png"
    plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Scatter Validation Plot: {scatter_path}")
    
    # Figures already saved to outputs/STAIR_Fusion_Evaluation/
    return metrics

if __name__ == "__main__":
    run_fusion_evaluation()
