import ee
import time
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stair.core import init_earth_engine, get_landsat_collection
from stair.diagnostics import create_chip_bounds
from stair.temporal import impute_temporal
from stair.adaptive import stair_adaptive_pipeline
import config

def main():
    init_earth_engine()
    
    print("Setting AOI to Clayton, MI (7km chip)...")
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    scale = 30.0
    
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]
    
    # 1. Select the specific cloudy dates for the Paper Figure
    paper_dates = [
        "2021-05-23",
        "2021-08-04"
    ]
    
    year = 2021
    export_folder = "STAIR_Paper_Figures_2021"
    
    # 2. Get the clear reference date for K-Means mapping
    summer_col = ee.ImageCollection(config.LANDSAT8_COLLECTION)\
        .filterBounds(region)\
        .filterDate(f"{year}-06-01", f"{year}-08-31")\
        .sort("CLOUD_COVER")
    
    best_img = summer_col.first()
    reference_date = ee.Date(best_img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    
    print(f"Automatically selected clear reference date for {year}: {reference_date}")
    ref_col = get_landsat_collection(region, reference_date, ee.Date(reference_date).advance(1, "day").format("YYYY-MM-dd").getInfo(), apply_mask=True)
    image_t1 = ref_col.first()
    
    print("\nPrecomputing multi-band K-Means clusters for Adaptive Correction...")
    ref_stack = image_t1.select(bands_to_process)
    training_data = ref_stack.sample(region=region, scale=scale, numPixels=5000, tileScale=1)
    clusterer = ee.Clusterer.wekaKMeans(4).train(training_data)
    master_segments = ref_stack.cluster(clusterer).rename("seg")
    
    # 3. Pull the entire time series (March 1 to October 31) for the target year
    target_start = f"{year}-03-01"
    target_end = f"{year}-10-31"
    
    # Pad the training collection by 1 year on both sides to guarantee temporal context for linear imputation
    context_start = f"{year-1}-01-01"
    context_end = f"{year+1}-12-31"
    
    full_col = ee.ImageCollection(config.LANDSAT8_COLLECTION)\
        .filterBounds(region)\
        .filterDate(target_start, target_end)
        
    training_col = get_landsat_collection(region, context_start, context_end, max_cloud_cover=100.0, apply_mask=True)
    
    tasks = []
    
    # We want to export these specific intermediate stages to see the progression
    stages_to_export = [
        "stage1_broken",     # Masked image (clouds = holes)
        "stage2_temporal",   # Base Temporal Linear Prediction
        "stage3a_global",    # Global Spatial offset applied
        "stage3b_adaptive"   # Adaptive K-Means offset applied (Final)
    ]
    
    print(f"\nSubmitting Paper Figure generation tasks to folder: '{export_folder}'...")
    
    for target_date in paper_dates:
        print(f"\n  -> Processing figure data for: {target_date}")
        
        # Get the cloudy target image for this day
        target_img_masked = training_col.filterDate(target_date, ee.Date(target_date).advance(1, "day")).first()
        
        # Prepare a dictionary to hold the lists of bands for each stage
        stage_bands = {stage: [] for stage in stages_to_export}
        
        for band in bands_to_process:
            # Temporal Prediction
            pred_linear, _ = impute_temporal(training_col, target_img_masked, target_date, band_name=band)
            
            # STAIR Pipeline
            results = stair_adaptive_pipeline(
                raw_image_t0=target_img_masked,
                image_t1=image_t1,
                linear_prediction=pred_linear,
                region=region,
                n_clusters=4,
                band_name=band,
                scale=scale,
                tile_scale=1,
                precomputed_segments=master_segments
            )
            
            # Save the band results for each stage
            for stage in stages_to_export:
                stage_bands[stage].append(results[stage])
                
        # 1. Export the absolute RAW image (with clouds)
        raw_img = full_col.filterDate(target_date, ee.Date(target_date).advance(1, "day")).first()
        raw_fcc = raw_img.select(bands_to_process).clip(region)
        
        raw_desc = f"Figure_{target_date.replace('-', '')}_0_Raw"
        raw_task = ee.batch.Export.image.toDrive(
            image=raw_fcc, description=raw_desc, folder=export_folder, fileNamePrefix=raw_desc, scale=scale, region=region, maxPixels=1e10
        )
        raw_task.start()
        tasks.append(raw_task)
        
        # 2. Export all intermediate STAIR stages
        for idx, stage in enumerate(stages_to_export, start=1):
            stage_fcc = ee.Image.cat(stage_bands[stage]).clip(region)
            
            # Naming convention: Figure_Date_1_stage1_broken, Figure_Date_2_stage2_temporal, etc.
            stage_desc = f"Figure_{target_date.replace('-', '')}_{idx}_{stage}"
            
            stage_task = ee.batch.Export.image.toDrive(
                image=stage_fcc, description=stage_desc, folder=export_folder, fileNamePrefix=stage_desc, scale=scale, region=region, maxPixels=1e10
            )
            stage_task.start()
            tasks.append(stage_task)
            
        time.sleep(1)

    print(f"\nSubmitted {len(tasks)} tasks for the Paper Figures! Track them at https://code.earthengine.google.com/tasks")

if __name__ == "__main__":
    main()
