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
    
    # 1. Define Clayton, MI AOI (7km x 7km)
    print("Setting AOI to Clayton, MI (7km chip)...")
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    scale = 30.0
    
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]
    
    years_to_process = [2021, 2022, 2023]
    tasks = []
    
    for year in years_to_process:
        print(f"\n=========================================")
        print(f"       PROCESSING YEAR {year}            ")
        print(f"=========================================")
        
        # 2. Get the clear reference date for K-Means mapping
        # Automatically find the clearest image in the summer months for the reference
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
        
        # 3. Pull the entire time series (March 1 to October 31)
        start_date = f"{year}-03-01"
        end_date = f"{year}-10-31"
        
        # Get all Landsat 8 images in this window
        print(f"\nPulling historical time series for interpolation from {start_date} to {end_date}...")
        full_col = ee.ImageCollection(config.LANDSAT8_COLLECTION)\
            .filterBounds(region)\
            .filterDate(start_date, end_date)
            
        # Get the list of actual dates where the satellite passed over
        date_list = full_col.aggregate_array('system:time_start').map(lambda d: ee.Date(d).format('YYYY-MM-dd')).distinct().getInfo()
        
        print(f"Found {len(date_list)} satellite passes over Clayton in this 7-month period for {year}.")
        
        # Get the masked training collection (clouds removed, leaving gaps)
        training_col = get_landsat_collection(region, start_date, end_date, max_cloud_cover=100.0, apply_mask=True)
        
        # 4. Loop through every date and generate the STAIR output
        print(f"\nSubmitting STAIR interpolation tasks to Google Earth Engine for {year}...")
        
        export_folder = f"STAIR_Clayton_{year}"
        print(f"All images for {year} will be exported to Google Drive folder: '{export_folder}'")
        
        for target_date in date_list:
            print(f"  -> Preparing task for: {target_date}")
            
            # Get the cloudy target image for this day
            target_img_masked = training_col.filterDate(target_date, ee.Date(target_date).advance(1, "day")).first()
            
            final_bands = []
            
            for band in bands_to_process:
                # 1. Temporal Prediction (fills gaps using before/after images)
                pred_linear, _ = impute_temporal(training_col, target_img_masked, target_date, band_name=band)
                
                # 2. Adaptive Correction (fixes colors inside gaps)
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
                
                final_bands.append(results["stage3b_adaptive"])
                
            # Combine bands into a single False Color Composite (NIR, Red, Green)
            stair_fcc = ee.Image.cat(final_bands).clip(region)
            
            # 1. Export STAIR Image
            stair_desc = f"STAIR_Clayton_{target_date.replace('-', '')}"
            stair_task = ee.batch.Export.image.toDrive(
                image=stair_fcc,
                description=stair_desc,
                folder=export_folder,
                fileNamePrefix=stair_desc,
                scale=scale,
                region=region,
                maxPixels=1e10
            )
            stair_task.start()
            tasks.append(stair_task)
            
            # 2. Export Raw Image (with clouds, for comparison)
            raw_img = full_col.filterDate(target_date, ee.Date(target_date).advance(1, "day")).first()
            raw_fcc = raw_img.select(bands_to_process).clip(region)
            
            raw_desc = f"Raw_Clayton_{target_date.replace('-', '')}"
            raw_task = ee.batch.Export.image.toDrive(
                image=raw_fcc,
                description=raw_desc,
                folder=export_folder,
                fileNamePrefix=raw_desc,
                scale=scale,
                region=region,
                maxPixels=1e10
            )
            raw_task.start()
            tasks.append(raw_task)
            
            # Sleep to avoid API limits
            time.sleep(1)

    print(f"\nSubmitted {len(tasks)} total tasks! Track them at https://code.earthengine.google.com/tasks")

if __name__ == "__main__":
    main()
