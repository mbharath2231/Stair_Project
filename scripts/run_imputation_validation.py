#!/usr/bin/env python3
"""
STAIR Validation: Simulating SLC-off gaps to analyze imputation differences.
This script replicates the methodology from Figure 3 of the STAIR paper (Luo et al. 2018).
It takes a perfectly clear image, artificially introduces missing gaps (stripes), 
runs the imputation pipeline, and exports the stages to Google Drive for QGIS analysis.
"""

import sys
from pathlib import Path
import ee

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from stair.core import init_earth_engine, get_landsat_collection, load_aoi
from stair.temporal import impute_temporal
from stair.adaptive import stair_adaptive_pipeline
from stair.diagnostics import create_chip_bounds

def create_stripe_mask(region: ee.Geometry, spacing: int = 40, width: int = 15) -> ee.Image:
    """
    Creates a diagonal stripe mask to simulate Landsat 7 SLC-off gaps.
    1 = Valid pixel, 0 = Gap
    """
    coords = ee.Image.pixelCoordinates(ee.Projection('EPSG:3857'))
    # Create diagonal stripes by combining X and Y coordinates
    diagonal = coords.select('x').add(coords.select('y'))
    # Modulo creates repeating pattern
    pattern = diagonal.mod(spacing * 30) # roughly 30m per pixel
    # Create the gap (width)
    return pattern.gt(width * 30)

def main():
    init_earth_engine()
    
    aoi, _ = load_aoi()
    centroid = aoi.centroid(maxError=1).getInfo()["coordinates"]
    target_point = ee.Geometry.Point([float(centroid[0]), float(centroid[1])])
    # 7km x 7km chip (3500m radius)
    region = create_chip_bounds(target_point, radius_meters=3500)
    
    scale = 30.0
    
    # We will process NIR (B5), Red (B4), and Green (B3) for False Color Composite
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]
    
    target_date = "2021-08-20" 
    print(f"Selected cloudy target date: {target_date}")
    target_col = get_landsat_collection(region, target_date, ee.Date(target_date).advance(1, "day").format("YYYY-MM-dd").getInfo(), apply_mask=True)
    raw_image_t0 = target_col.first()
    
    target_img_unmasked = ee.ImageCollection(config.LANDSAT8_COLLECTION).filterBounds(region).filterDate(target_date, ee.Date(target_date).advance(1, "day").format("YYYY-MM-dd").getInfo()).first()
    
    reference_date = "2021-06-17" 
    print(f"Selected clear reference date: {reference_date}")
    ref_col = get_landsat_collection(region, reference_date, ee.Date(reference_date).advance(1, "day").format("YYYY-MM-dd").getInfo(), apply_mask=True)
    image_t1 = ref_col.first()
    
    training_col = get_landsat_collection(
        region, "2021-04-01", "2021-11-01", max_cloud_cover=20.0, apply_mask=True
    )
    
    # Intentional Temporal Failure Test (15 day window)
    t0_millis = ee.Date(target_date).millis()
    window_days = 15
    start_exclusion = ee.Date(target_date).advance(-window_days, 'day').millis()
    end_exclusion = ee.Date(target_date).advance(window_days, 'day').millis()
    training_col = training_col.filter(
        ee.Filter.Or(
            ee.Filter.lt("system:time_start", start_exclusion),
            ee.Filter.gt("system:time_start", end_exclusion)
        )
    )
    
    # Dictionaries to hold the single-band results
    stage_images = {
        "0_Original_Cloudy": [],
        "1_Cloud_Masked": [],
        "2_Temporal_Interpolation": [],
        "3_Global_Correction": [],
        "4_Adaptive_Correction": []
    }
    
    # NEW: Precompute unified Land Cover segments using all bands!
    # This guarantees that the NIR, Red, and Green offsets are applied to the exact same field boundaries.
    print("\nPrecomputing multi-band K-Means clusters for Adaptive Correction...")
    ref_stack = image_t1.select(bands_to_process)
    training_data = ref_stack.sample(region=region, scale=scale, numPixels=5000, tileScale=1)
    clusterer = ee.Clusterer.wekaKMeans(4).train(training_data)
    master_segments = ref_stack.cluster(clusterer).rename("seg")
    
    for band in bands_to_process:
        print(f"\n--- Processing Band: {band} ---")
        pred_linear, _ = impute_temporal(training_col, raw_image_t0, target_date, band_name=band)
        
        results = stair_adaptive_pipeline(
            raw_image_t0=raw_image_t0,
            image_t1=image_t1,
            linear_prediction=pred_linear,
            region=region,
            n_clusters=4,
            band_name=band,
            scale=scale,
            tile_scale=1,
            precomputed_segments=master_segments
        )
        
        stage_images["0_Original_Cloudy"].append(target_img_unmasked.select(band))
        stage_images["1_Cloud_Masked"].append(results["stage1_broken"])
        stage_images["2_Temporal_Interpolation"].append(results["stage2_temporal"])
        stage_images["3_Global_Correction"].append(results["stage3a_global"])
        stage_images["4_Adaptive_Correction"].append(results["stage3b_adaptive"])
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_folder = "STAIR_Validation_QGIS"
    print(f"\nSubmitting multi-band exports to Google Drive Folder: '{export_folder}'")
    
    tasks = []
    for name, band_list in stage_images.items():
        # Combine the [B5, B4, B3] single-band images into one multi-band image
        multi_band_img = ee.Image(band_list).toDouble()
        desc = f"STAIR_FCC_{name}"
        file_prefix = f"STAIR_FCC_{timestamp}_{name}"
        task = ee.batch.Export.image.toDrive(
            image=multi_band_img,
            description=desc,
            folder=export_folder,
            fileNamePrefix=file_prefix,
            scale=scale,
            region=region,
            maxPixels=1e10
        )
        task.start()
        tasks.append(task)
        print(f"Submitted multi-band task: {desc}")

    print("\nTasks submitted! You can track them at https://code.earthengine.google.com/tasks")
    print("Once finished, download the GeoTIFFs from Google Drive and drag them into QGIS.")
    print("You can visually compare how the cloud holes blend in '2_Temporal', '3_Global', and '4_Adaptive'.")

if __name__ == "__main__":
    main()
