import geopandas as gpd
import ee
from pathlib import Path
import time
import traceback
import sys

print("--- STARTING STAIR SECTION 2.2 SCRIPT ---")

try:
    # 1. Initialize Earth Engine
    project_id = "stair-499915" 
    try:
        ee.Initialize(project=project_id)
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project=project_id)

    print("Cloud Engine Online. Replicating STAIR Section 2.2...")

    # 2. Load the Boundary
    BASE_DIR = Path(__file__).resolve().parent.parent
    shapefile_path = BASE_DIR / "data" / "raisin_outline.shp"
    
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Cannot find shapefile at {shapefile_path}")
        
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    aoi = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)

    # 3. Define the Target (Broken) Date and the Timeline (Reference Images)
    target_date = '2021-07-03' 
    time_window_start = '2021-05-15'
    time_window_end = '2021-08-15'

    print(f"Target Date to Impute: {target_date}")
    print(f"Gathering reference images from {time_window_start} to {time_window_end}...")

    # 4. Cloud Masking Function (Creates the "gaps" we need to fill)
    def mask_clouds(image):
        qa = image.select('QA_PIXEL')
        # Bits 3 and 4 are cloud shadow and cloud
        cloud_shadow_bitmask = (1 << 3)
        clouds_bitmask = (1 << 4)
        mask = qa.bitwiseAnd(cloud_shadow_bitmask).eq(0) \
                 .And(qa.bitwiseAnd(clouds_bitmask).eq(0))
        return image.updateMask(mask)

    # 5. Get the Image Collection and mask clouds
    landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(aoi) \
        .filterDate(time_window_start, time_window_end) \
        .map(mask_clouds)

    # 6. PAPER SECTION 2.2.1: Temporal Interpolation via Linear Regression
    def add_time_band(img):
        date_millis = ee.Image(img.date().millis()).divide(1000 * 60 * 60 * 24).toFloat().rename('t')
        return img.select('SR_B5').addBands(date_millis)

    regression_collection = landsat_collection.map(lambda img: add_time_band(img).select(['t', 'SR_B5']))

    print("Calculating temporal linear regression (slope 'a' and intercept 'b') for all pixels...")
    temporal_regression = regression_collection.reduce(ee.Reducer.linearFit())

    slope_a = temporal_regression.select('scale')
    intercept_b = temporal_regression.select('offset')

    # --- PROFESSOR'S EVALUATION REQUESTS ---
    print("Calculating R-square and Data Point Counts for evaluation...")
    n_points = regression_collection.select('SR_B5').count().rename('N_Points')
    pearsons = regression_collection.reduce(ee.Reducer.pearsonsCorrelation())
    r_square = pearsons.select('correlation').pow(2).rename('R_Square')
    slope_matrix = slope_a.rename('Slope')
    intercept_matrix = intercept_b.rename('Intercept')
    
    evaluation_matrix = ee.Image([slope_matrix, intercept_matrix, r_square, n_points]).toDouble().clip(aoi)
    # ---------------------------------------

    # 7. Apply the equation to our Target Date to fill the gaps
    target_date_ee = ee.Date(target_date)
    t0 = ee.Image(target_date_ee.millis()).divide(1000 * 60 * 60 * 24)

    predicted_gaps = t0.multiply(slope_a).add(intercept_b).rename('Imputed_NIR')
    original_broken_image = landsat_collection.filterDate(target_date_ee, target_date_ee.advance(1, 'day')).first().select('SR_B5')

    # Blend them together (Temporal Patching)
    temporal_patched_image = original_broken_image.unmask(predicted_gaps).clip(aoi)
    print("Section 2.2.1 complete. Temporal gaps filled.")

    # 8. PAPER SECTION 2.2.2: Spatial Interpolation
    print("Executing Section 2.2.2: Spatial Interpolation to patch remaining holes...")
    # We use a 10-pixel radius (~300 meters) focal mean to blur surviving pixels into the missing voids
    spatial_neighborhood = temporal_patched_image.focal_mean(radius=10, kernelType='square', units='pixels', iterations=2)
    
    # Unmask the remaining holes with the spatial neighborhood data
    fully_patched_image = temporal_patched_image.unmask(spatial_neighborhood).clip(aoi)
    print("Section 2.2.2 complete. Remaining massive voids filled using spatial neighbors.")

    # 9. Export to Google Drive for Analysis
    task_before = ee.batch.Export.image.toDrive(
        image=original_broken_image.clip(aoi),
        description='STAIR_Before_Original',
        folder='STAIR_Adaptive_Correction',
        scale=30,
        region=aoi,
        maxPixels=1e10
    )
    
    task_temporal = ee.batch.Export.image.toDrive(
        image=temporal_patched_image,
        description='STAIR_After_Temporal_Only',
        folder='STAIR_Adaptive_Correction',
        scale=30,
        region=aoi,
        maxPixels=1e10
    )

    task_spatial = ee.batch.Export.image.toDrive(
        image=fully_patched_image,
        description='STAIR_After_Full_Spatial_Patch',
        folder='STAIR_Adaptive_Correction',
        scale=30,
        region=aoi,
        maxPixels=1e10
    )

    task_eval = ee.batch.Export.image.toDrive(
        image=evaluation_matrix,
        description='STAIR_Evaluation_Matrix',
        folder='STAIR_Adaptive_Correction',
        scale=30,
        region=aoi,
        maxPixels=1e10
    )

    task_before.start()
    task_temporal.start()
    task_spatial.start()
    task_eval.start()
    print("Exporting Before, Temporal, Spatial, and Evaluation files to Google Drive...")

    # 10. Task Monitoring Loop
    while task_before.active() or task_temporal.active() or task_spatial.active() or task_eval.active():
        print(f"Status - Before: {task_before.status()['state']} | Temp: {task_temporal.status()['state']} | Spat: {task_spatial.status()['state']} | Eval: {task_eval.status()['state']}... waiting 10s.")
        time.sleep(10)

    print(f"\nProcess Finished!")
    if task_before.status()['state'] == 'COMPLETED' and task_spatial.status()['state'] == 'COMPLETED':
        print("SUCCESS: All files generated! Check your Google Drive folder: 'STAIR_Adaptive_Correction'")
    else:
        print("FAILED: One or more tasks encountered an error on Google's servers.")

except Exception as e:
    print("\n!!! SCRIPT CRASHED ON YOUR COMPUTER !!!")
    traceback.print_exc()
    sys.exit(1)