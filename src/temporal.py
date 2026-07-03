import geopandas as gpd
import ee
from pathlib import Path
import time
import traceback
import sys

print("--- STARTING STAIR SECTION 2.2.1 SCRIPT ---")

try:
    # 1. Initialize Earth Engine
    project_id = "stair-499915" 
    try:
        ee.Initialize(project=project_id)
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project=project_id)

    print("Cloud Engine Online. Replicating STAIR Section 2.2.1...")

    # 2. Load the Boundary
    BASE_DIR = Path(__file__).resolve().parent.parent
    shapefile_path = BASE_DIR / "data" / "raisin_outline.shp"
    
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Cannot find shapefile at {shapefile_path}")
        
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    aoi = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)

    # 3. Define the Target (Broken) Date and the Timeline (Reference Images)
    target_date = '2021-07-03' 
    time_window_start = '2021-04-01'
    time_window_end = '2021-10-31'

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
    # Equation: L_linear(pg, t0) = a*t0 + b
    # We need to find 'a' (slope) and 'b' (intercept) for every single pixel over time.

    def add_time_band(img):
        # Add a band that represents the "time" (t) for the linear regression (X-axis)
        # We use milliseconds since epoch, scaled down to days for stability
        # CRITICAL FIX: Added .toFloat() to force Earth Engine to treat all dates as the same data type
        date_millis = ee.Image(img.date().millis()).divide(1000 * 60 * 60 * 24).toFloat().rename('t')
        # We will run the regression on the NIR band (SR_B5) as an example (Y-axis)
        return img.select('SR_B5').addBands(date_millis)

    # Map the time band across our reference collection
    # We put 't' first because linearFit expects (X, Y) which is (independent, dependent)
    regression_collection = landsat_collection.map(lambda img: add_time_band(img).select(['t', 'SR_B5']))

    print("Calculating temporal linear regression (slope 'a' and intercept 'b') for all pixels...")
    # ee.Reducer.linearFit() calculates the slope (scale) and intercept (offset) across time
    temporal_regression = regression_collection.reduce(ee.Reducer.linearFit())

    slope_a = temporal_regression.select('scale')
    intercept_b = temporal_regression.select('offset')

    # 7. Apply the equation to our Target Date to fill the gaps
    target_date_ee = ee.Date(target_date)
    t0 = ee.Image(target_date_ee.millis()).divide(1000 * 60 * 60 * 24)

    # L_linear = a * t0 + b
    predicted_gaps = t0.multiply(slope_a).add(intercept_b).rename('Imputed_NIR')

    # Get the original broken image
    original_broken_image = landsat_collection.filterDate(target_date_ee, target_date_ee.advance(1, 'day')).first().select('SR_B5')

    # Blend them together: Keep the original good pixels, but fill the holes with our prediction
    gap_filled_image = original_broken_image.unmask(predicted_gaps).clip(aoi)

    print("Section 2.2.1 complete. Gaps mathematically filled using temporal interpolation.")

    # 8. Export BOTH to Google Drive for Before/After Analysis
    
    # Task A: Export the "Before" (Original with gaps)
    task_before = ee.batch.Export.image.toDrive(
        image=original_broken_image.clip(aoi),
        description='STAIR_Before_Original',
        folder='STAIR_Temporal_Interpolation',
        scale=30,
        region=aoi,
        maxPixels=1e10
    )
    
    # Task B: Export the "After" (Imputed/Filled)
    task_after = ee.batch.Export.image.toDrive(
        image=gap_filled_image,
        description='STAIR_After_Imputed',
        folder='STAIR_Temporal_Interpolation',
        scale=30,
        region=aoi,
        maxPixels=1e10
    )

    task_before.start()
    task_after.start()
    print("Exporting both 'Before' and 'After' files to Google Drive...")

    # 9. Task Monitoring Loop
    while task_before.active() or task_after.active():
        print(f"Status - Before: {task_before.status()['state']} | After: {task_after.status()['state']}... waiting 10s.")
        time.sleep(10)

    print(f"\nProcess Finished!")
    print(f"Before Task Status: {task_before.status()['state']}")
    print(f"After Task Status: {task_after.status()['state']}")
    
    if task_before.status()['state'] == 'COMPLETED' and task_after.status()['state'] == 'COMPLETED':
        print("SUCCESS: Both files generated! Check your Google Drive folder: 'STAIR_Temporal'")
    else:
        print("FAILED: One or both tasks encountered an error on Google's servers.")
        print(f"Before Error: {task_before.status().get('error_message', 'None')}")
        print(f"After Error: {task_after.status().get('error_message', 'None')}")

except Exception as e:
    print("\n!!! SCRIPT CRASHED ON YOUR COMPUTER !!!")
    traceback.print_exc()
    sys.exit(1)