import geopandas as gpd
import pandas as pd
import ee
from pathlib import Path
import time

# 1. Brutal Initialization Check
project_id = "stair-499915" 
try:
    ee.Initialize(project=project_id)
except Exception as e:
    print("Authentication dead. Re-authenticating...")
    ee.Authenticate()
    ee.Initialize(project=project_id)

print("Cloud Engine Online.")

# 2. Load the Boundary
BASE_DIR = Path(__file__).resolve().parent.parent
shapefile_path = BASE_DIR / "data" / "raisin_outline.shp"

if not shapefile_path.exists():
    raise FileNotFoundError(f"CRITICAL: Where is the shapefile? Not found at {shapefile_path}")

gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
aoi = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)
print("AOI locked.")

# ---------------------------------------------------------
# PHASE 1: LEARN THE PHYSICS (THE ANCHOR)
# ---------------------------------------------------------
print("\n--- PHASE 1: LOCKING SPATIAL RULES ---")
# We use your proven June 17 Golden Day to anchor the spatial regression
golden_landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi).filterDate('2021-06-17', '2021-06-18').first()
golden_modis = ee.ImageCollection("MODIS/061/MOD09GQ").filterBounds(aoi).filterDate('2021-06-17', '2021-06-18').first()

red_stack = golden_modis.select('sur_refl_b01').addBands(golden_landsat.select('SR_B4'))
nir_stack = golden_modis.select('sur_refl_b02').addBands(golden_landsat.select('SR_B5'))

moving_window = ee.Kernel.square(radius=15, units='pixels')
print("Calculating Alpha and Beta coefficients on Google Servers...")
red_weights = red_stack.reduceNeighborhood(reducer=ee.Reducer.linearFit(), kernel=moving_window)
nir_weights = nir_stack.reduceNeighborhood(reducer=ee.Reducer.linearFit(), kernel=moving_window)

red_alpha, red_beta = red_weights.select('scale'), red_weights.select('offset')
nir_alpha, nir_beta = nir_weights.select('scale'), nir_weights.select('offset')
print("Rules locked.")

# ---------------------------------------------------------
# PHASE 2: 6-MONTH TEMPORAL LOOP (DATA GENERATION)
# ---------------------------------------------------------
print("\n--- PHASE 2: 6-MONTH TIME SERIES GENERATION ---")
# Define your 6-month growing season
start_season = '2021-05-01'
end_season = '2021-10-31'

# Create a list of dates, jumping every 8 days to avoid API rate limits
# This yields roughly 23 distinct images for your ML model time-series
date_range = pd.date_range(start=start_season, end=end_season, freq='8D')

print(f"Submitting {len(date_range)} STAIR synthesis tasks to Earth Engine...")

for target_date in date_range:
    date_str = target_date.strftime('%Y-%m-%d')
    date_ee = ee.Date(date_str)
    
    # 1. Pull MODIS for the target day
    target_modis = ee.ImageCollection("MODIS/061/MOD09GQ") \
        .filterBounds(aoi) \
        .filterDate(date_ee, date_ee.advance(1, 'day')) \
        .first()
    
    # 2. Apply STAIR Math (Imputation of high-res missing data)
    synth_red = target_modis.select('sur_refl_b01').multiply(red_alpha).add(red_beta)
    synth_nir = target_modis.select('sur_refl_b02').multiply(nir_alpha).add(nir_beta)
    
    # 3. Calculate NDVI
    synth_ndvi = synth_nir.subtract(synth_red).divide(synth_nir.add(synth_red)).rename('ndvi').clip(aoi)
    
    # 4. Submit to Google Drive
    file_name = f'stair_ndvi_{target_date.strftime("%Y%m%d")}'
    
    task = ee.batch.Export.image.toDrive(
        image=synth_ndvi,
        description=file_name,
        folder='STAIR_6Month_Dataset', 
        fileNamePrefix=file_name,
        scale=30, 
        region=aoi,
        crs='EPSG:4326',
        maxPixels=1e10 
    )
    task.start()
    print(f"Submitted task for: {date_str}")
    
    # Sleep briefly so Google doesn't ban us for API spamming
    time.sleep(1)

print("\n--- PIPELINE EXECUTION COMPLETE ---")
print("All tasks are now in Google's queue. Check your Earth Engine Task Manager or Google Drive.")
print("This will take 30-60 minutes to complete entirely. Do not run this script again until they finish.")