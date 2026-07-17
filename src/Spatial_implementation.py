import geopandas as gpd
import ee
from pathlib import Path
import time
import sys

print("--- STARTING FULL WATERSHED STAIR INTERPOLATION ---")

# 1. Initialize Earth Engine
project_id = "stair-499915" 
try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

print("Cloud Engine Online. Targeting the entire Raisin Watershed...")

# 2. Load the Full Watershed Boundary
BASE_DIR = Path(__file__).resolve().parent.parent
shapefile_path = BASE_DIR / "data" / "raisin_outline.shp"

if not shapefile_path.exists():
    raise FileNotFoundError(f"CRITICAL ERROR: Shapefile not found at {shapefile_path}")
    
gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
aoi = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)

# 3. Define the Target Date and Window
# SWITCHED TO A BRUTAL DATE: July 19th had terrible persistent cloud cover.
target_date = '2021-07-19' 
time_window_start = '2021-05-15'
time_window_end = '2021-08-15'

print(f"Target Date to Impute: {target_date}")
print(f"Gathering historical data from {time_window_start} to {time_window_end}...")

# 4. Cloud Masking Function
def mask_clouds(image):
    qa = image.select('QA_PIXEL')
    cloud_shadow_bitmask = (1 << 3)
    clouds_bitmask = (1 << 4)
    mask = qa.bitwiseAnd(cloud_shadow_bitmask).eq(0) \
             .And(qa.bitwiseAnd(clouds_bitmask).eq(0))
    return image.updateMask(mask)

landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(aoi) \
    .filterDate(time_window_start, time_window_end) \
    .map(mask_clouds)

# 5. Calculate Temporal Linear Regression (y = mx + b)
print("Calculating timeline math (slope and intercept) for the entire watershed...")
def add_time_band(img):
    date_millis = ee.Image(img.date().millis()).divide(1000 * 60 * 60 * 24).toFloat().rename('t')
    return img.select('SR_B5').addBands(date_millis)

regression_collection = landsat_collection.map(add_time_band).select(['t', 'SR_B5'])
temporal_regression = regression_collection.reduce(ee.Reducer.linearFit())

slope_a = temporal_regression.select('scale')
intercept_b = temporal_regression.select('offset')

# Calculate Evaluation Metrics
print("Generating Evaluation Metrics (R-Square & Data Points)...")
n_points = regression_collection.select('SR_B5').count().rename('N_Points')
pearsons = regression_collection.reduce(ee.Reducer.pearsonsCorrelation())
r_square = pearsons.select('correlation').pow(2).rename('R_Square')

evaluation_matrix = ee.Image([slope_a.rename('Slope'), intercept_b.rename('Intercept'), r_square, n_points]).toDouble().clip(aoi)

# 6. Apply STAIR Interpolation to Target Date
print(f"Executing STAIR Sections 2.2.1 and 2.2.2 on {target_date}...")
target_date_ee = ee.Date(target_date)
t0 = ee.Image(target_date_ee.millis()).divide(1000 * 60 * 60 * 24).toFloat()

# STAGE 1: Original Broken Data
original_broken = landsat_collection.filterDate(target_date_ee, target_date_ee.advance(1, 'day')).first().select('SR_B5').rename('Band1_Raw')

# STAGE 2: Temporal Interpolation Only (y = mx + b)
predicted_temporal = t0.multiply(slope_a).add(intercept_b)
temporal_patched = original_broken.unmask(predicted_temporal).rename('Band1_Temporal_Patch')

# STAGE 3: Full Spatial Interpolation (Blurring remaining holes)
spatial_neighborhood = temporal_patched.focal_mean(radius=10, kernelType='square', units='pixels', iterations=2)
fully_patched = temporal_patched.unmask(spatial_neighborhood).rename('Band1_Spatial_Patch')

# Cast everything to Double to prevent EE datatype crash errors
stage1_img = original_broken.toDouble().clip(aoi)
stage2_img = temporal_patched.toDouble().clip(aoi)
stage3_img = fully_patched.toDouble().clip(aoi)

# 7. Export the distinct files
print("Submitting Export Tasks to Google Drive...")
tasks = [
    ee.batch.Export.image.toDrive(image=stage1_img, description='Watershed_July19_Stage1_Broken', folder='STAIR_Watershed', scale=30, region=aoi, maxPixels=1e10),
    ee.batch.Export.image.toDrive(image=stage2_img, description='Watershed_July19_Stage2_Temporal', folder='STAIR_Watershed', scale=30, region=aoi, maxPixels=1e10),
    ee.batch.Export.image.toDrive(image=stage3_img, description='Watershed_July19_Stage3_Spatial', folder='STAIR_Watershed', scale=30, region=aoi, maxPixels=1e10),
]

for t in tasks:
    t.start()

print("All tasks submitted! Please wait for Google servers to process them (~5-10 minutes).")
while any(t.active() for t in tasks):
    statuses = [t.status()['state'][:4] for t in tasks] 
    print(f"Tasks: {statuses} ... waiting 15s")
    time.sleep(15)

print("\nSUCCESS! Check your 'STAIR_Watershed' Drive folder for the July 19th sequence.")