import geopandas as gpd
import ee
from pathlib import Path
import requests
import zipfile
import io

# 1. Authenticate and Initialize the Cloud Engine
project_id = "stair-499915" # Your registered project ID

try:
    ee.Initialize(project=project_id)
except Exception as e:
    print("Initializing failed. Forcing re-authentication...")
    ee.Authenticate()
    ee.Initialize(project=project_id)

print("Cloud Engine Initialized.")

# 2. Dynamically Resolve the Absolute Path
BASE_DIR = Path(__file__).resolve().parent.parent
shapefile_path = BASE_DIR / "data" / "raisin_outline.shp"

if not shapefile_path.exists():
    raise FileNotFoundError(f"CRITICAL: Shapefile not found at {shapefile_path}. Check your folder structure.")

print(f"Loading Shapefile from: {shapefile_path}")
gdf = gpd.read_file(shapefile_path)

# 3. Ensure the Coordinate System is standard GPS (WGS84 / EPSG:4326)
if gdf.crs != "EPSG:4326":
    print(f"Reprojecting from {gdf.crs} to EPSG:4326...")
    gdf = gdf.to_crs("EPSG:4326")

# 4. Extract the Geometry for Earth Engine
geojson_geom = gdf.geometry.iloc[0].__geo_interface__
aoi = ee.Geometry(geojson_geom)

print("Area of Interest (AOI) successfully mapped to Earth Engine.")

# ---------------------------------------------------------
# PHASE 1: LEARN THE SPATIAL RULES (THE GOLDEN DAY)
# ---------------------------------------------------------
print("\n--- PHASE 1: LEARNING SPATIAL RULES ---")
start_date = '2021-05-01'
end_date = '2021-07-01' 

print(f"Searching for clear NASA data between {start_date} and {end_date}...")

landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(aoi) \
    .filterDate(start_date, end_date) \
    .filter(ee.Filter.lt('CLOUD_COVER', 10))

count = landsat_collection.size().getInfo()
if count == 0:
    raise ValueError(f"CRITICAL: No clear Landsat images found between {start_date} and {end_date}. Expand your search window.")

landsat_golden = landsat_collection.first()
landsat_id = landsat_golden.get('system:id').getInfo()

# Synchronize MODIS to the exact Landsat Date
golden_time_ms = landsat_golden.get('system:time_start').getInfo()
golden_date_string = ee.Date(golden_time_ms).format('YYYY-MM-dd').getInfo()

golden_date_start = ee.Date(golden_date_string)
golden_date_end = golden_date_start.advance(1, 'day')

modis_golden = ee.ImageCollection("MODIS/061/MOD09GQ") \
    .filterBounds(aoi) \
    .filterDate(golden_date_start, golden_date_end) \
    .first()

modis_id = modis_golden.get('system:id').getInfo()

print(f"Landsat ID: {landsat_id}")
print(f"MODIS ID:   {modis_id}")

# Isolate the Raw Surface Reflectance Bands
landsat_red = landsat_golden.select('SR_B4').rename('landsat_red')
landsat_nir = landsat_golden.select('SR_B5').rename('landsat_nir')

modis_red = modis_golden.select('sur_refl_b01').rename('modis_red')
modis_nir = modis_golden.select('sur_refl_b02').rename('modis_nir')

# Stack the Matrices Separately [X (MODIS), Y (Landsat)]
red_stack = modis_red.addBands(landsat_red)
nir_stack = modis_nir.addBands(landsat_nir)

# Execute the Adaptive Spatial Regression for EACH Band
moving_window = ee.Kernel.square(radius=15, units='pixels')

print("Executing server-side spatial linear regression for Red & NIR bands...")
red_weights = red_stack.reduceNeighborhood(reducer=ee.Reducer.linearFit(), kernel=moving_window)
nir_weights = nir_stack.reduceNeighborhood(reducer=ee.Reducer.linearFit(), kernel=moving_window)

# Extract the learned rules (Scale = Alpha, Offset = Beta)
red_alpha = red_weights.select('scale')
red_beta = red_weights.select('offset')
nir_alpha = nir_weights.select('scale')
nir_beta = nir_weights.select('offset')

print("Spatial Rules locked in memory.")

# ---------------------------------------------------------
# PHASE 2: SYNTHESIZING THE MISSING DAY
# ---------------------------------------------------------
print("\n--- PHASE 2: SYNTHESIZING MISSING DATA ---")

# Target a day we know Landsat is offline
prediction_date = '2021-06-22'
pred_start = ee.Date(prediction_date)
pred_end = pred_start.advance(1, 'day')

print(f"Targeting Prediction Day: {prediction_date} (Landsat is offline)")

# Pull the MODIS image for the missing day
modis_pred_raw = ee.ImageCollection("MODIS/061/MOD09GQ") \
    .filterBounds(aoi) \
    .filterDate(pred_start, pred_end) \
    .first()

modis_pred_red = modis_pred_raw.select('sur_refl_b01')
modis_pred_nir = modis_pred_raw.select('sur_refl_b02')

# Execute the STAIR Equations: Synthetic = (MODIS * alpha) + beta
print("Applying spatial rules to synthesize high-res raw bands...")
synthetic_red = modis_pred_red.multiply(red_alpha).add(red_beta).rename('synthetic_red')
synthetic_nir = modis_pred_nir.multiply(nir_alpha).add(nir_beta).rename('synthetic_nir')

# Calculate Final Synthetic NDVI from our synthetic bands
print("Calculating final Synthetic NDVI...")
# NDVI = (NIR - Red) / (NIR + Red)
synthetic_ndvi = synthetic_nir.subtract(synthetic_red).divide(synthetic_nir.add(synthetic_red)).rename('synthetic_ndvi')

# Clip the final output to strictly fit your local shapefile boundary
final_fused_matrix = synthetic_ndvi.clip(aoi)

print("\n--- PIPELINE SUCCESS ---")
print(f"Synthetic 30m NDVI Matrix successfully generated for {prediction_date}.")
print(f"Final available bands in memory: {final_fused_matrix.bandNames().getInfo()}")

# ---------------------------------------------------------
# PHASE 3: EXPORT TO GOOGLE DRIVE (BATCH EXPORT)
# ---------------------------------------------------------
print("\n--- PHASE 3: EXPORTING TO GOOGLE DRIVE ---")
print("Watershed is too large for direct download. Submitting batch task to Google servers...")

file_name = f'synthetic_ndvi_{prediction_date.replace("-", "")}'

# Create the batch task
export_task = ee.batch.Export.image.toDrive(
    image=final_fused_matrix,
    description=file_name,
    folder='STAIR_Exports', # This folder will be created in your Google Drive
    fileNamePrefix=file_name,
    scale=30, # Strict Landsat 30m resolution
    region=aoi,
    crs='EPSG:4326',
    maxPixels=1e10 # Override the default 100M pixel safety limit
)

# Send the command to Google's servers
export_task.start()

print("\n--- FINAL SUCCESS ---")
print("Task successfully submitted to Earth Engine!")
print("1. Go to your Google Drive associated with this project.")
print("2. Wait a few minutes (cloud compute takes time).")
print("3. Look for a folder named 'STAIR_Exports'. Your .tif file will appear there.")