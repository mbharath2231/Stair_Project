import geopandas as gpd
import ee
from pathlib import Path

project_id = "stair-499915" 
try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

print("Cloud Engine Online. Starting Red Band Validation Test...")

BASE_DIR = Path(__file__).resolve().parent.parent
shapefile_path = BASE_DIR / "data" / "raisin_outline.shp"

if not shapefile_path.exists():
    raise FileNotFoundError(f"CRITICAL: Shapefile not found at {shapefile_path}")

gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
aoi = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)

print("PHASE 1: Learning Red Band physics from June 17...")
golden_landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi).filterDate('2021-06-17', '2021-06-18').first()
golden_modis = ee.ImageCollection("MODIS/061/MOD09GQ").filterBounds(aoi).filterDate('2021-06-17', '2021-06-18').first()

red_stack = golden_modis.select('sur_refl_b01').addBands(golden_landsat.select('SR_B4'))
moving_window = ee.Kernel.square(radius=15, units='pixels')

# Calculate Alpha (scale) and Beta (offset) ONLY for the Red Band
red_weights = red_stack.reduceNeighborhood(reducer=ee.Reducer.linearFit(), kernel=moving_window)
red_alpha = red_weights.select('scale')
red_beta = red_weights.select('offset')

print("PHASE 2: Setting up the July 3 Test Day...")
test_date_start = '2021-07-03'
test_date_end = '2021-07-04'

# 1. The Blurry Input (Real MODIS Red Band on July 3)
test_modis = ee.ImageCollection("MODIS/061/MOD09GQ").filterBounds(aoi).filterDate(test_date_start, test_date_end).first()
modis_red_blurry = test_modis.select('sur_refl_b01').rename('MODIS_Red_Blurry')

# 2. The Physical Truth (Real Landsat Red Band on July 3)
test_landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi).filterDate(test_date_start, test_date_end).first()
landsat_red_real = test_landsat.select('SR_B4').rename('REAL_Landsat_Red')

print("PHASE 3: Predicting the Sharp Red Band using STAIR math...")
# Equation: Synthetic = (MODIS * Alpha) + Beta
stair_red_sharp = modis_red_blurry.multiply(red_alpha).add(red_beta).rename('STAIR_Red_Sharp')

print("PHASE 4: Exporting the 3-Layer Validation Matrix...")
# We stack them together into one file for direct pixel-by-pixel comparison in QGIS
validation_image = ee.Image([modis_red_blurry, stair_red_sharp, landsat_red_real]).clip(aoi)

file_name = 'red_band_validation_20210703'
task = ee.batch.Export.image.toDrive(
    image=validation_image,
    description=file_name,
    folder='STAIR_Exports_Red', 
    fileNamePrefix=file_name,
    scale=30, # Force standard 30m resolution 
    region=aoi,
    crs='EPSG:4326',
    maxPixels=1e10 
)
task.start()

print(f"Submitted task to Google Drive: {file_name}")
print("Wait for it to process, download it, and drag it into QGIS for the ultimate pixel test.")