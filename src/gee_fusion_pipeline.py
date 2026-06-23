import geopandas as gpd
import ee
from pathlib import Path

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

# 5. Find a TRUE Golden Day (Clear Skies)
# We cannot hardcode a single day. We must search a temporal window.
start_date = '2021-05-01'
end_date = '2021-07-01' # 2-month window guarantees at least a few Landsat passes

print(f"Searching for clear NASA data between {start_date} and {end_date}...")

# Query Landsat, filter by area, date, and strictly less than 10% cloud cover
landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(aoi) \
    .filterDate(start_date, end_date) \
    .filter(ee.Filter.lt('CLOUD_COVER', 10))

# DEFENSIVE CHECK: Did we actually find anything?
count = landsat_collection.size().getInfo()
if count == 0:
    raise ValueError(f"CRITICAL: No clear Landsat images found between {start_date} and {end_date}. Expand your search window.")

# Extract the best image and its ID
landsat_golden = landsat_collection.first()
landsat_id = landsat_golden.get('system:id').getInfo()

# 6. Synchronize MODIS to the exact Landsat Date (Corrected)
golden_time_ms = landsat_golden.get('system:time_start').getInfo()

# Strip the hours/minutes to get the pure calendar day as a string (YYYY-MM-dd)
golden_date_string = ee.Date(golden_time_ms).format('YYYY-MM-dd').getInfo()

# Create a strict calendar-day window
golden_date_start = ee.Date(golden_date_string)
golden_date_end = golden_date_start.advance(1, 'day')

# Query MODIS
modis_golden = ee.ImageCollection("MODIS/061/MOD09GQ") \
    .filterBounds(aoi) \
    .filterDate(golden_date_start, golden_date_end) \
    .first()

modis_id = modis_golden.get('system:id').getInfo()

print("\n--- PERFECT MATCHING GOLDEN DAY ---")
print(f"Landsat ID: {landsat_id}")
print(f"MODIS ID:   {modis_id}")
print("-----------------------------------")