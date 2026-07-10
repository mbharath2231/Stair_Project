import geopandas as gpd
import ee
from pathlib import Path
import sys
import time

print("--- STARTING 6-MONTH SPATIAL TIME-SERIES EXPORT ---")

# 1. Initialize Earth Engine
project_id = "stair-499915" 
try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

print("Cloud Engine Online.")

# 2. Load the Boundary
BASE_DIR = Path(__file__).resolve().parent.parent
shapefile_path = BASE_DIR / "data" / "raisin_outline.shp"

if not shapefile_path.exists():
    raise FileNotFoundError(f"Cannot find shapefile at {shapefile_path}")
    
gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
aoi = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)

print("Scanning USDA Cropland Data Layer to isolate a guaranteed agricultural pixel...")

# We force EE to find Corn (1) or Soy (5)
cdl = ee.Image('USDA/NASS/CDL/2021').select('cropland')
corn_soy_mask = cdl.eq(1).Or(cdl.eq(5))

# BULLETPROOF SEARCH: We use native sampling to pull 50 valid crop pixels. 
crop_points = corn_soy_mask.updateMask(corn_soy_mask).sample(
    region=aoi,
    scale=30,
    numPixels=50, 
    geometries=True
)

features = crop_points.getInfo().get('features', [])

if not features:
    raise ValueError("CRITICAL FAILURE: Could not find any Corn or Soy pixels in this AOI.")

coords = features[0]['geometry']['coordinates']
target_lon, target_lat = coords[0], coords[1]
target_pixel = ee.Geometry.Point([target_lon, target_lat])

# CREATE THE 5KM x 5KM PIECE OF LAND (Bounding Box)
# A 2500m buffer around the center point creates a 5km diameter chip
chip_bounds = target_pixel.buffer(2500).bounds()

print(f"Target locked. Created a 5km x 5km patch of land around Longitude: {target_lon:.4f}, Latitude: {target_lat:.4f}")

# 3. Define the 6-Month Time Window
start_date = '2021-04-01'
end_date = '2021-10-31'

# 4. Define Cloud Mask
def mask_clouds(image):
    qa = image.select('QA_PIXEL')
    cloud_shadow_bitmask = (1 << 3)
    clouds_bitmask = (1 << 4)
    mask = qa.bitwiseAnd(cloud_shadow_bitmask).eq(0) \
             .And(qa.bitwiseAnd(clouds_bitmask).eq(0))
    return image.updateMask(mask)

# 5. Get the Image Collections (Clipped to our 5km chip to save memory)
raw_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(chip_bounds).filterDate(start_date, end_date)
masked_collection = raw_collection.map(mask_clouds)

# 6. Calculate STAIR Temporal Regression for the 5km patch
print("Calculating STAIR Temporal physics for this specific 5km patch...")
def add_time_band(img):
    t = ee.Image(img.date().millis()).divide(1000 * 60 * 60 * 24).toFloat().rename('t')
    return img.select('SR_B5').addBands(t)

reg_col = masked_collection.map(add_time_band).select(['t', 'SR_B5'])
regression = reg_col.reduce(ee.Reducer.linearFit())

slope = regression.select('scale')
intercept = regression.select('offset')

# 7. Extract Every Single Pass and Export
print("Fetching the schedule of all Landsat overpasses...")
# Convert EE Collection to a List so we can loop through it in Python
image_list = raw_collection.toList(raw_collection.size())
num_images = image_list.size().getInfo()

print(f"Discovered {num_images} total Landsat passes over this farm in 6 months.")
print("Queueing Google Drive exports for EVERY pass...")

for i in range(num_images):
    # Extract the specific image from the EE List
    raw_img = ee.Image(image_list.get(i))
    
    # Get the date string for the filename
    date_str = raw_img.date().format('YYYY-MM-dd').getInfo()
    
    # Band 1: Raw Data (Includes the physical clouds)
    raw_nir = raw_img.select('SR_B5').rename('Band1_Raw_With_Clouds')
    
    # Band 2: STAIR Imputed Data (The pure mathematical guess based on the trendline)
    t = ee.Image(raw_img.date().millis()).divide(1000 * 60 * 60 * 24).toFloat()
    stair_imputed = t.multiply(slope).add(intercept).rename('Band2_STAIR_Imputed_Only')
    
    # Band 3: Final Patched Data (Clear pixels survive, cloudy holes get filled with STAIR)
    masked_nir = mask_clouds(raw_img).select('SR_B5')
    final_patched = masked_nir.unmask(stair_imputed).rename('Band3_Final_Patched')
    
    # Combine into a single multi-band export stack and FORCE double-precision to prevent export crashes
    export_stack = ee.Image([raw_nir, stair_imputed, final_patched]).toDouble().clip(chip_bounds)
    
    # Format filename: STAIR_5km_YYYYMMDD
    file_name = f'STAIR_5km_{date_str.replace("-", "")}'
    
    task = ee.batch.Export.image.toDrive(
        image=export_stack,
        description=file_name,
        folder='STAIR_Time_Lapse',
        scale=30,
        region=chip_bounds,
        maxPixels=1e8
    )
    task.start()
    print(f"[{i+1}/{num_images}] Export task submitted for {date_str}...")
    
    # Brief pause to prevent Earth Engine API rate limiting
    time.sleep(0.5)

print("\nSUCCESS: All tasks have been submitted to Google's servers!")
print("Check your Google Drive folder: 'STAIR_Time_Lapse'.")
print("Warning: It will take a few minutes for Google to process and save all 14 files.")