import geopandas as gpd
import ee
from pathlib import Path
import time

# 1. Initialize
project_id = "stair-499915" 
try:
    ee.Initialize(project=project_id)
except:
    ee.Authenticate()
    ee.Initialize(project=project_id)

BASE_DIR = Path(__file__).resolve().parent.parent
gdf = gpd.read_file(BASE_DIR / "data" / "raisin_outline.shp").to_crs("EPSG:4326")
aoi = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)

# 2. Setup Collection (July 19th - The "Broken" Target)
target_date = '2021-07-19'
col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi).filterDate('2021-05-15', '2021-08-15')

def mask_clouds(img):
    qa = img.select('QA_PIXEL')
    return img.updateMask(qa.bitwiseAnd(1<<3).eq(0).And(qa.bitwiseAnd(1<<4).eq(0)))

# Regression Setup
masked_col = col.map(mask_clouds)
t_col = masked_col.map(lambda i: i.select('SR_B5').addBands(ee.Image(i.date().millis()).divide(1000*60*60*24).rename('t')))
fit = t_col.reduce(ee.Reducer.linearFit())
slope, intercept = fit.select('scale'), fit.select('offset')

# 3. Process Stages
raw = col.filterDate(target_date, ee.Date(target_date).advance(1, 'day')).first().select('SR_B5')
mask = raw.mask() # Current valid pixels

# Temporal Patching
t_val = ee.Date(target_date).millis().divide(1000*60*60*24)
predicted = ee.Image(t_val).multiply(slope).add(intercept)
temporal = raw.unmask(predicted)

# Spatial Patching (Blur remaining gaps)
spatial = temporal.unmask(temporal.focal_mean(10))

# 4. Diagnostic Tracker: The "Staircase" Visualizer
# 0 = Original (Valid), 1 = Temporal Patch, 2 = Spatial Patch
tracker = ee.Image(0).where(mask.eq(0).And(temporal.mask().eq(1)), 1) \
                     .where(mask.eq(0).And(temporal.mask().eq(0)), 2) \
                     .clip(aoi)

# 5. Export
print("Submitting diagnostic pipeline to Google Drive...")
for img, name in [(raw, 'Stage1_Broken'), (temporal, 'Stage2_Temporal'), (spatial, 'Stage3_Spatial'), (tracker, 'Diagnostic_Fill_Tracker')]:
    ee.batch.Export.image.toDrive(
        image=img.toDouble().clip(aoi),
        description=f'Watershed_July19_{name}',
        folder='STAIR_Staircase_Analysis',
        scale=30, region=aoi, maxPixels=1e10
    ).start()

print("Tasks submitted! Check 'STAIR_Staircase_Analysis' in your Drive.")