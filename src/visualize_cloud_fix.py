import geopandas as gpd
import ee
import folium
from pathlib import Path

# 1. Initialize
project_id = "stair-499915" 
try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

print("Cloud Engine Initialized.")

# 2. Load AOI
BASE_DIR = Path(__file__).resolve().parent.parent
shapefile_path = BASE_DIR / "data" / "raisin_outline.shp"
gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
aoi = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)

# ---------------------------------------------------------
# PHASE 1: RE-LEARN THE SPATIAL RULES (GOLDEN DAY)
# ---------------------------------------------------------
print("Pulling June 17 Golden Day to learn spatial rules...")
golden_landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi).filterDate('2021-06-17', '2021-06-18').first()
golden_modis = ee.ImageCollection("MODIS/061/MOD09GQ").filterBounds(aoi).filterDate('2021-06-17', '2021-06-18').first()

red_stack = golden_modis.select('sur_refl_b01').addBands(golden_landsat.select('SR_B4'))
nir_stack = golden_modis.select('sur_refl_b02').addBands(golden_landsat.select('SR_B5'))

moving_window = ee.Kernel.square(radius=15, units='pixels')
red_weights = red_stack.reduceNeighborhood(reducer=ee.Reducer.linearFit(), kernel=moving_window)
nir_weights = nir_stack.reduceNeighborhood(reducer=ee.Reducer.linearFit(), kernel=moving_window)

red_alpha, red_beta = red_weights.select('scale'), red_weights.select('offset')
nir_alpha, nir_beta = nir_weights.select('scale'), nir_weights.select('offset')

# ---------------------------------------------------------
# PHASE 2: FIND A RUINED CLOUDY DAY
# ---------------------------------------------------------
print("Searching for a highly clouded Landsat image to use as a test case...")
cloudy_landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(aoi) \
    .filterDate('2021-07-01', '2021-08-30') \
    .filter(ee.Filter.gt('CLOUD_COVER', 40)) \
    .first()

cloudy_date = ee.Date(cloudy_landsat.get('system:time_start')).format('YYYY-MM-dd').getInfo()
print(f"Found a ruined image on: {cloudy_date}")

# Calculate the REAL (but corrupted) NDVI
real_red = cloudy_landsat.select('SR_B4')
real_nir = cloudy_landsat.select('SR_B5')
real_ndvi = real_nir.subtract(real_red).divide(real_nir.add(real_red)).clip(aoi)

# ---------------------------------------------------------
# PHASE 3: APPLY STAIR TO FIX THE MISSING DATA
# ---------------------------------------------------------
print("Synthesizing the STAIR prediction for the same day...")
cloudy_date_start = ee.Date(cloudy_date)
cloudy_modis = ee.ImageCollection("MODIS/061/MOD09GQ") \
    .filterBounds(aoi) \
    .filterDate(cloudy_date_start, cloudy_date_start.advance(1, 'day')) \
    .first()

synth_red = cloudy_modis.select('sur_refl_b01').multiply(red_alpha).add(red_beta)
synth_nir = cloudy_modis.select('sur_refl_b02').multiply(nir_alpha).add(nir_beta)
synth_ndvi = synth_nir.subtract(synth_red).divide(synth_nir.add(synth_red)).clip(aoi)

# ---------------------------------------------------------
# PHASE 4: VISUALIZE BEFORE AND AFTER
# ---------------------------------------------------------
print("Generating Interactive HTML Map...")
def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; Google Earth Engine',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)
folium.Map.add_ee_layer = add_ee_layer

centroid = aoi.centroid().getInfo()['coordinates']
m = folium.Map(location=[centroid[1], centroid[0]], zoom_start=11)

# NDVI color palette (Red/Yellow = bad/clouds/dirt, Green = healthy crops)
ndvi_vis = {'min': 0.0, 'max': 0.9, 'palette': ['red', 'yellow', 'green']}

# Add layers
m.add_ee_layer(real_ndvi, ndvi_vis, f"1. REAL Landsat (Corrupted by Clouds) - {cloudy_date}")
m.add_ee_layer(synth_ndvi, ndvi_vis, f"2. SYNTHETIC STAIR (Cloud-Free Fix) - {cloudy_date}")

# Add Boundary
folium.GeoJson(gdf, name="Watershed Boundary", style_function=lambda x: {'fillColor': '#00000000', 'color': 'blue', 'weight': 2}).add_to(m)

m.add_child(folium.LayerControl())
output_html = BASE_DIR / "data" / "cloud_validation_map.html"
m.save(str(output_html))

print(f"\nSUCCESS: Validation map saved to {output_html}")
print("Open the HTML file. Turn off the Synthetic layer to see the cloud damage, then turn it back on to see the STAIR fix.")