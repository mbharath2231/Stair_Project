import geopandas as gpd
import ee
import folium
from pathlib import Path

project_id = "stair-499915"
try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

print("Engine Initialized.")

BASE_DIR = Path(__file__).resolve().parent.parent
shapefile_path = BASE_DIR / "data" / "raisin_outline.shp"

if not shapefile_path.exists():
    raise FileNotFoundError(f"CRITICAL: Shapefile not found at {shapefile_path}")

gdf = gpd.read_file(shapefile_path)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

geojson_geom = gdf.geometry.iloc[0].__geo_interface__
aoi = ee.Geometry(geojson_geom)

# 3. Pull the Exact Golden Day We Found (June 17, 2021)
date_start = '2021-06-17'
date_end = '2021-06-18'

landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(aoi) \
    .filterDate(date_start, date_end) \
    .first()

# 4. Isolate the Raw Bands
landsat_red = landsat.select('SR_B4')
landsat_nir = landsat.select('SR_B5')

# 5. Pure Folium Map Generation
print("Generating Physics Map...")

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

# Define visual parameters for raw surface reflectance
# Scale is roughly 7000 (low reflection) to 15000 (high reflection)
band_vis = {'min': 7000, 'max': 15000, 'palette': ['black', 'gray', 'white']}

# Add the isolated bands to the map
m.add_ee_layer(landsat_red, band_vis, "Landsat RED Band (Absorption)")
m.add_ee_layer(landsat_nir, band_vis, "Landsat NIR Band (Reflection)")

# Add the vector boundary
folium.GeoJson(
    gdf,
    name="Watershed Boundary",
    style_function=lambda x: {'fillColor': '#00000000', 'color': 'blue', 'weight': 3}
).add_to(m)

m.add_child(folium.LayerControl())

# 6. Export to HTML
output_html = BASE_DIR / "data" / "band_physics_map.html"
m.save(str(output_html))

print(f"\nSUCCESS: Map saved to {output_html}")
print("Open the HTML file. Toggle between the Red and NIR layers and look at the exact same fields.")