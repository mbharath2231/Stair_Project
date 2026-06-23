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

# 2. Load the AOI
BASE_DIR = Path(__file__).resolve().parent.parent
shapefile_path = BASE_DIR / "data" / "raisin_outline.shp"

gdf = gpd.read_file(shapefile_path)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

geojson_geom = gdf.geometry.iloc[0].__geo_interface__
aoi = ee.Geometry(geojson_geom)

# 3. Intentionally Find a Ruined (Cloudy) Day
print("Searching for a highly cloudy day...")
cloudy_landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(aoi) \
    .filterDate('2021-05-01', '2021-07-01') \
    .filter(ee.Filter.gt('CLOUD_COVER', 60)) \
    .first()

cloudy_date_ms = cloudy_landsat.get('system:time_start').getInfo()
cloudy_date = ee.Date(cloudy_date_ms).format('YYYY-MM-dd').getInfo()
print(f"Found Cloudy Landsat Image on: {cloudy_date}")

# 4. Pure Folium Map Generation
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

landsat_true_color = {'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min': 7000, 'max': 12000}

# Add the ruined layer
m.add_ee_layer(cloudy_landsat, landsat_true_color, f"Landsat True Color ({cloudy_date})")

# Add the boundary
folium.GeoJson(
    gdf,
    name="Watershed Boundary",
    style_function=lambda x: {'fillColor': '#00000000', 'color': 'red', 'weight': 3}
).add_to(m)

m.add_child(folium.LayerControl())

# 5. Export to HTML
output_html = BASE_DIR / "Data" / "cloudy_day_map.html"
m.save(str(output_html))

print(f"\nSUCCESS: Map saved to {output_html}")