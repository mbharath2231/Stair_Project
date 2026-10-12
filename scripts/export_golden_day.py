import ee
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stair.core import init_earth_engine, get_landsat_collection, get_modis_collection
from stair.diagnostics import create_chip_bounds

def main():
    init_earth_engine()
    
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]
    
    golden_date = "2021-07-26"
    export_folder = "STAIR_Golden_Day_Validation"
    
    print(f"Exporting RAW Landsat for {golden_date}...")
    landsat_col = get_landsat_collection(region, golden_date, "2021-07-27", apply_mask=False)
    raw_landsat = landsat_col.first().select(bands_to_process)
    
    task_l = ee.batch.Export.image.toDrive(
        image=raw_landsat,
        description=f"Actual_Landsat_30m_{golden_date.replace('-', '')}",
        folder=export_folder,
        scale=30.0,
        crs='EPSG:32616',
        region=region,
        maxPixels=1e10
    )
    task_l.start()
    
    print(f"Exporting RAW MODIS for {golden_date}...")
    modis_col = get_modis_collection(region, golden_date, "2021-07-27", apply_mask=False)
    raw_modis = modis_col.first().select(bands_to_process)
    
    task_m = ee.batch.Export.image.toDrive(
        image=raw_modis,
        description=f"Actual_MODIS_500m_{golden_date.replace('-', '')}",
        folder=export_folder,
        scale=500.0,
        crs='EPSG:4326', # MODIS native CRS is sinusoidal, but we export as WGS84 for easy viewing
        region=region,
        maxPixels=1e10
    )
    task_m.start()
    
    print("\nTasks submitted! Check Earth Engine Tasks tab.")

if __name__ == "__main__":
    main()
