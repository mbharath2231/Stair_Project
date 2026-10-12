import ee
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stair.core import init_earth_engine, get_landsat_collection, get_modis_collection
from stair.diagnostics import create_chip_bounds
from stair.fusion import fuse_date

def main():
    init_earth_engine()
    
    print("Setting AOI to Clayton, MI (7km chip)...")
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]
    
    # 1. The Golden Day
    # We scanned all Landsat passes. August 4 was 56% cloudy.
    # July 26 was 0.01% cloudy and is the absolute closest perfect day to August!
    golden_date = "2021-07-26"
    
    print(f"\nPulling Golden Landsat (30m) for {golden_date}...")
    landsat_col = get_landsat_collection(region, golden_date, "2021-07-27", apply_mask=True)
    golden_landsat = landsat_col.first().select(bands_to_process)
    
    print(f"Pulling Golden MODIS (500m) for {golden_date}...")
    modis_col = get_modis_collection(region, golden_date, "2021-07-27", apply_mask=True)
    golden_modis = modis_col.first().select(bands_to_process)
    
    export_folder = "STAIR_August_Predictions"
    
    # 2. Predict August 1 through August 10!
    days_to_predict = [
        "2021-08-01", "2021-08-02", "2021-08-03",
        "2021-08-04", "2021-08-05", "2021-08-06",
        "2021-08-07", "2021-08-08", "2021-08-09", "2021-08-10"
    ]
    
    for target_date in days_to_predict:
        print(f"\nRunning STAIR Fusion to predict Landsat on {target_date}...")
        
        predicted_landsat = fuse_date(
            prediction_date=target_date,
            golden_date=golden_date,
            golden_landsat=golden_landsat,
            golden_modis=golden_modis,
            aoi=region,
            kernel_radius=15 # 15 pixels * 30m = 450m radius moving window
        )
        
        desc = f"Predicted_Landsat_30m_{target_date.replace('-', '')}"
        task = ee.batch.Export.image.toDrive(
            image=predicted_landsat,
            description=desc,
            folder=export_folder,
            fileNamePrefix=desc,
            scale=30.0,
            crs='EPSG:32616', # Force UTM projection for perfect Landsat squares
            region=region,
            maxPixels=1e10
        )
        task.start()
        print(f"Submitted task: {desc}")
        
    print("\nAll 10 days submitted! Track them at: https://code.earthengine.google.com/tasks")

if __name__ == "__main__":
    main()
