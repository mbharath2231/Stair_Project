import ee
import sys
from pathlib import Path
import time
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
    
    # 1. The Golden Day (Clear Day)
    golden_date = "2021-06-17"
    
    print(f"\nPulling Golden Landsat (30m) for {golden_date}...")
    landsat_col = get_landsat_collection(region, golden_date, "2021-06-18", apply_mask=True)
    golden_landsat = landsat_col.first().select(bands_to_process)
    
    print(f"Pulling Golden MODIS (500m) for {golden_date}...")
    modis_col = get_modis_collection(region, golden_date, "2021-06-18", apply_mask=True)
    golden_modis = modis_col.first().select(bands_to_process)
    
    # 2. The Target Day (The day we want to predict)
    # We must pick a day where Landsat actually passed over so we can compare the prediction to reality!
    target_date = "2021-06-24"
    print(f"\nRunning STAIR Fusion to predict Landsat on {target_date}...")
    
    predicted_landsat = fuse_date(
        prediction_date=target_date,
        golden_landsat=golden_landsat,
        golden_modis=golden_modis,
        aoi=region,
        kernel_radius=15 # 15 pixels
    )
    
    # 3. Get the Actual Landsat (for comparison)
    actual_col = get_landsat_collection(region, target_date, "2021-06-25", apply_mask=True)
    actual_landsat = actual_col.first().select(bands_to_process)
    
    # 4. Export Both Images
    export_folder = "STAIR_Fusion_Test"
    
    # Export Predicted
    pred_desc = f"Predicted_Landsat_30m_{target_date.replace('-', '')}"
    print(f"\nExporting the Predicted Landsat image to Drive...")
    task1 = ee.batch.Export.image.toDrive(
        image=predicted_landsat, description=pred_desc, folder=export_folder, fileNamePrefix=pred_desc, scale=30.0, crs='EPSG:32616', region=region, maxPixels=1e10
    )
    task1.start()
    
    # Export Actual
    act_desc = f"Actual_Landsat_30m_{target_date.replace('-', '')}"
    print(f"Exporting the Actual Landsat image to Drive...")
    task2 = ee.batch.Export.image.toDrive(
        image=actual_landsat, description=act_desc, folder=export_folder, fileNamePrefix=act_desc, scale=30.0, crs='EPSG:32616', region=region, maxPixels=1e10
    )
    task2.start()
    
    print(f"\nSubmitted 2 tasks for {target_date}!")
    print("Track it at: https://code.earthengine.google.com/tasks")

if __name__ == "__main__":
    main()
