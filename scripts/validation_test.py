import ee
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stair.core import init_earth_engine, get_landsat_collection, get_modis_collection
from stair.diagnostics import create_chip_bounds
from stair.fusion import fuse_date

def main():
    init_earth_engine()
    
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]
    
    # 1. The Foundation: July 26 (Our flawless Golden Day)
    golden_date = "2021-07-26"
    print(f"Loading Golden Day: {golden_date}...")
    golden_landsat = get_landsat_collection(region, golden_date, "2021-07-27", apply_mask=True).first().select(bands_to_process)
    golden_modis = get_modis_collection(region, golden_date, "2021-07-27", apply_mask=True).first().select(bands_to_process)
    
    # 2. The Target Date: July 19 (0% clouds!)
    # We will hide the July 19 Landsat, predict it using STAIR, and then export both to compare!
    target_date = "2021-07-19"
    export_folder = "STAIR_Validation_Test"
    
    print(f"\nTarget Date for Validation: {target_date}")
    
    # --- A. Predict the Landsat (STAIR) ---
    print(f"Running STAIR Fusion to predict Landsat on {target_date}...")
    predicted_landsat = fuse_date(
        prediction_date=target_date,
        golden_date=golden_date,
        golden_landsat=golden_landsat,
        golden_modis=golden_modis,
        aoi=region,
        kernel_radius=15
    )
    
    task_pred = ee.batch.Export.image.toDrive(
        image=predicted_landsat,
        description=f"Validation_Predicted_Landsat_30m_{target_date.replace('-', '')}",
        folder=export_folder,
        scale=30.0,
        crs='EPSG:32616',
        region=region,
        maxPixels=1e10
    )
    task_pred.start()
    
    # --- B. Export the ACTUAL Landsat (Ground Truth) ---
    print(f"Exporting ACTUAL Landsat on {target_date} for side-by-side comparison...")
    actual_landsat_col = get_landsat_collection(region, target_date, "2021-07-20", apply_mask=False)
    actual_landsat = actual_landsat_col.first().select(bands_to_process)
    
    task_actual = ee.batch.Export.image.toDrive(
        image=actual_landsat,
        description=f"Validation_Actual_Landsat_30m_{target_date.replace('-', '')}",
        folder=export_folder,
        scale=30.0,
        crs='EPSG:32616',
        region=region,
        maxPixels=1e10
    )
    task_actual.start()
    
    print("\nValidation tasks submitted! Check Earth Engine Tasks tab.")

if __name__ == "__main__":
    main()
