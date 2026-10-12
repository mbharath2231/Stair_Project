import ee
import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stair.core import init_earth_engine, get_modis_collection
from stair.diagnostics import create_chip_bounds

def main():
    init_earth_engine()
    
    print("Setting AOI to Clayton, MI (7km chip)...")
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    
    # We will look at August 4, 2021 (a clear summer day)
    target_date = "2021-08-04"
    next_day = "2021-08-05"
    
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]
    
    # 1. Pull the RAW MODIS collection (no mask)
    print(f"\nPulling RAW MODIS for {target_date}...")
    modis_raw_col = get_modis_collection(region, target_date, next_day, apply_mask=False)
    raw_image = modis_raw_col.first().select(bands_to_process).clip(region)
    
    # 2. Pull the MASKED MODIS collection
    print(f"Pulling MASKED MODIS for {target_date}...")
    modis_masked_col = get_modis_collection(region, target_date, next_day, apply_mask=True)
    masked_image = modis_masked_col.first().select(bands_to_process).clip(region)
    
    # 3. Export both to Google Drive
    export_folder = "STAIR_Fusion_Test"
    
    # Export Raw
    raw_desc = f"MODIS_0_Raw_{target_date.replace('-', '')}"
    print(f"Exporting: {raw_desc}")
    task_raw = ee.batch.Export.image.toDrive(
        image=raw_image, description=raw_desc, folder=export_folder, fileNamePrefix=raw_desc, scale=500.0, region=region, maxPixels=1e10
    )
    task_raw.start()
    
    # Export Masked
    masked_desc = f"MODIS_1_Masked_{target_date.replace('-', '')}"
    print(f"Exporting: {masked_desc}")
    task_masked = ee.batch.Export.image.toDrive(
        image=masked_image, description=masked_desc, folder=export_folder, fileNamePrefix=masked_desc, scale=500.0, region=region, maxPixels=1e10
    )
    task_masked.start()
    
    print("\nSubmitted both tasks! Track them at: https://code.earthengine.google.com/tasks")

if __name__ == "__main__":
    main()
