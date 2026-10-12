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
    
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]
    export_folder = "STAIR_MODIS_Aug1_10"
    tasks = []
    
    # Loop from August 1st to August 10th, 2021
    for day in range(1, 11):
        target_date = f"2021-08-{day:02d}"
        next_day = f"2021-08-{day+1:02d}"
        print(f"\nProcessing MODIS for {target_date}...")
        
        # 1. Pull RAW
        modis_raw_col = get_modis_collection(region, target_date, next_day, apply_mask=False)
        raw_image = modis_raw_col.first().select(bands_to_process).clip(region)
        
        # 2. Pull MASKED
        modis_masked_col = get_modis_collection(region, target_date, next_day, apply_mask=True)
        masked_image = modis_masked_col.first().select(bands_to_process).clip(region)
        
        # 3. Export Raw
        raw_desc = f"MODIS_0_Raw_{target_date.replace('-', '')}_Proj"
        task_raw = ee.batch.Export.image.toDrive(
            image=raw_image, description=raw_desc, folder=export_folder, fileNamePrefix=raw_desc, scale=500.0, crs='EPSG:4326', region=region, maxPixels=1e10
        )
        task_raw.start()
        tasks.append(task_raw)
        
        # 4. Export Masked
        masked_desc = f"MODIS_1_Masked_{target_date.replace('-', '')}_Proj"
        task_masked = ee.batch.Export.image.toDrive(
            image=masked_image, description=masked_desc, folder=export_folder, fileNamePrefix=masked_desc, scale=500.0, crs='EPSG:4326', region=region, maxPixels=1e10
        )
        task_masked.start()
        tasks.append(task_masked)
        
        time.sleep(1)

    print(f"\nSubmitted {len(tasks)} tasks! Track them at: https://code.earthengine.google.com/tasks")

if __name__ == "__main__":
    main()
