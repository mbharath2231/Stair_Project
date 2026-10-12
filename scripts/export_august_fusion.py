import ee
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stair.core import init_earth_engine, get_landsat_collection, get_modis_collection
from stair.diagnostics import create_chip_bounds
from stair.fusion import fuse_date

def main():
    print("Initializing Google Earth Engine...")
    init_earth_engine()
    
    # Define Clayton, MI AOI (7km x 7km)
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]
    
    # Base Golden Day: July 26, 2021 (100% cloud-free baseline)
    golden_date = "2021-07-26"
    print(f"\nLoading Golden Reference Day: {golden_date}...")
    golden_landsat = get_landsat_collection(
        region, golden_date, "2021-07-27", apply_mask=True
    ).first().select(bands_to_process)
    
    golden_modis = get_modis_collection(
        region, golden_date, "2021-07-27", apply_mask=True
    ).first().select(bands_to_process)
    
    export_folder = "STAIR_August_Test"
    print(f"Target Google Drive Folder: '{export_folder}'")
    
    # -------------------------------------------------------------
    # 1. Export Predicted Landsat (STAIR Fusion) for August 1 to 4
    # -------------------------------------------------------------
    target_dates = ["2021-08-01", "2021-08-02", "2021-08-03", "2021-08-04"]
    
    for date_str in target_dates:
        clean_date = date_str.replace("-", "")
        task_desc = f"STAIR_Predicted_Landsat_30m_{clean_date}"
        print(f"\n[STAIR FUSION] Building 30m prediction for: {date_str}...")
        
        predicted_landsat = fuse_date(
            prediction_date=date_str,
            golden_date=golden_date,
            golden_landsat=golden_landsat,
            golden_modis=golden_modis,
            aoi=region,
            bands=bands_to_process
        )
        
        task = ee.batch.Export.image.toDrive(
            image=predicted_landsat,
            description=task_desc,
            folder=export_folder,
            fileNamePrefix=task_desc,
            scale=30.0,
            crs='EPSG:32616',
            region=region,
            maxPixels=1e10
        )
        task.start()
        print(f"  -> Submitted Export Task: {task_desc} (ID: {task.id})")
        
    # -------------------------------------------------------------
    # 2. Export Actual Ground Truth Landsat for August 4
    # -------------------------------------------------------------
    print(f"\n[GROUND TRUTH] Fetching Actual Landsat acquisition for 2021-08-04...")
    actual_col = get_landsat_collection(region, "2021-08-04", "2021-08-05", apply_mask=False)
    actual_landsat = actual_col.first().select(bands_to_process)
    
    actual_task_desc = "Actual_Landsat_30m_20210804"
    actual_task = ee.batch.Export.image.toDrive(
        image=actual_landsat,
        description=actual_task_desc,
        folder=export_folder,
        fileNamePrefix=actual_task_desc,
        scale=30.0,
        crs='EPSG:32616',
        region=region,
        maxPixels=1e10
    )
    actual_task.start()
    print(f"  -> Submitted Export Task: {actual_task_desc} (ID: {actual_task.id})")
    
    print("\n" + "=" * 60)
    print("ALL 5 TASKS SUBMITTED TO GOOGLE EARTH ENGINE SUCCESSFULLY!")
    print(f"Check your Google Drive folder: '{export_folder}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
