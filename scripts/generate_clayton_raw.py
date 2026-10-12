import ee
import time
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stair.core import init_earth_engine, get_landsat_collection
from stair.diagnostics import create_chip_bounds
import config

def main():
    init_earth_engine()
    
    print("Setting AOI to Clayton, MI (7km chip)...")
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    scale = 30.0
    
    bands_to_process = ["SR_B5", "SR_B4", "SR_B3"]
    
    start_date = "2021-03-01"
    end_date = "2021-10-31"
    
    # Get all Landsat 8 images WITHOUT cloud masking (to see actual clouds)
    print(f"\nPulling raw, unmasked historical time series from {start_date} to {end_date}...")
    raw_col = get_landsat_collection(region, start_date, end_date, max_cloud_cover=100.0, apply_mask=False)
    
    date_list = raw_col.aggregate_array('system:time_start').map(lambda d: ee.Date(d).format('YYYY-MM-dd')).distinct().getInfo()
    print(f"Found {len(date_list)} satellite passes over Clayton in this 7-month period.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_folder = f"Raw_Landsat_Clayton_{timestamp}"
    print(f"All RAW images will be exported to Google Drive folder: '{export_folder}'")
    
    tasks = []
    for target_date in date_list:
        print(f"Processing raw date: {target_date}")
        
        target_img = raw_col.filterDate(target_date, ee.Date(target_date).advance(1, "day")).first()
        raw_fcc = target_img.select(bands_to_process).clip(region)
        
        desc = f"Raw_Clayton_{target_date.replace('-', '')}"
        
        task = ee.batch.Export.image.toDrive(
            image=raw_fcc,
            description=desc,
            folder=export_folder,
            fileNamePrefix=desc,
            scale=scale,
            region=region,
            maxPixels=1e10
        )
        task.start()
        tasks.append(task)
        time.sleep(1)

    print(f"\nSubmitted {len(tasks)} tasks! Track them at https://code.earthengine.google.com/tasks")

if __name__ == "__main__":
    main()
