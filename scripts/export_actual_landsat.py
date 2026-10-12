import ee
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stair.core import init_earth_engine, get_landsat_collection
from stair.diagnostics import create_chip_bounds

def main():
    init_earth_engine()
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    
    target_date = "2021-06-22"
    landsat_col = get_landsat_collection(region, target_date, "2021-06-23", apply_mask=True)
    actual_landsat = landsat_col.first().select(["SR_B5", "SR_B4", "SR_B3"])
    
    desc = f"Actual_Landsat_30m_{target_date.replace('-', '')}"
    task = ee.batch.Export.image.toDrive(
        image=actual_landsat, description=desc, folder="STAIR_Fusion_Test", fileNamePrefix=desc, scale=30.0, crs='EPSG:32616', region=region, maxPixels=1e10
    )
    task.start()
    print(f"Submitted task: {desc}")

if __name__ == "__main__":
    main()
