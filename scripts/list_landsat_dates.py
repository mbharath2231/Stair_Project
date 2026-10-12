import ee
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stair.core import init_earth_engine
from stair.diagnostics import create_chip_bounds
from config import LANDSAT8_COLLECTION

def main():
    init_earth_engine()
    target_point = ee.Geometry.Point([-84.2333, 41.8611])
    region = create_chip_bounds(target_point, radius_meters=3500)
    
    col = ee.ImageCollection(LANDSAT8_COLLECTION).filterBounds(region).filterDate("2021-05-01", "2021-10-31")
    dates = col.aggregate_array('system:time_start').map(lambda d: ee.Date(d).format('YYYY-MM-dd')).getInfo()
    print("Available Landsat dates:")
    for d in dates:
        print(d)

if __name__ == "__main__":
    main()
