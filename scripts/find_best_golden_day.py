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
    
    # Get all images
    img_list = col.toList(col.size())
    size = img_list.size().getInfo()
    
    best_date = None
    lowest_cloud = 100
    
    print("Evaluating all Landsat passes for Clayton AOI:")
    for i in range(size):
        img = ee.Image(img_list.get(i))
        date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        
        qa = img.select('QA_PIXEL')
        cloud_mask = qa.bitwiseAnd(1 << 3).neq(0).Or(qa.bitwiseAnd(1 << 4).neq(0))
        
        stats = cloud_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        )
        aoi_clouds = stats.get('QA_PIXEL').getInfo() * 100
        
        print(f"{date}: {aoi_clouds:.2f}% clouds")
        
        if aoi_clouds < lowest_cloud:
            lowest_cloud = aoi_clouds
            best_date = date
            
    print(f"\nBest Golden Day: {best_date} with {lowest_cloud:.2f}% clouds")

if __name__ == "__main__":
    main()
