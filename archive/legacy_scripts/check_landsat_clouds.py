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
    
    date = "2021-08-04"
    col = get_landsat_collection(region, date, "2021-08-05", apply_mask=False)
    img = col.first()
    
    if img:
        cloud_cover = img.get('CLOUD_COVER').getInfo()
        print(f"August 4 Landsat Scene Cloud Cover (Whole Scene): {cloud_cover}%")
        
        # Calculate cloud cover specifically in our 7km AOI
        qa = img.select('QA_PIXEL')
        # Bit 3 is cloud, Bit 4 is cloud shadow
        cloud_mask = qa.bitwiseAnd(1 << 3).neq(0).Or(qa.bitwiseAnd(1 << 4).neq(0))
        
        stats = cloud_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        )
        aoi_clouds = stats.get('QA_PIXEL').getInfo() * 100
        print(f"August 4 Landsat AOI Cloud Cover (7km Chip): {aoi_clouds:.2f}%")
    else:
        print("No image found for August 4.")

if __name__ == "__main__":
    main()
