import geopandas as gpd
import ee
from pathlib import Path
import time
import sys
import traceback

print("--- STARTING STAIR 2.2.2 (SMART TRAINING DATA) ---")

try:
    # 1. Initialize Earth Engine
    project_id = "stair-499915" 
    try:
        ee.Initialize(project=project_id)
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project=project_id)

    print("Cloud Engine Online.")

    # 2. Load the Boundary
    BASE_DIR = Path(__file__).resolve().parent.parent
    shapefile_path = BASE_DIR / "data" / "raisin_outline.shp"

    if not shapefile_path.exists():
        raise FileNotFoundError(f"Cannot find shapefile at {shapefile_path}")
        
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    aoi = ee.Geometry(gdf.geometry.iloc[0].__geo_interface__)

    # 3. Setup Dates (4 weeks before and 4 weeks after the target)
    target_date_t0 = '2021-07-19'
    target_date_ee_t0 = ee.Date(target_date_t0)
    
    # The clear reference day for calculating the spatial delta
    reference_date_t1 = '2021-06-17' 
    
    # 8-week total window to ensure we catch enough clear Landsat passes (which only happen every 16 days)
    time_window_start = '2021-06-15'
    time_window_end = '2021-08-15'

    # 4. Define Cloud Mask
    def mask_clouds(image):
        qa = image.select('QA_PIXEL')
        cloud_shadow_bitmask = (1 << 3)
        clouds_bitmask = (1 << 4)
        mask = qa.bitwiseAnd(cloud_shadow_bitmask).eq(0) \
                 .And(qa.bitwiseAnd(clouds_bitmask).eq(0))
        return image.updateMask(mask)

    # --- STAGE 1: THE TARGET IMAGE (Allowed to have clouds) ---
    print(f"Pulling Target Image for {target_date_t0}...")
    target_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi)
    image_t0_raw = target_collection.filterDate(target_date_ee_t0, target_date_ee_t0.advance(1, 'day')).first()
    
    stage1_broken = mask_clouds(image_t0_raw).select('SR_B5')
    mask_t0 = stage1_broken.mask() 

    # --- THE SMART TRAINING DATASET ---
    print(f"Building Training Model (Rejecting scenes with > 20% cloud cover)...")
    # CRITICAL CHANGE: We apply a metadata filter so the timeline math ONLY uses good, clear days
    training_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(aoi) \
        .filterDate(time_window_start, time_window_end) \
        .filter(ee.Filter.lt('CLOUD_COVER', 20)) \
        .map(mask_clouds)

    # --- STAGE 2: TEMPORAL INTERPOLATION ---
    def add_time_band(img):
        t = ee.Image(img.date().millis()).divide(1000 * 60 * 60 * 24).toFloat().rename('t')
        return img.select('SR_B5').addBands(t)

    reg_col = training_collection.map(add_time_band).select(['t', 'SR_B5'])
    temporal_regression = reg_col.reduce(ee.Reducer.linearFit())

    slope = temporal_regression.select('scale')
    intercept = temporal_regression.select('offset')
    t0_millis = ee.Image(target_date_ee_t0.millis()).divide(1000 * 60 * 60 * 24).toFloat()
    
    L_linear = t0_millis.multiply(slope).add(intercept).rename('L_linear')
    stage2_temporal = stage1_broken.unmask(L_linear).clip(aoi)

    # --- STAGE 3a: GLOBAL CORRECTION ---
    print("Executing Stage 3a: Global Correction...")
    image_t1_raw = target_collection.filterDate(ee.Date(reference_date_t1), ee.Date(reference_date_t1).advance(1, 'day')).first().select('SR_B5')
    
    global_gap_dict = image_t1_raw.updateMask(mask_t0.eq(0)).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=120, bestEffort=True, maxPixels=1e10
    )
    global_valid_dict = image_t1_raw.updateMask(mask_t0.eq(1)).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=120, bestEffort=True, maxPixels=1e10
    )

    mean_gap_global = ee.Number(global_gap_dict.get('SR_B5', 0))
    mean_valid_global = ee.Number(global_valid_dict.get('SR_B5', 0))
    delta_global = mean_gap_global.subtract(mean_valid_global)
    
    L_global_gaps = L_linear.add(ee.Image.constant(delta_global))
    stage3a_global = stage1_broken.unmask(L_global_gaps).clip(aoi)

    # --- STAGE 3b: ADAPTIVE CORRECTION (K-MEANS) ---
    print("Executing Stage 3b: Adaptive Correction...")
    training_data = image_t1_raw.sample(region=aoi, scale=120, numPixels=5000, tileScale=8)
    clusterer = ee.Clusterer.wekaKMeans(4).train(training_data)
    segments = image_t1_raw.cluster(clusterer).rename('seg')

    def get_segment_means(value_img, mask_img):
        stack = value_img.rename('val').updateMask(mask_img).addBands(segments)
        reduction = stack.reduceRegion(
            reducer=ee.Reducer.mean().group(groupField=1, groupName='seg'),
            geometry=aoi, scale=120, bestEffort=True, maxPixels=1e10, tileScale=8
        )
        groups = ee.List(ee.Dictionary(reduction).get('groups', ee.List([])))
        
        def get_key(g): return ee.Dictionary(g).getNumber('seg')
        # CRITICAL SAFETY: If the mean is missing, force it to 0 so the math doesn't crash
        def get_val(g): return ee.Dictionary(g).getNumber('mean')
        
        keys = ee.List([-1]).cat(groups.map(get_key))
        vals = ee.List([0.0]).cat(groups.map(get_val))
        return segments.remap(keys, vals, 0).rename('mean_val')

    L_Cg_t1_adap = get_segment_means(image_t1_raw, mask_t0.eq(0))
    L_Cf_t1_adap = get_segment_means(image_t1_raw, mask_t0.eq(1))
    
    delta_adap = L_Cg_t1_adap.subtract(L_Cf_t1_adap)
    L_adap_gaps = L_linear.add(delta_adap)
    stage3b_adaptive = stage1_broken.unmask(L_adap_gaps).clip(aoi)

    # --- EXPORT PIPELINE ---
    print("Submitting Export Tasks...")
    folder_name = 'STAIR_Smart_Model'
    
    tasks = [
        ee.batch.Export.image.toDrive(image=stage1_broken.toDouble().clip(aoi), description='Stage1_Broken', folder=folder_name, scale=30, region=aoi, maxPixels=1e10),
        ee.batch.Export.image.toDrive(image=stage2_temporal.toDouble().clip(aoi), description='Stage2_Temporal', folder=folder_name, scale=30, region=aoi, maxPixels=1e10),
        ee.batch.Export.image.toDrive(image=stage3a_global.toDouble().clip(aoi), description='Stage3a_Global', folder=folder_name, scale=30, region=aoi, maxPixels=1e10),
        ee.batch.Export.image.toDrive(image=stage3b_adaptive.toDouble().clip(aoi), description='Stage3b_Adaptive', folder=folder_name, scale=30, region=aoi, maxPixels=1e10),
        ee.batch.Export.image.toDrive(image=segments.clip(aoi), description='Stage4_Clusters', folder=folder_name, scale=30, region=aoi, maxPixels=1e10)
    ]

    for t in tasks: t.start()

    while any(t.active() for t in tasks):
        statuses = [t.status()['state'][:4] for t in tasks] 
        print(f"Tasks: {statuses} ... waiting 15s")
        time.sleep(15)

    print("\n--- PROCESS FINISHED ---")
    for t in tasks:
        if t.status()['state'] == 'FAILED':
            print(f"Error in {t.status()['description']}: {t.status().get('error_message')}")
            
    print(f"Check the '{folder_name}' folder in Google Drive.")

except Exception as e:
    print("\n!!! SCRIPT CRASHED !!!")
    traceback.print_exc()
    sys.exit(1)