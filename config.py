"""
Centralized Configuration for STAIR Project
Contains default parameters, paths, Earth Engine project identifiers, and satellite band configurations.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
MAPS_OUTPUT_DIR = OUTPUT_DIR / "maps"
SHAPEFILE_PATH = DATA_DIR / "raisin_outline.shp"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAPS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Google Earth Engine Configuration
# Overridable via environment variable EE_PROJECT_ID
EE_PROJECT_ID = os.getenv("EE_PROJECT_ID", "stair-499915")

# Satellite Collections
LANDSAT8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
MODIS_COLLECTION = "MODIS/061/MOD09GQ"
CDL_COLLECTION = "USDA/NASS/CDL/2021"

# Satellite Band Mappings
LANDSAT_BANDS = {
    "red": "SR_B4",
    "nir": "SR_B5",
    "qa": "QA_PIXEL"
}

MODIS_BANDS = {
    "red": "sur_refl_b01",
    "nir": "sur_refl_b02"
}

# STAIR Algorithm Default Parameters
DEFAULT_SCALE = 30.0  # Landsat standard 30m resolution
DEFAULT_KERNEL_RADIUS = 15  # Moving window radius in pixels for spatial regression
DEFAULT_KMEANS_CLUSTERS = 4  # Number of clusters for adaptive correction
DEFAULT_CLOUD_THRESHOLD = 20.0  # Max cloud cover percentage for training scenes

# Default Dates (Raisin Watershed study period)
DEFAULT_TARGET_DATE = "2021-07-19"      # Date with heavy cloud cover to impute
DEFAULT_REFERENCE_DATE = "2021-06-17"   # Clear "Golden Day" reference
DEFAULT_PREDICTION_DATE = "2021-06-22"  # Missing Landsat date for MODIS fusion
DEFAULT_SEASON_START = "2021-05-01"     # Agricultural growing season start
DEFAULT_SEASON_END = "2021-10-31"       # Agricultural growing season end
