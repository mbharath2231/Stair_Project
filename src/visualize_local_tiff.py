import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# 1. Resolve the path to your downloaded file
BASE_DIR = Path(__file__).resolve().parent.parent
# UPDATE THIS EXACT FILENAME TO MATCH WHAT YOU DOWNLOADED
tif_path = BASE_DIR / "data" / "synthetic_ndvi_20210622.tif" 

if not tif_path.exists():
    raise FileNotFoundError(f"CRITICAL: Put the .tif file in {tif_path}")

# 2. Open the GeoTIFF matrix
print(f"Loading matrix from: {tif_path.name}")
with rasterio.open(tif_path) as src:
    matrix = src.read(1) # Read the first (and only) band
    
    # 3. Mask out the empty space (NoData values)
    # Earth Engine pads the outside of your watershed with massive negative numbers or zeros. We hide them.
    matrix_masked = np.ma.masked_where(matrix < -1.0, matrix)

    # 4. Plot the Matrix
    print("Rendering plot...")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # We use a Red-Yellow-Green colormap. 
    # Red = water/dirt, Green = heavy crop cover
    img = ax.imshow(matrix_masked, cmap='RdYlGn', vmin=0.0, vmax=0.9)
    
    plt.colorbar(img, ax=ax, shrink=0.7, label='NDVI Value')
    ax.set_title("STAIR Synthetic NDVI Matrix\n(June 22, 2021)")
    ax.axis('off') # Hide the raw matrix coordinate axes
    
    plt.show()