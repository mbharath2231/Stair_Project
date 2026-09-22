# STAIR: Spatio-Temporal Adaptive Image Restoration & Fusion

This repository provides a modular, production-grade Python framework implementing the **STAIR** (Spatio-Temporal Adaptive Image Restoration) methodology and spatiotemporal data fusion between **Landsat 8** (30m high spatial resolution, 16-day revisit) and **MODIS** (250m coarse spatial resolution, daily revisit) using **Google Earth Engine (GEE)**.

The framework is configured for monitoring agricultural dynamics in the **Raisin River Watershed** and can be extended to any Area of Interest (AOI).

---

## Key Features

1. **STAIR Section 2.2.1: Temporal Linear Imputation**
   - Fits pixel-wise linear regression models ($L(t) = a \cdot t + b$) over clear historical observations.
   - Computes evaluation metrics ($R^2$ correlation, valid point counts, slope, intercept).
   - Fills cloud holes and sensor artifacts on broken target dates.

2. **STAIR Section 2.2.2: Spatial & Adaptive Correction**
   - **Global Correction**: Computes global gap offset $\Delta_{\text{global}}$ from a clear reference scene ($t_1$).
   - **Adaptive K-Means Correction**: Unsupervised Weka K-Means clustering segments the landscape, deriving cluster-specific offsets $\Delta_{\text{adap}}(k)$ to account for land cover variation.
   - **Edge Blending**: Focal mean smoothing on gap boundaries eliminates artificial stitching seams.

3. **Spatiotemporal Data Fusion (Landsat + MODIS)**
   - Uses a concurrent clear day ("Golden Day") to learn spatial conversion rules via moving-window linear regression ($15\times 15$ kernel) between Landsat and MODIS Red/NIR surface reflectance.
   - Synthesizes 30m resolution synthetic Red, NIR, and NDVI for dates when Landsat is unavailable or obscured.

4. **Agricultural Sampling & Diagnostics**
   - Samples Corn and Soybean crop pixels using the USDA Cropland Data Layer (CDL).
   - Generates 5km localized chips and multi-layer diagnostic fill trackers (Original vs. Temporal vs. Spatial).

5. **Rich Visualization & Local Plotting**
   - Generates interactive Folium HTML maps with dynamic GEE tile layers (True Color, Red/NIR physics, Cloud before/after).
   - High-resolution local GeoTIFF inspection and plotting with `rasterio` and `matplotlib`.

---

## Directory Structure

```
Stair_Project/
├── .gitignore                   # Comprehensive ignores for OS, virtual environments, and outputs
├── requirements.txt             # Pinned Python dependencies (UTF-8)
├── config.py                    # Central configuration (Earth Engine project ID, default paths, parameters)
├── data/
│   ├── raisin_outline.*         # Watershed boundary shapefile
│   └── outputs/                 # Export directory for plots and maps (git-ignored)
│       └── maps/                # Folium HTML maps
├── stair/                       # Core STAIR Python Library
│   ├── __init__.py              # Library exports
│   ├── core.py                  # Earth Engine init, shapefile loader, cloud masking, collection builders
│   ├── temporal.py              # Section 2.2.1 temporal regression and imputation
│   ├── adaptive.py              # Section 2.2.2 global and K-Means adaptive correction
│   ├── fusion.py                # Landsat-MODIS spatiotemporal fusion and NDVI synthesis
│   ├── diagnostics.py           # CDL crop sampling, 5km chips, staircase fill tracker
│   └── visualization.py         # Folium HTML map generator and rasterio plotting
└── scripts/                     # Unified CLI Entrypoints
    ├── run_temporal.py          # Run temporal imputation (Section 2.2.1)
    ├── run_adaptive.py          # Run adaptive correction (Section 2.2.2: 5km or full watershed)
    ├── run_fusion.py            # Run Landsat-MODIS fusion for missing day synthesis
    ├── run_time_series.py       # Multi-month growing season time-series batch export
    └── run_visualization.py     # Interactive Folium maps and local GeoTIFF plots
```

---

## Setup Instructions

### 1. Environment Installation
Ensure Python 3.10+ is installed. Activate your environment and install dependencies:

```bash
python3 -m venv crop_env
source crop_env/bin/activate
pip install -r requirements.txt
```

### 2. Earth Engine Authentication
Authenticate with Google Earth Engine and set your registered Cloud Project ID:

```bash
earthengine authenticate
```

You can set your project ID in `config.py` or export it as an environment variable:
```bash
export EE_PROJECT_ID="your-project-id"
```

---

## Usage Guide

### 1. Temporal Linear Imputation (Section 2.2.1)
Fills cloud holes on a target date using historical linear regression:
```bash
python scripts/run_temporal.py --target-date 2021-07-19 --start-date 2021-05-15 --end-date 2021-08-15
```

### 2. Adaptive Correction & K-Means (Section 2.2.2)
Runs full STAIR multi-stage restoration (Raw → Temporal → Global → K-Means → Edge Blending):
```bash
# Localized 5km farm chip demo (30m resolution):
python scripts/run_adaptive.py --mode 5km --target-date 2021-07-03 --reference-date 2021-06-17

# Full watershed execution:
python scripts/run_adaptive.py --mode full_watershed --target-date 2021-07-19 --reference-date 2021-06-17
```

### 3. Landsat-MODIS Spatiotemporal Fusion
Learns spatial rules on a clear golden day and synthesizes high-res 30m NDVI for a missing day:
```bash
python scripts/run_fusion.py --prediction-date 2021-06-22 --golden-date 2021-06-17

# With Red-band physical truth validation against real Landsat:
python scripts/run_fusion.py --prediction-date 2021-07-03 --golden-date 2021-06-17 --validate-red
```

### 4. Multi-Date Time Series Generation
Export an entire 6-month growing season sequence or crop-specific time lapse:
```bash
# Growing season NDVI synthesis (every 8 days):
python scripts/run_time_series.py --mode fusion_season --start-date 2021-05-01 --end-date 2021-10-31

# Targeted 5km crop chip for every season overpass:
python scripts/run_time_series.py --mode crop_chip --start-date 2021-04-01 --end-date 2021-10-31
```

### 5. Interactive Visualization & Plotting
Generate interactive HTML maps or render local GeoTIFF files:
```bash
# Plot downloaded GeoTIFF matrix:
python scripts/run_visualization.py --mode plot_tiff

# Generate all interactive Folium HTML maps:
python scripts/run_visualization.py --mode all
```

---

## Theoretical Background (Luo et al., 2018)

- **Temporal Imputation**:
  $$L_{\text{linear}}(p_g, t_0) = a \cdot t_0 + b$$
- **Global Spatial Correction**:
  $$\Delta_{\text{global}} = \bar{L}(t_1)_{gap} - \bar{L}(t_1)_{valid}$$
  $$L_{\text{global}} = L_{\text{linear}} + \Delta_{\text{global}}$$
- **Adaptive K-Means Correction**:
  $$\Delta_{\text{adap}}(k) = \bar{L}_{Cg, t_1}(k) - \bar{L}_{Cf, t_1}(k)$$
  $$L_{\text{adap}} = L_{\text{linear}} + \Delta_{\text{adap}}$$
- **Landsat-MODIS Spatial Regression**:
  $$\text{Landsat}_{\text{band}} = \alpha \cdot \text{MODIS}_{\text{band}} + \beta$$
