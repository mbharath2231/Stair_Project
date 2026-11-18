# Clayton cloud filling validation with probability densities

The experiment hides known Landsat pixels on the existing 167 × 167
Clayton grid. It uses March 13 and April 14, 2024, the spring fusion
benchmark dates, and July 22, 2022, the previous restoration benchmark.
This experiment evaluates Landsat gap filling before fusion.

For each date, run the temporal, global, and adaptive stages on the same
cloud-shaped mask, then repeat with diagonal stripes. Fit spatial
corrections using the visible target pixels and a different date. The
target is excluded from temporal references. The closest reference
within 21 days with at least 95% QA-clear coverage is selected by date and
coverage only. The selected references are February 26, April 13, and July
21, respectively. Each was 100% QA-clear on this grid.

Score only originally QA-clear withheld pixels that have finite values
in all methods. All three methods filled all 6,385 withheld reflectance
pixels per cloud test and all 8,489 per stripe test. April has 6,383 and
8,488 eligible NDVI pixels in the cloud and stripe masks, respectively,
because some withheld reference reflectance values fail the NDVI validity
rule. No stage changed any observed reflectance outside the holdout.

## How to read the density plots

The curves are area-normalized histograms with identical bins for all
methods and dates shown in a figure. Compare the curves for restored and
observed values to assess the brightness distribution and its spread.
Wasserstein-1 quantifies the difference between the two empirical
distributions in the original reflectance or NDVI units. Pixel locations
do not enter the Wasserstein calculation. Paired RMSE, MAE, bias, and
error density use the original pixel locations. Report coverage with
both comparisons.

## Cloud mask results

Each entry lists Wasserstein-1 followed by paired RMSE. Smaller values
indicate closer agreement for the respective measure.

| Target | Variable | Temporal | Global | Adaptive |
| --- | --- | ---: | ---: | ---: |
| March 13, 2024 | NIR | 0.00905 / 0.02889 | 0.01195 / 0.02993 | 0.01342 / 0.03066 |
| March 13, 2024 | NDVI | 0.00365 / 0.03106 | 0.01037 / 0.03286 | 0.01113 / 0.03356 |
| April 14, 2024 | NIR | 0.02787 / 0.04397 | 0.00869 / 0.03507 | 0.00899 / 0.03478 |
| April 14, 2024 | NDVI | 0.02030 / 0.06802 | 0.02592 / 0.06749 | 0.01260 / 0.06548 |
| July 22, 2022 | NIR | 0.01872 / 0.03377 | 0.00264 / 0.02821 | 0.00378 / 0.02808 |
| July 22, 2022 | NDVI | 0.01530 / 0.07496 | 0.01519 / 0.07267 | 0.01337 / 0.07244 |

On July 22, global correction brought the NIR distribution closest to
the withheld values. Adaptive correction produced the lowest NIR RMSE on
those pixels. On March 13, temporal interpolation had the lowest NIR
RMSE under the cloud mask and adaptive correction had the lowest NIR
RMSE under the stripe mask (0.03045; temporal was 0.03217). The ranking
depends on the target date, gap geometry, and evaluation measure.

This is a controlled test on one chip and three dates. For inference
across dates or areas, use independent dates and spatial blocks as
replicates. A simulated mask allows comparison with known values.
Naturally cloudy locations lack the same-day QA-clear reference needed
for this experiment.

## Reproduce

Run `python scripts/run_cloud_density_validation.py` with the project
environment. Earth Engine authentication is required for the first run.
A rerun can reuse the observation cache with `--cache` and write to a
fresh directory with `--output`. By default, figures and `run.json` are
in `outputs/fusion_validation/cloud_density_clayton/`. The manifest
records reference dates, scene IDs, coverage, and all metrics for
NIR, red, green, and NDVI.

## Review the five stages in QGIS

Export the same cloud-shaped holdout as five GeoTIFFs per date, then
build the QGIS project with the Python bundled with QGIS.

```sh
python scripts/run_cloud_density_validation.py --patterns cloud --export-stages --output outputs/fusion_validation/cloud_fill_qgis_clayton
QT_QPA_PLATFORM=offscreen /Applications/QGIS.app/Contents/MacOS/python scripts/build_cloud_fill_qgis.py outputs/fusion_validation/cloud_fill_qgis_clayton
```

Open `clayton_cloud_fill.qgz` from the output folder. Use **Project →
Layouts → All dates | five stages** for the overview and the dated layouts
for a closer view of the mask. The original GeoTIFF stores the QA-clear
observations used as validation truth. The five stages have the same
30 m grid and a shared NIR/red/green display stretch. The QGIS project
uses relative paths, so keep it with the `cloud/` folder when transferring.

## Review NDVI in QGIS

Derive NDVI from the NIR and red surface reflectance of each exported
stage, using the Landsat formula `(NIR − red) / (NIR + red)`. The export
applies the same validity rule as the density evaluation. It also writes
three signed error maps per date. Each shows restored minus original NDVI
only where all three fills and the reference have valid NDVI inside the
simulated gap.

```sh
python scripts/export_cloud_fill_ndvi.py outputs/fusion_validation/cloud_fill_qgis_clayton outputs/fusion_validation/cloud_fill_ndvi_clayton
QT_QPA_PLATFORM=offscreen /Applications/QGIS.app/Contents/MacOS/python scripts/build_cloud_fill_qgis.py outputs/fusion_validation/cloud_fill_ndvi_clayton
```

Open `ndvi_clayton_cloud_fill.qgz` and choose **All dates | five stages**
or **All dates | NDVI errors** in **Project → Layouts**. The first uses a
shared NDVI scale from −0.2 to 1.0. In the error comparison, blue indicates
underestimation, orange indicates overestimation, and light gray marks
pixels outside the common scoring area. The fixed display limits do not
change the stored GeoTIFF values.

Method references:

- [NIST handbook on probability density normalization](https://www.itl.nist.gov/div898/handbook/eda/section3/eda33e.htm)
- [Wasserstein-1 distance definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)
- [Luo et al. (2018) STAIR](https://doi.org/10.1016/j.rse.2018.04.042)
- [USGS Landsat NDVI formula](https://www.usgs.gov/landsat-missions/landsat-normalized-difference-vegetation-index)
