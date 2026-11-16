# River Raisin five-day fusion pilot

Open `raisin_five_day_fusion.qgz` in the generated
`outputs/fusion_validation/raisin_2022-07-22_five_days/` folder. Keep the
entire folder together when copying it to another device. The QGIS project uses
relative paths to the GeoTIFFs and the watershed GeoPackage.

The **Prediction days** group contains July 22–26, 2022, and **Restored anchor
pairs** contains the five available dates selected to bracket the forecast
window. Expand one date and toggle its layers in the map canvas. Under
**Project → Layouts**, use **Five-day forecast coverage**, **2022-07-22 sensor
restoration stages**, and **2022-07-22 withheld Landsat validation**. The
reflectance composites use the same stretch and channel order of NIR, red,
and green. Anchor ages are two bands in days, prior first and later second.

Each available scene has original reflectance, QA-masked reflectance,
temporal interpolation, global correction, and adaptive correction. The
Landsat daily mosaics are scaled to surface reflectance. MODIS MCD43A4.061 is
screened for full BRDF inversion in all three
bands. Its restoration runs on the native sinusoidal grid before bilinear
resampling to the Landsat grid. The NASA grid definition uses a sphere with
radius 6,371,007.181 m. MCD43A4 uses a 16-day retrieval window, so nearby dates
share inputs. [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD43A4)
and [NASA's MODIS grid definition](https://modis-land.gsfc.nasa.gov/GCTP.html).

The 30 m grid covers the watershed polygon in
`data/raisin_outline.shp`. Areas outside the boundary and unresolved pixels
are NoData. Fusion picks the closest supported anchor residual before and
after each pixel's target date and adds the time-weighted residual to the
target MODIS layer. The run uses a common projected grid. It applies no
estimated cross-sensor shift. Sensor restoration shares the maintained
Clayton temporal and spatial functions with adjustments to handle larger
rasters and choose references for the watershed.

July 22 is the only day with a Landsat acquisition over the watershed in the
forecast window. All five days are excluded from Landsat anchor restoration,
spatial reference selection, and prediction. The July 22 Landsat spatial
stages are provided for separate gap-fill inspection. The July 22 QA-clear
Landsat layer is used for scoring fusion against nearest-clear Landsat
interpolation, interpolation using the same anchors as fusion, and restored
MODIS alone. All four methods use common pixels for each metric comparison.

## July 22–26 results

The watershed polygon contains 3,130,003 pixel centers on the 30 m grid.
The generated project reopens with 118 raster layers and the boundary.

| Date | Landsat availability | Valid fusion | Unresolved pixels |
| --- | --- | ---: | ---: |
| July 22 | Available and withheld | 99.45% | 17,256 |
| July 23 | No acquisition | 99.28% | 22,462 |
| July 24 | No acquisition | 99.27% | 22,699 |
| July 25 | No acquisition | 99.30% | 21,983 |
| July 26 | No acquisition | 99.31% | 21,528 |

On July 22, the four methods shared 2,904,030 valid NDVI comparison
pixels. STAIR's NDVI RMSE was 0.06201, compared with 0.06238 for the
nearest-clear Landsat temporal baseline, 0.06154 for Landsat interpolation
with the same anchors, and 0.14848 for restored MODIS. The standard
Landsat temporal baseline had lower NIR and green RMSE than STAIR.
July 23–26 have coverage checks only. See `run.json` for per-band
scores, coverage, source scenes, grid transforms, and correction support.

An independent read of the exported GeoTIFFs confirmed the grid and NoData
metadata, exact preservation of QA-clear values across restoration stages,
and the two-anchor fusion formula on every valid pixel. The maximum
reconstruction difference was 5.22e-8 reflectance units. The 25 existing
tests also passed.

## Inspect QA exclusions and fusion gaps

Open `raisin_fusion_gap_review.qgz` for the two diagnostic layers in every
prediction-day group. **Pixels removed by MODIS QA** highlights coarse cells
that had original reflectance and were excluded by the quality rule. Turn it
on over **MODIS | 01 Original reflectance** to see where that happened.
**Fusion gap reasons** highlights missing inputs at 30 m and has a separate
legend category for each combination of missing target MODIS, prior anchor,
and later anchor. Expand the layer to see the legend.

On July 22, the original MODIS raster had reflectance in all 13,088 native
watershed cells. The quality rule excluded 75. Temporal interpolation restored
57, leaving 18 unresolved at native resolution. At 30 m, 17,256 pixels had
missing fusion inputs. Of those, 12,344 were missing only a later anchor,
2,207 only target MODIS, 507 only a prior anchor, and 2,198 had a combination
of missing inputs. A usable anchor needs both Landsat and MODIS on the same
date at that pixel. The diagnostic reconstruction matched all five exported
fusion masks exactly. See `gap_diagnostics/summary.json` for the counts.

Generate the diagnosis from the saved products with the project Python.
Then run the QGIS build command below with `--gap-review`.

```sh
python scripts/analyze_watershed_gaps.py outputs/fusion_validation/raisin_2022-07-22_five_days
```

## Restore MODIS with more temporal support

The first pilot excluded MODIS observations from the five prediction dates
when selecting neighboring observations for restoration. The revised runner
uses QA-approved MODIS from any other date. It still withholds Landsat from
all five dates. The same temporal and spatial restoration runs for MODIS on
the target and anchor dates and for Landsat on the anchor dates. Fusion then
estimates Landsat-resolution reflectance on the prediction dates.

When an entire daily MODIS retrieval is absent, the original and QA layers
show NoData and the runner attempts temporal interpolation on the same grid.
Global and adaptive correction require some valid observations on the day
being restored. With no observations to calibrate those corrections, the
runner retains the temporal estimate. Pixels with no valid observation on
one or both sides within the permitted window remain NoData.

The revised comparison uses a maximum MODIS observation age of 16 days on
each side and saves the results separately. **MODIS | 07 MODIS temporal fill
source** distinguishes QA-approved observations, gaps filled within eight
days per side, and gaps requiring a more distant observation. The QA rule
and the Landsat holdout are shared with the first pilot.

```sh
python scripts/run_watershed_fusion.py --modis-days-per-side 16 \
  --output outputs/fusion_validation/raisin_2022-07-22_modis_fill
```

The revised project is `raisin_fusion_gap_review.qgz` in that output folder.
Additional MODIS support increased valid 30 m fusion coverage on every date.
All previously valid pixels remained valid, and all QA-approved native MODIS
values were preserved exactly.

| Date | Extra native MODIS cells restored | Extra 30 m fusion pixels | Valid fusion |
| --- | ---: | ---: | ---: |
| July 22 | 6 | 2,119 | 99.52% |
| July 23 | 21 | 5,682 | 99.46% |
| July 24 | 28 | 7,377 | 99.51% |
| July 25 | 27 | 7,137 | 99.53% |
| July 26 | 25 | 6,683 | 99.53% |

On July 22, NDVI RMSE went from 0.06207 to 0.06221 on the 2,947,477
pixels with valid NDVI in both runs and withheld Landsat. The 1,886
newly covered pixels with valid reference and predicted NDVI had RMSE
0.12105. The wider window increases coverage, with higher error on the
newly supported locations in this benchmark.

## Reproduce the pilot

Run from the repository root with an authenticated Earth Engine project and
the project Python environment.

```sh
python scripts/run_watershed_fusion.py
```

Generate the QGIS project with QGIS Python. On macOS with the packaged QGIS
application, use the following command.

```sh
QT_QPA_PLATFORM=offscreen /Applications/QGIS.app/Contents/MacOS/python scripts/build_watershed_qgis.py \
  outputs/fusion_validation/raisin_2022-07-22_five_days
```

On Windows, run the same script and folder argument with the QGIS Python
environment. The existing Landsat cache can be reused for a rerun. Set
`--output` to a fresh directory. The generated project, cache, and imagery
are ignored by Git.
