# STAIR output downloads

## Clayton training comparison and cloud validation, November 18

The [latest handoff release](https://github.com/mbharath2231/Stair_Project/releases/tag/untagged-b2806ae9169ed4d2b92d)
contains the trained models and the QGIS reviews from this checkpoint.
Sign in with the GitHub account used for this repository to access the
draft release. Extract both ZIPs into the repository root and verify
their SHA-256 checksums using the release's `SHA256SUMS.txt`.

| ZIP | Contents |
| --- | --- |
| `clayton-training-history-qgis-and-models.zip` | Both fitted ridge corrections, 224 GeoTIFFs for withheld 2025 acquisitions, metrics, plots and the fixed-scale NIR/red/NDVI QGIS project |
| `clayton-cloud-fill-validation.zip` | Five-stage reflectance and NDVI QGIS projects, their TIFFs, and cloud holdout density reports |

Open
`outputs/fusion_validation/history_comparison_2005_2025/training_history_comparison.qgz`
for the training comparison. Its `rasters/` folder must remain beside
the project. The training periods are March–October 2005–2024 and
March–August 2024, with 2025 reserved for evaluation.
The [comparison guide](../docs/TRAINING_HISTORY_COMPARISON.md) records the
model choice, results, limitations and exact reproduction commands.

To rerun training, also download
`clayton-reflectance-input-cache-2005-2025.zip` from the historical
handoff below and extract it into the repository root. The cached
experiment runs offline. Source code and setup instructions are on
branch `codex/stair-fcc-handoff`.

## Clayton historical imagery and validation

Sign in with a GitHub account that can push to this repository and open the
[Clayton historical handoff](https://github.com/mbharath2231/Stair_Project/releases/tag/untagged-d2de7b254ab20b9d8f7c)
draft release. Extract the ZIPs you want into the repository root. Each
archive contains its repository-relative paths. Use the release's
`SHA256SUMS.txt` to verify the downloads.

| ZIP | Contents |
| --- | --- |
| `clayton-fcc-2005-2025.zip` | March–October Landsat and MODIS false-color TIFFs and the shared QGIS style |
| `clayton-reflectance-input-cache-2005-2025.zip` | Historical Red and NIR inputs for model reruns |
| `clayton-reflectance-geotiffs-2005-2025.zip` | Landsat and MODIS annual Red and NIR TIFFs |
| `clayton-ndvi-2005-2025.zip` | Historical NDVI outputs, source layers, and reports |
| `clayton-ndvi-2016-2025-full-year.zip` | Earlier full-year NDVI inputs and outputs |
| `clayton-pilot-qgis-and-validation.zip` | Weekly fusion, temporal, and spatial validation outputs |

For the false-color images, open TIFFs in
`data/clayton_fcc_march_october/<year>/Landsat/` or `<year>/MODIS/`.
The 2005 historical review project uses both the historical NDVI and
reflectance GeoTIFF ZIPs. See the [work checkpoint](../docs/WORK_STATE.md)
for dataset details and reproduction commands.

## Clayton Township March and April fusion

| Week | QGIS archive |
| --- | --- |
| March 13–19, 2024 | [March fusion ZIP](STAIR_Clayton_March_2024_QGIS.zip) |
| April 14–20, 2024 | [April fusion ZIP](STAIR_Clayton_April_2024_QGIS.zip) |

Each ZIP contains seven 30 m fused reflectance GeoTIFFs, seven MODIS GeoTIFFs
aligned to the same grid, one withheld Landsat benchmark GeoTIFF, and the
validation report. Extract the ZIP and open `fusion_week.qgz` beside its
GeoTIFFs. In QGIS, choose **Project → Layouts → FCC week comparison**.

The [weekly fusion runner](../scripts/run_fusion_week.py) generates the
rasters. The [QGIS builder](../scripts/build_fusion_qgis.py) creates the
project. See the [spring run commands](../docs/WORK_STATE.md#spring-2024-qgis-examples).

## River Raisin watershed handoff

The November 16 handoff includes both River Raisin watershed runs, the
Clayton spring validation, and the cached Landsat inputs. Sign in with a
GitHub account that can push to this repository, then open the
[River Raisin watershed handoff](https://github.com/mbharath2231/Stair_Project/releases/tag/untagged-75d14956f6d77391eb32)
draft release and download the ZIPs you need.

| ZIP | Contents |
| --- | --- |
| `raisin-watershed-modis-fill.zip` | Revised five-day run with expanded MODIS restoration and QGIS gap diagnostics |
| `raisin-watershed-original.zip` | Original five-day watershed run and QGIS gap review |
| `clayton-spring-validation.zip` | March and April fusion outputs and the spring NDVI project |
| `raisin-landsat-cache.zip` | Cached Landsat inputs for rerunning the watershed workflow |

Extract the ZIPs directly into `outputs/fusion_validation/`. Each archive
contains its run or cache folder. For the latest run, open
`outputs/fusion_validation/raisin_2022-07-22_modis_fill/raisin_fusion_gap_review.qgz`
in QGIS. Keep the GeoTIFFs alongside the project in their extracted folders.
The draft release also includes `SHA256SUMS.txt` for archive verification.

The extracted files under `outputs/fusion_validation/` are ignored by Git.
See the
[device handoff guide](../docs/CONTINUATION.md) and
[watershed pilot results](../docs/WATERSHED_PILOT.md).
