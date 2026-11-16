# STAIR output downloads

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
