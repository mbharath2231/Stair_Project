# STAIR output downloads

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

Git tracks this index. The generated files under `outputs/fusion_validation/`
are ignored and come from the release archives. See the
[device handoff guide](../docs/CONTINUATION.md) and
[watershed pilot results](../docs/WATERSHED_PILOT.md).
