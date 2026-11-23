# Continue STAIR on the other device

Read [WORK_STATE.md](WORK_STATE.md) for the scientific checkpoint, tested
entry points, and next validation task.

## Start with the latest handoff

Read the [November 23 study guide](handoffs/2026-11-23/STAIR_Clayton_Study_Guide_2026-11-23.docx).
Download the [spatial correction archives](../outputs/README.md#clayton-spatial-correction-and-nonlinear-models-november-23)
and extract them into the repository root. Start with the nonlinear QGIS
project and the final page of the guide. The next experiment compares
the benefit of adding neighborhood inputs while holding the tree size
fixed. The [handoff index](handoffs/README.md) lists the dated guides.

## Get the handoff branch

In the existing checkout, check `git status --short --branch` and preserve
any device-specific edits before switching. Then run:

```sh
git fetch origin
git switch --track origin/codex/stair-fcc-handoff
git log -1 --oneline
```

If the local branch already exists, use `git switch codex/stair-fcc-handoff`
and `git pull --ff-only`. For a new checkout:

```sh
git clone --branch codex/stair-fcc-handoff https://github.com/mbharath2231/Stair_Project.git
```

This branch carries source, tests, and documentation, including the Word
guides in `docs/handoffs/`. The virtual environment, Earth Engine login,
original papers, observation caches, and large generated outputs live
outside Git.

## Open the latest fusion results

The [output download index](../outputs/README.md) links to the GitHub draft
releases and lists the QGIS and cache archives. For the 2005–2025
false-color imagery and historical Clayton work, follow the Clayton
historical handoff entry and extract the ZIPs into the repository root.

The latest watershed run and its baseline are described in
[WATERSHED_PILOT.md](WATERSHED_PILOT.md). Extract the watershed handoff ZIPs
into `outputs/fusion_validation/` in the new checkout. Each archive
includes its run folder, so extract it directly into `fusion_validation`.
Open `raisin_2022-07-22_modis_fill/raisin_fusion_gap_review.qgz` for the
revised run and `raisin_2022-07-22_five_days/raisin_fusion_gap_review.qgz`
for the original run. The Landsat cache archive is useful for reruns.
The projects and their TIFFs open without Earth Engine credentials.

Download and fully extract the
[spring QGIS packages](https://drive.google.com/drive/folders/1wpdxaci5AXc_LljR3KscUgpXQo3YUm8_).
Open `fusion_week.qgz` in a March or April package for the weekly FCC.
Open `spring_review.qgz` in the NDVI package for the cross-month review.
Keep the extracted rasters beside their project files. The
[March fusion audit](WORK_STATE.md#march-13-fusion-audit) records the error
analysis and the next validation task.

## Continue the Landsat and MODIS comparison

The sensor comparison fits Landsat-only, MODIS-only and fused linear
corrections for each training history. Download
`clayton-sensor-comparison.zip` and extract it into the repository root.
[Sensor comparison handoff](https://github.com/mbharath2231/Stair_Project/releases/tag/untagged-9ec4a8fa9ad617609df6)

Open the QGIS project in
`outputs/fusion_validation/sensor_comparison_2005_2025/`. The run's
`run.json` records the training dates and comparison pixels. The same
folder has the six saved fits and a CSV of per-date scores. The
[comparison guide](TRAINING_HISTORY_COMPARISON.md#separate-landsat-and-modis-inputs)
describes how to reproduce the results with the historical input cache.

Use the linear fits, existing fusion and plain Landsat interpolation as
benchmarks when evaluating end-to-end daily fusion. The November 23
nonlinear benchmark evaluates Landsat spatial correction before fusion.

## Continue the original training history comparison

The subsequent [training-history comparison](TRAINING_HISTORY_COMPARISON.md)
trains two identical ridge corrections using 2005–2024 and March–August
2024 supervision, then evaluates withheld 2025 Landsat acquisitions.
It provides offline reproduction commands and the fixed-scale NIR, red
and NDVI QGIS review in
`outputs/fusion_validation/history_comparison_2005_2025/`.

Download the model/QGIS and cloud-validation ZIPs from the
[November 18 handoff release](https://github.com/mbharath2231/Stair_Project/releases/tag/untagged-b2806ae9169ed4d2b92d)
while signed in to the repository's GitHub account. Extract the ZIPs into
the repository root. To reproduce training, also extract the historical
input-cache ZIP listed in the [output download index](../outputs/README.md).
Open `training_history_comparison.qgz` inside the extracted training folder.

The next scientific work is to explain the coverage gaps, compare anchor
quality and test additional independent years. NIR gains and small NDVI
gains from the ridge correction are recorded in the comparison guide.

## Continue the Clayton cloud-filling evaluation

The [cloud-filling validation guide](CLOUD_FILLING_VALIDATION.md) records the
March 13 and April 14, 2024, and July 22, 2022 results. It includes commands to
regenerate the density plots, five-stage Landsat GeoTIFFs, NDVI and signed
error GeoTIFFs, and their QGIS layouts. To reopen existing outputs on another
device, transfer and extract `clayton_cloud_fill_qgis.zip` or
`clayton_cloud_fill_ndvi_qgis.zip` from `outputs/fusion_validation/` and open
the QGIS project inside the extracted folder.

## Set up Python and verify

Use Python 3.10 or later. The handoff was tested on Windows with Python 3.12.
From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-nonlinear.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_project.py --json
```

On macOS or Linux, use `python3` to create the environment and
`.venv/bin/python` for the project commands. The test suite runs offline and
does not need the observation cache. The readiness check reports a missing
cache until it is copied or fetched.

## Before the next device handoff

Write or update that day's Word summary under `docs/handoffs/YYYY-MM-DD/`.
Include the completed experiments, score tables and what they can show,
paths to reproducible results, and the first task for the other device.
Add the summary to the [handoff index](handoffs/README.md) and commit and
push it with the code. List any new ZIPs in the
[output download index](../outputs/README.md).

## Reopen the reviewed FCC in QGIS

1. Download the [five reviewed July 29 GeoTIFFs](https://drive.google.com/drive/folders/1pIxeL7LvhbqrsVK77INyJaIUbZGw43P4)
   to `outputs/spatial_validation/2022-07-29_reviewed/`. Remove the
   `STAIR_20220729_` prefix from each filename. The expected names are
   `01_original.tif`, `02_cloud_masked.tif`, `03_temporal.tif`,
   `04_global.tif`, and `05_adaptive.tif`.
2. Use the Python distributed with QGIS to run the layout builder. The command
   below uses the Windows path tested here. Use the launcher path from the
   other device's QGIS installation.

   ```powershell
   & 'C:\Program Files\QGIS 3.44.15\bin\python-qgis-ltr.bat' scripts/build_stair_qgis.py outputs/spatial_validation/2022-07-29_reviewed --fcc
   ```

3. Open `stair_fcc_comparison.qgz` from that directory and choose **Project →
   Layouts → FCC comparison**. The builder also writes
   `stair_fcc_comparison.png`. Keep the QGZ next to the five TIFFs so its
   relative paths resolve. If you transfer the already generated QGZ together
   with the named TIFFs, open it directly.

The reviewed TIFFs can also be opened directly with QGIS. Use **Multiband
color**, TIFF bands **1/2/3** for display RGB, and **Stretch to MinMax** with
the shared limits in [WORK_STATE.md](WORK_STATE.md). Viewing the Drive TIFFs
in QGIS requires no Earth Engine login.

## Rerun from Earth Engine

Authenticate separately on the new device. The default cloud project is
`stair-project-509517`. Use `EE_PROJECT_ID` or `--project` if you need a
different authorized project. These Windows commands regenerate local results:

```powershell
.\.venv\Scripts\earthengine.exe authenticate
.\.venv\Scripts\python.exe scripts/export_stair_stages.py --output outputs/spatial_validation/2022-07-29_reproduced
.\.venv\Scripts\python.exe scripts/export_stair_stages.py --target 2022-07-22 --reference 2022-07-21 --holdout-cloud --output outputs/spatial_validation/2022-07-22_holdout_reviewed
```

On macOS or Linux, the authentication launcher is `.venv/bin/earthengine`.
The July 29 command writes five TIFFs and `run.json`. The July 22 holdout
writes scores to `run.json` and keeps the withheld observations out of model
fitting. Compare scene IDs, grid, coverage, class count, and metrics with the
checkpoint. Add `--drive-folder <new-folder-name>` only when you want a fresh
set of exports. The reviewed Drive folder already has its five TIFFs.

## Recover the separate temporal baseline

Copy `data/temporal/clayton_20220722.npz` to the same relative path if you
have the tested cache. The expected SHA-256 is
`5a941fac181e19e889521a24e7a6a1a8a7c7a4e6f3164aa5e7a7525bc4f5fa69`.
Otherwise, after authenticating, run `scripts/run_temporal_validation.py
--fetch` with the project Python and record the new cache hash and scene
manifest. With a cache in place, the script runs locally. It accepts
`--pattern cloud`, `--pattern stripes`, and `--pattern scene`; give each
run a distinct `--output` directory. See [TEMPORAL_STAGE.md](TEMPORAL_STAGE.md)
for the original temporal experiment.

Keep both supplied papers available locally if you will audit the method.
Luo et al. (2018) defines STAIR as SaTellite dAta IntegRation. The cover-crop
paper is the reference for later application work.
