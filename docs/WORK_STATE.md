# STAIR working state

Updated 2026-11-08. Read this file at the start of an implementation session,
then check `git status`, the current commit, and `docs/TEMPORAL_STAGE.md` before
editing. This file records the intended next step; Git records source changes.

## Goal

Build one reproducible STAIR gap-filling pipeline with temporal interpolation,
global correction, and adaptive correction. Each stage must have explicit
support and fallback behavior and must be evaluated on held-out clear pixels.
Only after these three stages are implemented and validated should work move on
to Landsat-MODIS fusion or cover-crop analysis.

## Current checkpoint

- Last reviewed starting commit: `57a68d587ac2bceb6e560a8d1aa47c635459b5e9`.
- Maintained temporal implementation: `stair/temporal.py` and
  `scripts/run_temporal_validation.py`.
- Local automated suite: 21 tests passed on November 6, 2026, including
  offline Earth Engine graph checks and a synthetic end-to-end report run.
- The real observation cache `data/temporal/clayton_20220722.npz` is absent in
  this checkout. The measured real-data results in `docs/TEMPORAL_STAGE.md`
  were produced on the other device and have not been reproduced here.
- Local project interpreter `crop_env/bin/python` is Python 3.13.9. Pytest was
  installed into that ignored environment. Recreate environments per device;
  never treat virtual environments or Earth Engine credentials as shared files.
- Working tree also contains uncommitted cross-device changes in
  `README.md`, `docs/CONTINUATION.md`, and `scripts/check_project.py`.

## Next implementation session

1. Inspect `git status` and preserve any edits already in progress. Run
   `crop_env/bin/python scripts/check_project.py --json` (or the current device's
   project interpreter) to record the commit, packages, and cache fingerprint.
2. Obtain the exact existing observation cache from the other device or fetch
   original observations with Earth Engine. Confirm its `run.json` provenance
   and hash before comparing reported real-data scores.
3. Treat temporal-only prediction as the frozen baseline. Add a held-out
   validation path that passes only the artificially gapped target and allowed
   references to correction code; truth is available only to scoring.
4. Resolve and document the spatial-correction equation before modifying
   `stair/adaptive.py`. The current implementation combines clear-pixel temporal
   bias and reference gap/clear contrast. The paper text available so far does
   not settle whether this double-counts spatial contrast. Do not choose an
   equation solely because it improves this one validation chip.
5. Implement and compare global correction first. Then implement adaptive
   class-specific correction using shared, reproducible segments. At each stage
   require exact preservation of observed clear pixels, constant-scene and
   unchanged-scene invariants, no fabricated zeros, explicit sparse-support
   fallbacks, identical holdout masks, and saved diagnostics/metrics.
6. Compare temporal, global, and adaptive outputs on the same cloud-shaped,
   stripe, and full-scene holdouts. Report coverage, unresolved pixels, RMSE,
   MAE, bias, correlation, and prediction R-squared. Re-run offline Earth Engine
   graph checks; run actual Earth Engine parity only when authorized credentials
   and the real cache are available.
7. Update `docs/TEMPORAL_STAGE.md`, this checkpoint, and the README only after
   observed results support the description. Commit changes when the user asks
   or when explicitly agreed; until then leave them available for review.

## Cross-device working rule

Use the Git remote and commit history to move source. Before starting, compare
the local commit and status with this checkpoint. Preserve uncommitted edits on
either machine before synchronizing. Move the ignored `.npz` observation cache
through a user-controlled file transfer and verify SHA-256 on both devices.
Install dependencies from `requirements-dev.txt` into a new local environment.
Never copy virtual environments, authentication files, or credentials. Do not
claim a previous experiment was reproduced until the input-cache hash and
scientific module hashes match.

When resuming this project, continue from this checkpoint and the current Git
state; do not restart the audit or overwrite work already present.
