"""End-to-end cached-data run and geospatial output contracts, without network."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import rasterio


def test_cached_cli_outputs_grid_masks_scores_and_provenance(tmp_path):
    dates = ["2022-07-01", "2022-07-06", "2022-07-11"]
    base = np.linspace(.1, .4, 3*16*16).reshape(3, 16, 16)
    observations = np.stack([base, base+.05, base+.10])
    # Persistent cloud at one pixel: must remain missing even in the clear target.
    observations[[0, 2], :, 8, 8] = np.nan
    metadata = {"target_date": dates[1], "label": "Synthetic test fixture",
                "bands": ["SR_B5", "SR_B4", "SR_B3"], "units": "surface_reflectance",
                "crs": "EPSG:32616", "transform": [30, 0, 700000, 0, -30, 4600000]}
    cache, out = tmp_path / "fixture.npz", tmp_path / "result"
    np.savez_compressed(cache, observations=observations, dates=dates, manifest=json.dumps(metadata))
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts/run_temporal_validation.py"), "--input", str(cache),
                    "--target-date", dates[1], "--pattern", "scene", "--output", str(out)], check=True, capture_output=True)
    report = json.loads((out / "run.json").read_text())
    assert report["excluded_target_day"] == dates[1]
    assert dates[1] not in report["reference_dates"]
    assert report["metrics"]["scored_pixels"] == 255
    assert report["metrics"]["unresolved_pixels"] == 1
    assert report["metrics"]["bands"]["SR_B5"]["rmse"] < 1e-14
    assert len(report["cache_sha256"]) == 64
    with rasterio.open(out / "03_temporal_filled.tif") as src:
        assert src.crs.to_epsg() == 32616
        assert list(src.transform)[:6] == metadata["transform"]
        assert src.descriptions == tuple(metadata["bands"])
        assert src.read(masked=True).mask[:, 8, 8].all()
    with rasterio.open(out / "07_prediction_status.tif") as src:
        assert src.read(1)[8, 8] == 0
        assert src.nodata == 255
    assert (out / "index.html").exists()
    assert (out / "temporal_overview.png").stat().st_size > 1000
