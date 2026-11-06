"""Offline source/raster audit; writes only docs/project_audit.json.

Requires NumPy. Never initializes Earth Engine or submits exports. The minimal
TIFF decoder supports only the explicitly asserted formats of audited inputs;
use Rasterio for production raster processing.
"""
import ast
import json
import math
import struct
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def tiff_tags(path):
    data = path.read_bytes()
    endian = '<' if data[:2] == b'II' else '>'
    magic = struct.unpack_from(endian + 'H', data, 2)[0]
    assert magic in (42, 43), (str(path), magic)
    big = magic == 43
    offset = struct.unpack_from(endian + ('Q' if big else 'I'), data, 8 if big else 4)[0]
    n = struct.unpack_from(endian + ('Q' if big else 'H'), data, offset)[0]
    formats = {1: 'B', 2: 'c', 3: 'H', 4: 'I', 5: 'II', 6: 'b',
               7: 'B', 8: 'h', 9: 'i', 11: 'f', 12: 'd', 16:'Q', 17:'q', 18:'Q'}
    tags = {}
    for i in range(n):
        entry = offset + (8 if big else 2) + (20 if big else 12) * i
        tag, typ, count, val = struct.unpack_from(endian + ('HHQQ' if big else 'HHII'), data, entry)
        if typ not in formats:
            continue
        fmt = endian + formats[typ] * count
        size = struct.calcsize(fmt)
        pos = entry + (12 if big else 8) if size <= (8 if big else 4) else val
        tags[tag] = struct.unpack_from(fmt, data, pos)
    return data, endian, tags


def read_tiff(path):
    data, endian, t = tiff_tags(path)
    width, height, bands = t[256][0], t[257][0], t.get(277, (1,))[0]
    bits = t[258][0]
    fmt = t.get(339, (1,))[0]
    assert len(set(t[258])) == 1 and t.get(284, (1,))[0] == 1
    assert t.get(317, (1,))[0] == 1
    dtype = np.dtype(endian + {1: 'u', 2: 'i', 3: 'f'}[fmt] + str(bits // 8))
    compression = t.get(259, (1,))[0]

    def decode(off, size):
        chunk = data[off:off + size]
        if compression in (8, 32946):
            chunk = zlib.decompress(chunk)
        else:
            assert compression == 1, compression
        return np.frombuffer(chunk, dtype=dtype)

    result = np.empty((height, width, bands), dtype=dtype)
    if 324 in t:
        tw, th = t[322][0], t[323][0]
        across = math.ceil(width / tw)
        for i, (off, size) in enumerate(zip(t[324], t[325])):
            tile = decode(off, size).reshape(th, tw, bands)
            y, x = (i // across) * th, (i % across) * tw
            nh, nw = min(th, height-y), min(tw, width-x)
            result[y:y+nh, x:x+nw] = tile[:nh, :nw]
    else:
        rows = t.get(278, (height,))[0]
        for i, (off, size) in enumerate(zip(t[273], t[279])):
            y = i * rows
            nh = min(rows, height-y)
            result[y:y+nh] = decode(off, size).reshape(nh, width, bands)
    return result.transpose(2, 0, 1), t


def scores(pred, truth, mask=None):
    mask = (np.isfinite(pred) & np.isfinite(truth) & (truth > 0)) if mask is None else mask
    p, y = pred[mask].astype(float), truth[mask].astype(float)
    err = p - y
    return {'n': len(y), 'rmse': float(np.sqrt(np.mean(err**2))),
            'mae': float(np.mean(abs(err))), 'bias': float(np.mean(err)),
            'pearson_r': float(np.corrcoef(p, y)[0, 1]),
            'r_squared_score': float(1 - np.sum(err**2) / np.sum((y-y.mean())**2))}


report = {'scope': 'Offline inspection of existing artifacts; no QA certification or GEE execution.'}
sources = list(ROOT.glob('stair/*.py')) + list(ROOT.glob('scripts/*.py')) + list(ROOT.glob('archive/**/*.py')) + [ROOT / 'config.py']
report['python_sources'] = []
for path in sources:
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source, filename=str(path))
    report['python_sources'].append({'path': path.relative_to(ROOT).as_posix(),
        'lines': len(source.splitlines()), 'syntax': 'pass',
        'functions': [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]})

inventory = []
for path in sorted(ROOT.glob('**/*.tif')):
    if 'tmp' in path.relative_to(ROOT).parts:
        continue
    _, _, t = tiff_tags(path)
    keys = t.get(34735, ())
    epsg = None
    for i in range(4, len(keys), 4):
        if keys[i] in (2048, 3072) and keys[i+1] == 0:
            epsg = keys[i+3]
    inventory.append({'path': path.relative_to(ROOT).as_posix(),
        'width': t[256][0], 'height': t[257][0], 'bands': t.get(277,(1,))[0],
        'bits': t[258], 'sample_format': t.get(339,(1,)), 'epsg': epsg,
        'pixel_scale': t.get(33550), 'tie_point': t.get(33922),
        'compression': t.get(259,(1,))[0]})
report['raster_inventory'] = inventory

base = ROOT / 'outputs/STAIR_Fusion_Evaluation'
arrays, tags = {}, {}
for key, filename in [('prior','1_Prior_Ref_Landsat_20220714.tif'),
                      ('post','2_Post_Ref_Landsat_Imputed_20220729.tif'),
                      ('modis','3_Target_MODIS_Resampled_20220722.tif'),
                      ('fused','4_STAIR_Fused_Prediction_20220722.tif'),
                      ('truth','5_True_GroundTruth_Landsat_20220722.tif')]:
    arrays[key], tags[key] = read_tiff(base / filename)
report['saved_benchmark_grid_match'] = all(
    [tags[k].get(n) for n in (256,257,33550,33922,34735)] ==
    [tags['truth'].get(n) for n in (256,257,33550,33922,34735)] for k in tags)
report['saved_benchmark_metrics'] = {
    band: {key: scores(arrays[key][i], arrays['truth'][i])
           for key in ('prior','modis','fused')}
    for i,band in enumerate(['NIR','Red','Green'])}
ndvi = lambda x: (x[0]-x[1])/np.maximum(x[0]+x[1], 1e-4)
report['saved_ndvi_metrics_original_mask'] = {}
for key in ('prior','modis','fused'):
    pred, truth = ndvi(arrays[key]), ndvi(arrays['truth'])
    mask = np.isfinite(pred) & np.isfinite(truth) & (arrays['truth'][0] > 0)
    report['saved_ndvi_metrics_original_mask'][key] = scores(pred,truth,mask)
report['saved_benchmark_ranges'] = {key: {'min':float(np.nanmin(x)), 'max':float(np.nanmax(x)),
    'nonfinite':int(np.count_nonzero(~np.isfinite(x)))} for key,x in arrays.items()}
report['common_support_metrics'] = {}
for i, band in enumerate(['NIR','Red','Green']):
    mask = (arrays['truth'][i] > 0)
    for key in ('truth','prior','modis','fused'):
        mask &= np.isfinite(arrays[key][i])
    report['common_support_metrics'][band] = {key:scores(arrays[key][i],arrays['truth'][i],mask)
        for key in ('prior','modis','fused')}
report['benchmark_missing_pixels'] = int(np.any(~np.isfinite(arrays['fused']),axis=0).sum())
report['benchmark_total_pixels'] = int(np.prod(arrays['fused'].shape[1:]))
same_date_pairs = [
 ('outputs/Real_Cloud_Fusion_20220729/QGIS_Layers/4_STAIR_Imputed_CloudFree_Landsat_FCC.tif',
  'outputs/Real_Cloud_Fusion_20220729/QGIS_Layers/6_FINAL_STAIR_Fused_30m_Prediction_FCC.tif'),
 ('outputs/5km_Stage_By_Stage_Progression/2_Landsat_Progression/2_Landsat_Stage3_Adaptive_CloudFree_FCC.tif',
  'outputs/5km_Stage_By_Stage_Progression/3_Fusion_And_GroundTruth/3_Fusion_3_STAIR_Fused_Prediction_30m_FCC.tif')]
report['same_date_fcc_checks'] = []
for before, after in same_date_pairs:
    a,_ = read_tiff(ROOT/before)
    b,_ = read_tiff(ROOT/after)
    report['same_date_fcc_checks'].append({'before':before,'after':after,
        'equal':bool(np.array_equal(a,b)), 'max_difference':int(np.max(abs(a.astype(int)-b.astype(int))))})

# Execute only the dependency-free numerical function's AST, not package imports.
tree = ast.parse((ROOT/'stair/fusion.py').read_text())
func = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stair_dual_fusion_numpy')
namespace = {'np':np, 'Tuple':tuple}
exec(compile(ast.Module(body=[func],type_ignores=[]),'stair/fusion.py','exec'),namespace)
fuse = namespace['stair_dual_fusion_numpy']
l1,m1,l2,m2,mt = [np.full((3,2,2),v) for v in (.4,.3,.6,.35,.32)]
actual = fuse(l1,m1,l2,m2,mt,7/15,8/15)
expected = mt+(7/15)*(l1-m1)+(8/15)*(l2-m2)
np.testing.assert_allclose(actual,expected)
np.testing.assert_allclose(fuse(l1,m1,l2,m2,m1,1,0),l1)
np.testing.assert_allclose(fuse(l1,m1,l2,m2,m2,0,1),l2)
report['isolated_numpy_checks'] = {'weighted_residual':'pass','first_endpoint':'pass','second_endpoint':'pass'}
report['numeric_counterexamples'] = {
    'dummy_before_real_after': {'target_time':10,'before_time':0,'before_value':0,'after_time':20,'after_value':.4,
        'current_formula':0+(10/20)*.4,'expected_backward_fill':.4},
    'double_spatial_offset': {'perfect_temporal_gap':.6,'true_clear':.2,'pred_clear':.2,'ref_gap':.6,'ref_clear':.2,
        'current_corrected':.6+(.2-.2)+(.6-.2),'truth':.6}}

out = ROOT/'docs/project_audit.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Sources parsed:',len(sources),'lines:',sum(p['lines'] for p in report['python_sources']))
print('Rasters:',len(inventory),'compression:',dict(Counter(p['compression'] for p in inventory)))
print('Matching benchmark grids:',report['saved_benchmark_grid_match'])
print('Isolated NumPy checks:',report['isolated_numpy_checks'])
for band, values in report['common_support_metrics'].items():
    print(band, 'RMSE', {key:round(score['rmse'],6) for key,score in values.items()})
print('Missing:',report['benchmark_missing_pixels'],'/',report['benchmark_total_pixels'])
print('Identical same-date FCC pairs:',sum(p['equal'] for p in report['same_date_fcc_checks']))
print('Full audit:',out)
