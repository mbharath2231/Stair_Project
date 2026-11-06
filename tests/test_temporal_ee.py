"""Offline graph checks; these do not claim server-side numerical validation."""
import ee
from ee import apitestcase
from stair.temporal import interpolate_temporal_ee, impute_temporal
from stair.landsat import prepare_landsat


class TestTemporalGraph(apitestcase.ApiTestCase):
    def test_empty_collections_and_carry_graphs_serialize(self):
        raw = ee.Image.constant(.5).rename("SR_B5").clip(ee.Geometry.Rectangle([0, 0, .1, .1]))
        for carry in (False, True):
            result = interpolate_temporal_ee(ee.ImageCollection([]), raw, "2022-07-06", allow_one_sided=carry)
            for image in (result.prediction, result.filled, result.diagnostics):
                assert image.serialize()
        pred, filled = impute_temporal(ee.ImageCollection([]), raw, "2022-07-06")
        assert pred.serialize() and filled.serialize()

    def test_real_collection_graph_and_qa_preparation(self):
        raw = ee.Image("LANDSAT/LC08/C02/T1_L2/LC08_020031_20220722")
        prepared = prepare_landsat(raw)
        assert prepared.serialize()
        collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").map(prepare_landsat)
        result = interpolate_temporal_ee(collection, prepared, "2022-07-22")
        assert "qualityMosaic" in result.prediction.serialize()
