import numpy as np

from stair.spatial import fit_spatial_offsets


def test_unchanged_heterogeneous_scene_does_not_double_count_contrast():
    reference = np.array([[[.2, .6], [.3, .7]]])
    observed = reference.copy()
    observed[:, :, 1] = np.nan
    segments = np.array([[0, 0], [1, 1]])
    global_offset, class_offsets, _ = fit_spatial_offsets(
        observed, reference, reference, segments, min_pixels=1,
    )
    np.testing.assert_allclose(global_offset, [0], atol=1e-15)
    np.testing.assert_allclose(list(class_offsets.values()), [[0], [0]], atol=1e-15)


def test_class_biases_and_sparse_class_fallback():
    reference = np.array([[[.2, .5], [.3, .6], [.4, .7]]])
    observed = reference.copy()
    observed[:, :, 1] = np.nan
    observed[:, 0, 0] += .05
    observed[:, 1, 0] -= .03
    observed[:, 2, 0] = np.nan
    segments = np.array([[0, 0], [1, 1], [2, 2]])
    global_offset, offsets, support = fit_spatial_offsets(
        observed, reference, reference, segments, min_pixels=1,
    )
    np.testing.assert_allclose(offsets[0], [.05])
    np.testing.assert_allclose(offsets[1], [-.03])
    np.testing.assert_allclose(offsets[2], global_offset)
    assert support["classes"][2]["fallback_to_global"]


def test_fully_masked_scene_retains_temporal_estimate():
    reference = np.full((3, 2, 2), .2)
    global_offset, offsets, support = fit_spatial_offsets(
        np.full_like(reference, np.nan), reference, reference, np.zeros((2, 2)), min_pixels=1,
    )
    assert global_offset == offsets[0] == [0, 0, 0]
    assert support["global"]["fallback_to_temporal"]
