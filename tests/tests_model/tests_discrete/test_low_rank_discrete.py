import numpy as np
import pytest

from DensLowRank.model.discrete.low_rank_discrete import generate_dirichlet_low_rank_samples, our_algo

@pytest.mark.parametrize(("d1", "d2", "K", "n", "diric1", "diric2"),
                        (
    (3, 6, 10, 1000, 0.1, 0.2), # simple case
    (0, 0, 10, 1000, 0.1, 0.2), 
))
def test_generate_dirichlet_low_rank_samples_shape_res(d1, d2, K, n, diric1, diric2):

    res, Y1, Y2 = generate_dirichlet_low_rank_samples(d1, d2, K, n, diric1, diric2)
    # Check that the shape of res, Y1, and Y2 are correct
    assert res.shape == (d1, d2)
    assert Y1.shape == (d1, d2)
    assert Y2.shape == (d1, d2)
    

# def test_generate_dirichlet_low_rank_samples_values(d1, d2, K, n, diric1, diric2):

#     res, Y1, Y2 = generate_dirichlet_low_rank_samples(d1, d2, K, n, diric1, diric2)

#     # Check that Y1 and Y2 are scaled versions of res
#     assert np.allclose(Y1 / np.sum(Y1), res, rtol=1e-4)
#     assert np.allclose(Y2 / np.sum(Y2), res, rtol=1e-4)

#pytest.fixture
@pytest.mark.parametrize(("n", "Y1", "Y2", "alpha", "cstar", "Cbar", "expected_shape","error"),
                        (
    (1000, np.ones((10, 10)), np.ones((10, 10)), 1, 1, 1, (10, 10),True), # simple case
    (50, np.ones((10, 10)), np.ones((10, 10)), 1, 1, 1, (10, 10),True), # edge case
    (1000, np.ones((10, 10)), np.ones((5, 5)), 1, 1, 1, (10, 10),False), # invalid Y2 shape
    (1000, np.ones((10, 10)), np.ones((10, 10)), 0, 1, 1, (10, 10),False), # invalid alpha
    (1000, np.ones((10, 10)), np.ones((10, 10)), 1, 0, 1, (10, 10),False), # invalid cstar
    (1000, np.ones((10, 10)), np.ones((10, 10)), 1, 1, 0, (10, 10),False), # invalid Cbar
))
def test_our_algo_shape(n, Y1, Y2, alpha, cstar, Cbar, expected_shape,error):
    res = our_algo(n, Y1, Y2, alpha, cstar, Cbar)
    assert np.shape(res) == expected_shape

# @pytest.mark.parametrize(("n", "Y1", "Y2", "alpha", "cstar", "Cbar"),
#                         (
#     (np.ones((10, 10)) * 1000, np.ones((10, 10)), np.ones((10, 10)), 1, 1, 1)
# ))
# def test_our_algo_type(n, Y1, Y2, alpha, cstar, Cbar):
#     res = our_algo(n, Y1, Y2, alpha, cstar, Cbar)
#     assert res.dtype is  np.ndarray