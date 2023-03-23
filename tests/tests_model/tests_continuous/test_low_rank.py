import numpy as np
import pytest

from our_module import our_algo

@pytest.mark.parametrize("n, d, Y1, Y2, alpha, cstar, Cbar, expected_shape", [
    (1000, 10, np.ones((10, 10)), np.ones((10, 10)), 1, 1, 1, (10, 10)), # simple case
    (50, 10, np.ones((10, 10)), np.ones((10, 10)), 1, 1, 1, (10, 10)), # edge case
    (1000, 10, np.ones((10, 10)), np.ones((10, 5)), 1, 1, 1, (10, 10)), # invalid Y2 shape
    (1000, 10, np.ones((10, 10)), np.ones((10, 10)), 0, 1, 1, (10, 10)), # invalid alpha
    (1000, 10, np.ones((10, 10)), np.ones((10, 10)), 1, 0, 1, (10, 10)), # invalid cstar
    (1000, 10, np.ones((10, 10)), np.ones((10, 10)), 1, 1, 0, (10, 10)), # invalid Cbar
])
def test_our_algo(n, d, Y1, Y2, alpha, cstar, Cbar, expected_shape):
    res = our_algo(n, d, Y1, Y2, alpha, cstar, Cbar)
    assert np.shape(res) == expected_shape
