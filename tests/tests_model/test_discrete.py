import pytest
import numpy as np
from lowrankdensity.models.discrete import Discrete
from lowrankdensity.datasets._generate_samples import generate_lowrank_discrete


def test_compute_histograms_shape_Y1max():  # case 1 Y1max shape histogram
    X = np.array([[1, 2], [1, 3], [2, 2], [2, 3]])  # unique values
    # X = np.ones((10,3))
    expected_shape = (2, 2)
    model = Discrete(alpha=0.1)
    Y1max, Y2max = model._compute_histograms(X)
    assert Y1max.shape == expected_shape


def test_compute_histograms_shape_Y2max():  # case 2 Y2max shape histogram
    X = np.array([[1, 2], [1, 3], [2, 2], [2, 3]])
    expected_shape = (2, 2)
    model = Discrete(alpha=0.1)
    Y1max, Y2max = model._compute_histograms(X)
    assert Y2max.shape == expected_shape


def test_compute_histograms_type_Y1max():  # case 3 Y1max type assertion
    X = np.array([[1, 2], [1, 3], [2, 2], [2, 3]])
    model = Discrete(alpha=0.1)
    Y1max, Y2max = model._compute_histograms(X)
    assert Y1max.dtype == int


def test_compute_histograms_type_Y2max():  # case 4 Y2 max type assertion
    X = np.array([[1, 2], [1, 3], [2, 2], [2, 3]])
    model = Discrete(alpha=0.1)
    Y1max, Y2max = model._compute_histograms(X)
    assert Y2max.dtype == int


def test_compute_histograms_empty_input_error():  # indexerror case 5 empty input error for _compute_histograms
    X = np.array([])
    model = Discrete(alpha=0.1)
    with pytest.raises(IndexError):
        Y1max, Y2max = model._compute_histograms(X)


def test_compute_histograms_empty_parameter_error():  # typeerror case 6 empty parameters error for _compute_histograms
    model = Discrete(alpha=0.1)
    with pytest.raises(Exception):
        Y1max, Y2max = model._compute_histograms()


@pytest.mark.parametrize(("n", "Y1", "Y2", "expected_shape"),  # case 7 type + shape
                         (
    (100, np.ones((100, 2)), np.ones((100, 2)), (100, 2)),  # simple case
    (50, np.ones((50, 2)), np.ones((50, 2)), (50, 2)),  # edge case
))
def test__compute_matrix_shape_for_res(n, Y1, Y2, expected_shape):  # compute matrix with discrete = False
    try:
        model = Discrete(alpha=0.1)
        res = model._compute_matrix(n=n, Y1=Y1, Y2=Y2, discrete=False)
        assert res.shape == expected_shape
        assert res.dtype == float
    except BaseException:
        print(f"Assertion error: n={n}, Y1.shape={Y1.shape}, Y2.shape={Y2.shape}, expected_shape={expected_shape}")
        raise


@pytest.mark.parametrize(("n", "Y1", "Y2"),  # case 8 bad shapes
                         (
    (1000, np.ones((10, 0)), np.ones((0, 10))),
    (1000, np.ones((10, 2)), np.ones((0, 10))),
    (1000, np.ones((10, 2)), np.ones((10, 0))),
    (1000, np.ones((0, 2)), np.ones((10, 2))),
    (1000, np.ones((10, 0)), np.ones((10, 2))),
))
def test__compute_matrix_shape_error(n, Y1, Y2):
    # invalid shape errors
    with pytest.raises(Exception):  # index error
        model = Discrete()
        model._compute_matrix(n=n, Y1=Y1, Y2=Y2, discrete=False)


def test__compute_matrix_shape_with_generate_discrete_false_Y1():  # case 9 Y1 check when discrete is false
    X = generate_lowrank_discrete()
    model = Discrete()
    Y1, Y2 = model._compute_histograms(X)
    n = X.shape[0]
    res = model._compute_matrix(n=n, Y1=Y1, Y2=Y2, discrete=False)
    assert res.shape == Y1.shape
    assert res.dtype == float


def test__compute_matrix_shape_with_generate_discrete_true_Y1():  # case 10 Y1 when discrete is True
    X = generate_lowrank_discrete()
    model = Discrete()
    Y1, Y2 = model._compute_histograms(X)
    n = X.shape[0]
    res = model._compute_matrix(X, n=n, Y1=Y1, Y2=Y2, discrete=True)
    assert res.shape == Y1.shape
    assert res.dtype == float


def test__compute_matrix_shape_with_generate_discrete_false_Y2():  # case 11 Y2 check when discrete is false
    X = generate_lowrank_discrete()
    model = Discrete()
    Y1, Y2 = model._compute_histograms(X)
    n = X.shape[0]
    res = model._compute_matrix(n=n, Y1=Y1, Y2=Y2, discrete=False)
    assert res.shape == Y2.shape
    assert res.dtype == float


def test__compute_matrix_shape_with_generate_discrete_true_Y2():  # case 12 Y2 check when discrete is true macth histogram with matrix
    X = generate_lowrank_discrete()
    model = Discrete()
    Y1, Y2 = model._compute_histograms(X)
    n = X.shape[0]
    res = model._compute_matrix(X, n=n, Y1=Y1, Y2=Y2, discrete=True)
    assert res.shape == Y2.shape
    assert res.dtype == float


def test_fit_value_with_generate_discrete_match():  # case 13 compute matrix and fit function
    X = generate_lowrank_discrete()
    model = Discrete()
    model.fit(X)
    val = model.probability_matrix
    res = model._compute_matrix(X)
    assert res.dtype == val.dtype


def test_fit_dtype_with_generate_discrete_match():  # case 14 compute matrix and fit function
    X = generate_lowrank_discrete()
    model = Discrete()
    model.fit(X)
    val = model.probability_matrix
    res = model._compute_matrix(X)
    assert res.dtype == val.dtype


def test_fit_valid_input():  # case 15 checing the o/p of the function
    X = generate_lowrank_discrete()
    model = Discrete()
    res = model.fit(X)
    assert res is None


@pytest.mark.parametrize(("alpha"),  # case 16 alpha exception
                         (
    (-1),
    ('a'),
    (-0.5),
    (True)
))
def test_fit_error_alpha(alpha):  # case 17 alpha exception
    X = generate_lowrank_discrete()
    with pytest.raises(Exception):
        model = Discrete(alpha=alpha)
        model.fit(X)


@pytest.mark.parametrize(("X"),  # case 18 bad X values
                         (
    (np.ones((10, 3))),
    (np.ones((10, 0))),  # shape not 2
    (np.ones((0, 10))),  # empty
    (np.ones((0, 0))),  # empty
))
def test_fit_error_X(X):  # case 18 bad X values
    with pytest.raises(Exception):
        model = Discrete()
        model.fit(X)


def test_sample_shape():  # case 19 asserting shape for sample
    model = Discrete()
    X = generate_lowrank_discrete()
    n_samples = 1000
    model.fit(X)
    samples = model.sample(n_samples)
    assert samples.shape == (n_samples, 2)


def test_sample_type():  # case 20 asserting type
    model = Discrete()
    X = generate_lowrank_discrete()
    n_samples = 1000
    model.fit(X)
    samples = model.sample(n_samples)
    assert samples.dtype == int or float


@pytest.mark.parametrize(("sample"),  # case 21 bad sample values
                         (
    (-4),
    ('gh'),
    (-0.5),
    (True)
))
def test_sample_error_n_sample(sample):  # case 20 asserting type
    model = Discrete()
    X = generate_lowrank_discrete()
    model.fit(X)
    with pytest.raises(Exception):
        model.sample(sample)


def test_fit_empty_parameter_error():  # typeerror case 21 empty parameters error for fit
    model = Discrete(alpha=0.1)
    with pytest.raises(Exception):
        model.fit()
