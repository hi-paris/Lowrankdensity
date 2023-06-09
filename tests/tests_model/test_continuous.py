import pytest
import numpy as np
from lowrankdensity.models.continuous import Continuous
from lowrankdensity.datasets._generate_samples import generate_lowrank_continuous


def test_continuous_1d_empty_parameter_error():  # typeerror case 1 empty parameters error for continuous_1d
    model = Continuous()
    with pytest.raises(Exception):
        model.continuous_1d()


def test_fit_valid_result():  # case 2 checking the result value is none fit function
    model = Continuous()
    X = generate_lowrank_continuous()
    result = model.fit(X)
    assert result is None


@pytest.mark.parametrize(("alpha"),  # case 3 alpha error
                         (
    (-1),
    ('a'),
    (-0.5),
    (True)
))
def test_fit_error_alpha(alpha):  # alpha error
    X = generate_lowrank_continuous()
    with pytest.raises(Exception):
        model = Continuous(alpha=alpha)
        model.fit(X)


@pytest.mark.parametrize(("X"),  # case 4 bad X values
                         (
    (np.ones((10, 0))),  # shape not 2
    (np.ones((0, 10))),  # empty
    (np.ones((0, 0))),  # empty
))
def test_fit_error_X(X):  # case  bad X values
    with pytest.raises(Exception):
        model = Continuous()
        model.fit(X)


def test_pdf_type():  # case 5 type assert for pdf
    X = generate_lowrank_continuous()
    model = Continuous(alpha=0.1)
    model.fit(X)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    # lowrank_matrix = model.pdf(x,y)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_pdf_shape():  # case 6 shape assert for pdf
    X = generate_lowrank_continuous()
    model = Continuous(alpha=0.1)
    model.fit(X)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    res = model.pdf(x, y)
    assert res.shape == (x.shape[0], y.shape[0])


def test_pdf_dtype():  # case 7 dtype assert for pdf
    X = generate_lowrank_continuous()
    model = Continuous(alpha=0.1)
    model.fit(X)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    res = model.pdf(x, y)
    assert res.dtype == float or int


@pytest.mark.parametrize(("x", "y"),  # case 8 bad shape error for pdf
                         (
    (np.ones((10, 2)), np.ones((10, 2))),
    (np.ones((10, 0)), np.ones((10, 1))),
    (np.ones((10, 1)), np.ones((0, 3))),
    (np.ones((0, 10)), np.ones((0, 0))),
))
def test_pdf_shape_error(x, y):
    with pytest.raises(Exception):  # index error
        model = Continuous(alpha=0.1)
        model.pdf(x=x, y=y)


def test_pdf_empty_parameter_error():  # case 9 empty parameter error for pdf
    model = Continuous(alpha=0.1)
    with pytest.raises(Exception):
        model.pdf()


def test_sample_shape():  # case 10 sample shape
    model = Continuous()
    X = generate_lowrank_continuous()
    n_samples = 1000
    model.fit(X)
    samples = model.sample()
    assert samples.shape == (n_samples, 2)


def test_sample_type():  # case 11 sample type
    model = Continuous()
    X = generate_lowrank_continuous()
    model.fit(X)
    samples = model.sample()
    assert samples.dtype == int or float


@pytest.mark.parametrize(("sample"),  # case 12 bad sample
                         (
    (-4),
    ('gh'),
    (-0.5),
    (True)
))
def test_sample_error_n_sample(sample):  # case 12 bad sample
    model = Continuous()
    X = generate_lowrank_continuous()
    model.fit(X)
    with pytest.raises(Exception):
        model.sample(sample)
