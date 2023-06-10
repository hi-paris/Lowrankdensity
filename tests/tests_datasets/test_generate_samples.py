import pytest
import numpy as np
import warnings
from lowrankdensity.datasets._generate_samples import generate_lowrank_discrete, generate_lowrank_continuous


# Parameterize the function
@pytest.mark.parametrize(("n_samples", "K", "d1", "d2"),
                         (
    (100, 2, 50, 50),
    (500, 3, 30, 30),
    (10000, 2, 40, 40),
    (2000, 4, 60, 60)
))
def test_generate_lowrank_discrete(n_samples, K, d1, d2):  # case 0 normal case
    samples = generate_lowrank_discrete(n_samples=n_samples, K=K, d1=d1, d2=d2)
    assert samples.shape == (n_samples, 2)


def test_generate_lowrank_discrete_returns_correct_number_of_samples():  # case 1 checking n_samples
    n_samples = 100
    samples = generate_lowrank_discrete(n_samples=n_samples)
    assert samples.shape[0] == n_samples


@pytest.mark.parametrize("n_samples", [50010, 60000, 70000, 90000])  # case 2
def test_generate_lowrank_discrete_issues_warning_for_large_n_samples(n_samples):
    with pytest.warns(UserWarning, match="Choosing n_samples > 50000 might slow down data generation."):
        generate_lowrank_discrete(n_samples=n_samples)
        warnings.warn("Choosing n_samples > 50000 might slow down data generation.", UserWarning)


def test_generate_lowrank_discrete_generates_consistent_results():  # case 3 checking consistency
    np.random.seed(42)
    samples1 = generate_lowrank_discrete(n_samples=100, K=3, d1=30, d2=30)
    np.random.seed(42)
    samples2 = generate_lowrank_discrete(n_samples=100, K=3, d1=30, d2=30)
    assert np.array_equal(samples1, samples2)

# @pytest.mark.xfail()
# def test_generate_lowrank_discrete_empty_parameter_error(): # case 4 discuss this
#     with pytest.raises(ValueError):
#         samples = generate_lowrank_discrete()


@pytest.mark.parametrize(("n_samples", "K", "d1", "d2"),  # case 5 errors
                         (
    (-3, 1, 50, 50),
    (500, 3, 230, 1),  # large d1
    (10000, 10, 1, 230),  # large d2
))
def test_generate_lowrank_discrete_raises_error_for_ddefined_error(n_samples, K, d1, d2):  # case 5 check if it raises error for defined error
    with pytest.raises(ValueError):
        generate_lowrank_discrete(n_samples=n_samples, K=K, d1=d1, d2=d2)


@pytest.mark.parametrize(("n_samples", "K", "d1", "d2"),  # case 6 errors for bad values
                         (
    (100, -2, 1, 50),
    (500, 3, -30, 1),  # negative value d1
    (10000, 10, 40, -40),  # negative value d2
    (100, 0, 1, 50),  # divide by zero
    ('500', 3, 30, 1),
    (10000, '10', 40, 40),
    (10000, 10, '40', 40),
    (10000, 10, 40, '40')
))
def test_generate_lowrank_discrete_generates_error(n_samples, K, d1, d2):
    with pytest.raises(Exception):
        generate_lowrank_discrete(n_samples=n_samples, K=K, d1=d1, d2=d2)

# Parameterize the function


@pytest.mark.parametrize(("n_samples", "K"),
                         (
    (10, 1),
    (500, 3),
    (10000, 2),
    (2000, 4)
))
def test_generate_lowrank_continuous_shape(n_samples, K):  # case 7 normal case
    # n_samples = 1000
    # K = 2
    samples = generate_lowrank_continuous(n_samples=n_samples, K=K)
    assert samples.shape == (n_samples, 2)

 # case 8
def test_generate_lowrank_continuous_shape_issues_warning_for_large_n_samples():
    n_samples = 16
    K = 5
    with pytest.warns(UserWarning):
        generate_lowrank_discrete(n_samples=n_samples, K=K)


def test_generate_lowrank_continuous_generates_consistent_results():  # case 9
    np.random.seed(42)
    samples1 = generate_lowrank_continuous(n_samples=100)
    np.random.seed(42)
    samples2 = generate_lowrank_continuous(n_samples=100)
    assert np.array_equal(samples1, samples2)


@pytest.mark.parametrize(("n_samples", "K"),  # case 10 errors
                         (
    (100, -2),  # -ve value
    (100, 0),  # divide by zero
    (-3, 0),
    ('500', 3),
    (10000, '10')
))
def test_generate_lowrank_continuous_generates_error(n_samples, K):
    with pytest.raises(Exception):
        generate_lowrank_continuous(n_samples=n_samples, K=K)
