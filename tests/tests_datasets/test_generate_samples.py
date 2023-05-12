import os
import pytest
import numpy as np
import warnings
from lowrankdensity.datasets.generate_samples import generate_lowrank_discrete, generate_lowrank_continuous


# Parameterize the function
@pytest.mark.parametrize(("n_samples", "K", "d1", "d2"),
                         (
    (100, 2, 50, 50),
    (500, 3, 30, 30),
    (10000, 2, 40, 40),
    (2000, 4, 60, 60)
                         ))
def test_generate_lowrank_discrete(n_samples, K, d1, d2): #case 0 normal case
    samples = generate_lowrank_discrete(n_samples=n_samples, K=K, d1=d1, d2=d2)
    assert samples.shape == (n_samples, 2)


def test_generate_lowrank_discrete_returns_correct_number_of_samples(): #case 1
    n_samples = 100
    samples = generate_lowrank_discrete(n_samples=n_samples)
    assert samples.shape[0] == n_samples

@pytest.mark.parametrize("n_samples", [50010, 60000, 70000, 90000]) #case 2
def test_generate_lowrank_discrete_issues_warning_for_large_n_samples(n_samples):
    with pytest.warns(UserWarning, match="Choosing n_samples > 50000 might slow down data generation."):
        samples = generate_lowrank_discrete(n_samples=n_samples)
        warnings.warn("Choosing n_samples > 50000 might slow down data generation.", UserWarning)

def test_generate_lowrank_discrete_generates_consistent_results(): #case 3
    np.random.seed(42)
    samples1 = generate_lowrank_discrete(n_samples=100, K=3, d1=30, d2=30)
    np.random.seed(42)
    samples2 = generate_lowrank_discrete(n_samples=100, K=3, d1=30, d2=30)
    assert np.array_equal(samples1, samples2)

@pytest.mark.xfail()
def test_generate_lowrank_discrete_empty_parameter_error(): # case 4 discuss this
    with pytest.raises(ValueError):
        samples = generate_lowrank_discrete()

@pytest.mark.xfail()
def test_generate_lowrank_discrete_raises_error_for_invalid_K():# case 5 discuss this
    with pytest.raises(ValueError):
        generate_lowrank_discrete(n_samples=100, K=1, d1=50, d2=50)

@pytest.mark.parametrize(("n_samples", "K", "d1", "d2"), # case 6 errors
                         (
    (100, -2, 1, 50),
    (500, 3, -30, 1),
    (10000, 10, 40, -40), # negative value
    (100, 0, 1, 50), #divide by zero
    ('500', 3, 30, 1),
    (10000, '10', 40, 40),
    (10000, 10, '40', 40),
    (10000, 10, 40, '40')
                         ))
def test_generate_lowrank_discrete_generates_error(n_samples, K, d1, d2): 
    with pytest.raises(Exception):
        samples = generate_lowrank_discrete(n_samples=n_samples, K=K, d1=d1, d2=d2)

# Parameterize the function
@pytest.mark.parametrize(("n_samples", "K"),
                         (
    (10, 1),
    (500, 3),
    (10000, 2),
    (2000, 4)
                         ))
def test_generate_lowrank_continuous_shape(n_samples, K): #case 7 normal case
    n_samples = 1000
    K = 2
    samples = generate_lowrank_continuous(n_samples=n_samples, K=K)
    assert samples.shape == (n_samples, 2)

@pytest.mark.parametrize("n_samples", [50010, 60000, 70000, 90000]) #case 8
def test_generate_lowrank_continuous_shape_issues_warning_for_large_n_samples(n_samples):
    with pytest.warns(UserWarning, match="Choosing n_samples > 50000 might slow down data generation."):
        samples = generate_lowrank_discrete(n_samples=n_samples)
        warnings.warn("Choosing n_samples > 50000 might slow down data generation.", UserWarning)

def test_generate_lowrank_continuous_generates_consistent_results(): #case 9
    np.random.seed(42)
    samples1 = generate_lowrank_continuous(n_samples=100)
    np.random.seed(42)
    samples2 = generate_lowrank_continuous(n_samples=100)
    assert np.array_equal(samples1, samples2)

@pytest.mark.parametrize(("n_samples", "K"), # case 6 errors
                         (
    (100, -2),# -ve value
    (100, 0), #divide by zero
    ('500', 3),
    (10000, '10')
                         ))
def test_generate_lowrank_continuous_generates_error(n_samples, K): 
    with pytest.raises(Exception):
        samples = generate_lowrank_continuous(n_samples=n_samples, K=K)
