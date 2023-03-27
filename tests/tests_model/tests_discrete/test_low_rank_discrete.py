import numpy as np
import pytest

from DensLowRank.model.discrete.low_rank_discrete import generate_dirichlet_low_rank_samples, loss_calculation, our_algo

# @pytest.mark.parametrize(("d1", "d2", "K", "n", "diric1", "diric2"), #case 0
#                         (
#     (3, 6, 10, 1000, 0.1, 0.2), # simple case
#     (9, 12, 10, 1000, 0.3, 0.6), # #different dimension
# ))
# def test_generate_dirichlet_low_rank_samples_values(d1, d2, K, n, diric1, diric2):

#     res, Y1, Y2 = generate_dirichlet_low_rank_samples(d1, d2, K, n, diric1, diric2)

#     # Check that Y1 and Y2 are scaled versions of res
#     assert np.allclose(np.sum(Y1), n, rtol=1e-3)
#     assert np.allclose(np.sum(Y2), n, rtol=1e-3)

@pytest.mark.parametrize(("d1", "d2", "K", "n", "diric1", "diric2"), #case 1
                        (
    (3, 6, 10, 1000, 0.1, 0.2), # simple case
    (9, 12, 10, 1000, 0.3, 0.6), # simple case
    (0, 0, 10, 1000, 0.1, 0.2), # when array is empty; edge case
    (9, 12, 0, 1000, 0, 0) # when diric 1 and diric2 is 0; edge case
))
def test_dirichlet_shape_for_res(d1, d2, K, n, diric1, diric2):

    res,Y1,Y2 = generate_dirichlet_low_rank_samples(d1, d2, K, n, diric1, diric2)
    # Check that the shape of res
    assert np.shape(res) == (d1, d2)

@pytest.mark.parametrize(("d1", "d2", "K", "n", "diric1", "diric2"), #case 2
                        (
    (3, 6, 10, 1000, 0.1, 0.2), # simple case
    (9, 12, 10, 1000, 0.3, 0.6), # simple case; 
    (0, 0, 10, 1000, 0.1, 0.2), # when array is empty; edge case
))
def test_dirichlet_shape_for_Y1(d1, d2, K, n, diric1, diric2):

    res,Y1,Y2 = generate_dirichlet_low_rank_samples(d1, d2, K, n, diric1, diric2)
    # Check that the shape of Y1
    assert np.shape(Y1) == (d1, d2)

@pytest.mark.parametrize(("d1", "d2", "K", "n", "diric1", "diric2"), #case 3
                        (
    (3, 6, 10, 1000, 0.1, 0.2), # simple case
    (0, 0, 10, 1000, 0.1, 0.2), # when array is empty; edge case
))
def test_dirichlet_shape_for_Y2(d1, d2, K, n, diric1, diric2): 

    res,Y1,Y2 = generate_dirichlet_low_rank_samples(d1, d2, K, n, diric1, diric2)
    # Check that the shape of Y1
    assert np.shape(Y2) == (d1, d2)

    
@pytest.mark.parametrize(("d1", "d2", "K", "n", "diric1", "diric2"), #case 4
                        (
    (3, 6, 10, 1000, 0.1, 0.2), # simple case
    (0, 0, 10, 1000, 0.1, 0.2), # when array is empty; edge case

#    (0, 0, 10, 0, 0.1, 0.2), # 
))
def test_dirichlet_type_for_res(d1, d2, K, n, diric1, diric2):

    res,Y1,Y2 = generate_dirichlet_low_rank_samples(d1, d2, K, n, diric1, diric2)
    # Check that the type of res is np.ndarray
    assert type(res) == np.ndarray
    assert res.dtype == float

@pytest.mark.parametrize(("d1", "d2", "K", "n", "diric1", "diric2"), #case 5
                        (
    (3, 6, 10, 1000, 0.1, 0.2), # simple case
    (0, 0, 10, 1000, 0.1, 0.2), # when array is empty; edge case
))
def test_dirichlet_type_for_Y1(d1, d2, K, n, diric1, diric2):

    res,Y1,Y2 = generate_dirichlet_low_rank_samples(d1, d2, K, n, diric1, diric2)
    # Check that the type of Y1 is np.ndarray
    assert type(Y1) == np.ndarray
    assert Y1.dtype == float

@pytest.mark.parametrize(("d1", "d2", "K", "n", "diric1", "diric2"), #case 6
                        (
    (3, 6, 10, 1000, 0.1, 0.2), # simple case
    (0, 0, 10, 1000, 0.1, 0.2), # when array is empty edge case
))
def test_dirichlet_type_for_Y2(d1, d2, K, n, diric1, diric2): 

    res,Y1,Y2 = generate_dirichlet_low_rank_samples(d1, d2, K, n, diric1, diric2)
    # Check that the type of Y1 is np.ndarray
    assert type(Y2) == np.ndarray
    assert Y2.dtype == float

@pytest.mark.parametrize(("d1", "d2", "K", "n", "diric1", "diric2"), #case 7
                        (
    (0, 0, 10, 1000, 0.1, 0.2), # fail case
    (0, 5, 10, 1000, 0.1, 0.2), # fail case
))
#@pytest.mark.xfail(reason="Bug #1: Empty values not supported", strict=True)
def test_dirichlet_none_values_assert_error(d1, d2, K, n, diric1, diric2): 
    total = generate_dirichlet_low_rank_samples(d1, d2, K, n, diric1, diric2)
    # Check that the values of res, Y1, and Y2 are not empty
    assert not np.any(total)

def test_dirichlet_parameter_error(): #case 8
    # empty parameter exception
    with pytest.raises(Exception):
        generate_dirichlet_low_rank_samples()

@pytest.mark.parametrize(("d1", "d2", "K", "n", "diric1", "diric2"), #case 7
                        (
    (2.5, 3, 4, 100, 0.5, 0.5), # fail case
    (2, 3.5, 4, 100, 0.5, 0.5), # fail case
    (2, 3.5, 4, 100, 0.5, 0.5), 
    (2, 3, 4.5, 100, 0.5, 0.5),
    (2, 3, '4', 100, 0.5, 0.5),
    (2, 3, 4, '105', 0.5, 0.5),
    (2, 3, 4, 100, 0.5, '0.5'),
    (2, 3, 4, 100, '0.5', 0.5),
))
def test_dirichlet_type_error(d1, d2, K, n, diric1, diric2):
    with pytest.raises(TypeError):
        generate_dirichlet_low_rank_samples(d1, d2, K, n, diric1, diric2)


@pytest.mark.parametrize(("n", "Y1", "Y2", "alpha", "cstar", "Cbar", "expected_shape","error"), # case 9
                        (
    (1000, np.ones((10, 10)), np.ones((10, 10)), 1, 1, 1, (10, 10),False), # simple case
    (50, np.ones((10, 10)), np.ones((10, 10)), 1, 1, 1, (10, 10),False), # edge case
    (1000, np.ones((5, 5)), np.ones((5, 5)), 1, 1, 1, (10, 10),True), # different Y2 Y1 shape
#    (1000, np.ones((10, 0)), np.ones((0, 10)), 1, 1, 1, (10, 10),True), # bad shape 
    (1000, np.ones((10, 10)), np.ones((10, 10)), 0, 1, 1, (10, 10),False), # invalid alpha
    (1000, np.ones((10, 10)), np.ones((10, 10)), 1, 0, 1, (10, 10),False), # invalid cstar
    (1000, np.ones((10, 10)), np.ones((10, 10)), 1, 1, 0, (10, 10),False), # invalid Cbar

))
def test_our_algo_shape(n, Y1, Y2, alpha, cstar, Cbar, expected_shape,error):
    try:
        res = our_algo(n, Y1, Y2, alpha, cstar, Cbar)
        assert res.shape == expected_shape
        assert res.dtype == float
    except :
        pytest.fail("unexpected shape")


@pytest.mark.parametrize(("n", "Y1", "Y2", "alpha", "cstar", "Cbar"), # case 10
                        (
    (1000, np.ones((10, 0)), np.ones((0, 10)), 1, 1, 1), # invalid shape  passes
))
def test_our_algo_shape_error(n, Y1, Y2, alpha, cstar, Cbar): 
    # invalid shape errors
    with pytest.raises(Exception):
    #    pytest.fail("unexpected shape")
        our_algo(n, Y1, Y2, alpha, cstar, Cbar)
        


def test_our_algo_parameter_error(): #case 11
    # empty parameter exception
    with pytest.raises(Exception):
        our_algo()

@pytest.mark.parametrize(("n", "Y1", "Y2", "alpha", "cstar", "Cbar"), # case 12
                        (
    (1000, np.ones((5, 5)), np.ones((5, 5)), '1', 1, 1), # different Y2 Y1 shape
    (1000, np.ones((10, 10)), np.ones((10, 10)), 0, '1', 1), # invalid alpha
    (1000, np.ones((10, 10)), np.ones((10, 10)), 1, 1, '0'), # invalid Cbar

))
def test_our_algo_type_error(n, Y1, Y2, alpha, cstar, Cbar):
    with pytest.raises(TypeError):
        our_algo(n, Y1, Y2, alpha, cstar, Cbar)



def test_loss_calculation_type():
    error_low_rank, error_empirical = loss_calculation()
    assert type(error_low_rank) == np.ndarray
    assert error_low_rank.dtype == float




