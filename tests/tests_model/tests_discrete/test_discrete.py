import numpy as np
import pytest

from DensLowRank.model.discrete.discrete import Discrete
from DensLowRank.dataset.load_data import load_data
from scipy.stats.contingency import crosstab

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



def test_alpha_negative(): #case 1
    with pytest.raises(ValueError):
        Discrete(alpha='0.1')



@pytest.mark.parametrize(("n", "Y1", "Y2","expected_shape"), #case 2 type + shape
                        (
    (100, np.ones((100, 2)), np.ones((100, 2)),(100, 2)), # simple case
    (50, np.ones((50, 2)), np.ones((50, 2)),(50, 2)), # edge case
    (1000, np.ones((15, 2)), np.ones((15, 2)), (5, 2)), #  error case
))
def test__compute_matrix_shape_for_res(n, Y1, Y2,expected_shape):
    try: 
        df = load_data()
        model = Discrete()
        #X = np.ones((100, 2))
        #Y1 = crosstab(X[:int(len(X)/2),0], X[:int(len(X)/2),1])[1]
        #Y2 = crosstab(X[int(len(X)/2):,0], X[int(len(X)/2):,1])[1]
        #n=X.shape[0]
        res = model._compute_matrix(n=n, Y1=Y1, Y2=Y2)
        assert res.shape == expected_shape
        assert res.dtype == float
    except:
        print(f"Assertion error: n={n}, Y1.shape={Y1.shape}, Y2.shape={Y2.shape}, expected_shape={expected_shape}")
        raise

    

def test__compute_matrix_shape_with_load_data(): #case 3
    try: 
        df = load_data()
        model = Discrete()
        X = df[['petal.length', 'sepal.length']].to_numpy()
        Y1 = crosstab(X[:int(len(X)/2),0], X[:int(len(X)/2),1])[1]
        Y2 = crosstab(X[int(len(X)/2):,0], X[int(len(X)/2):,1])[1]
        n=X.shape[0]
        res = model._compute_matrix(n=n, Y1=Y1, Y2=Y2)
        assert res.shape == Y1.shape
        assert res.dtype == float
    except:
        print(f"Assertion error: n={n}, Y1.shape={Y1.shape}, Y2.shape={Y2.shape}, expected_shape={expected_shape}")
        raise

@pytest.mark.parametrize(("n", "Y1", "Y2"), # case 4
                        (
    (1000, np.ones((10, 0)), np.ones((0, 10))), # invalid shapes
    (1000, np.ones((10, 3)), np.ones((10, 2)))
))
def test__compute_matrix_shape_error(n, Y1, Y2): 
    # invalid shape errors
    with pytest.raises(Exception):
    #    pytest.fail("unexpected shape")
        model = Discrete()
        res = model._compute_matrix(n=n, Y1=Y1, Y2=Y2)


def test__compute_matrix_parameter_error(): #case 5
    # empty parameter exception
    model = Discrete()
    with pytest.raises(Exception):
        model._compute_matrix()
        #model.fit() ask this


def test_fit_empty_input_array(): # case 6
    discrete = Discrete(alpha=0.1)
    with pytest.raises(ValueError):
        discrete.fit(X=np.array([]), discrete_case=True)

@pytest.mark.parametrize(( "X"), # case 7
                        (
    (np.ones((10, 0))), # invalid shapes
    (np.ones((10, 3))),
    (np.ones((10, 10))
)))
def test_fit_input_array_shape_error(X): 
    discrete = Discrete(alpha=0.1, discrete=True) #ask this discrete true
    with pytest.raises(ValueError):
        discrete.fit(X)




