"""
Low-rank probability matrix estimator for discrete distributions

Authors : Laurène DAVID and Shreshtha SHAURYA
"""

import numpy as np
from scipy.stats.contingency import crosstab


class Discrete:
    """Low-rank Bivariate Discrete Probability Estimation

    Parameters
    ----------
    alpha : float, default 1
    Level of precision of matrix estimation


    Attributes
    ----------
    probability_matrix : np.ndarray of shape (d1,d2) with d1 : nbr of attributes of X1, d2 : nbr of attributes of X2
    Estimated probability matrix of the joint distribution of the data X=(X1,X2)


    Example
    ----------
    >>> from lowrankdensity.datasets import generate_lowrank_discrete
    >>> from lowrankdensity.models import Discrete

    # Generate low rank discrete samples
    >>> X = generate_lowrank_discrete()

    # Fit samples to the low rank Discrete model
    >>> model = Discrete(alpha=0.1)
    >>> model.fit(X)

    # Get the estimated probability matrix P
    >>> model.probability_matrix

    # Generate new samples
    >>> new_samples = model.sample(n_samples=1000)

    """

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.probability_matrix = None

    def _compute_histograms(self, X):
        # Create 2 matrices of same shape for final histograms
        d1max, d2max = np.unique(X[:, 0]), np.unique(X[:, 1])
        l1max, l2max = len(d1max), len(d2max)
        Y1max, Y2max = np.zeros((l1max, l2max), dtype=int), np.zeros((l1max, l2max), dtype=int)
        dictYmax = {k: dict(zip(d, np.arange(0, l))) for k, d, l in zip(["d1", "d2"], [d1max, d2max], [l1max, l2max])}

        # Create 2 histograms by splitting data in half
        c1, Y1 = crosstab(X[:int(len(X) / 2), 0], X[:int(len(X) / 2), 1])
        c2, Y2 = crosstab(X[int(len(X) / 2):, 0], X[int(len(X) / 2):, 1])
        dictY1 = {k: dict(zip(c1[i], np.arange(0, len(c1[i])))) for k, i in zip(["d1", "d2"], range(2))}
        dictY2 = {k: dict(zip(c2[i], np.arange(0, len(c2[i])))) for k, i in zip(["d1", "d2"], range(2))}

        # Fill 2 matrices Y1max, Y2max based on Y1, Y2
        index1 = np.array(np.meshgrid(c1[0], c1[1])).T.reshape(-1, 2)
        index2 = np.array(np.meshgrid(c2[0], c2[1])).T.reshape(-1, 2)

        for (i1, j1), (i2, j2) in zip(index1, index2):
            Y1max[dictYmax["d1"][i1], dictYmax["d2"][j1]] = Y1[dictY1["d1"][i1], dictY1["d2"][j1]]
            Y2max[dictYmax["d1"][i2], dictYmax["d2"][j2]] = Y2[dictY2["d1"][i2], dictY2["d2"][j2]]

        # Return the padded histograms Y1max and Y2max
        self.keys = tuple([d1max, d2max])
        return Y1max, Y2max

    def _compute_matrix(self, X=None, n=None, Y1=None, Y2=None, discrete=True):
        # Compute histograms in the discrete case
        if discrete:
            n = X.shape[0]
            Y1, Y2 = self._compute_histograms(X)

        Y1 = Y1 / np.sum(Y1)
        Y2 = Y2 / np.sum(Y2)
        d = np.max(np.shape(Y1))

        if (n <= d * np.log(d)):
            return (Y1 + Y2) / 2

        p, q = np.sum(Y1, axis=1), np.sum(Y1, axis=0)
        res = np.zeros(np.shape(Y1))
        T = int(np.log(d) / np.log(2))

        for t in range(T + 1):
            if (t < T):
                I = np.argwhere((p <= 2**(-t)) & (p > 2**(-t - 1)))
            else:
                I = np.argwhere((p <= 2**(-t)))

            if len(I) > 0:
                for u in range(T + 1):
                    if (u < T):
                        J = np.argwhere((q <= 2**(-u)) & (q > 2**(-u - 1)))
                    else:
                        J = np.argwhere(q <= 2**(-u))

                    if len(J) > 0:
                        M = np.zeros((len(I), len(J)))
                        row_id = np.zeros(Y2.shape[0], dtype=bool)
                        row_id[I] = True
                        col_id = np.zeros(Y2.shape[1], dtype=bool)
                        col_id[J] = True

                        M = Y2[row_id, :][:, col_id]

                        if (np.sum(M) < 2 * self.alpha * np.log(d) / (n * np.log(2))):
                            for i in range(len(I)):
                                for j in range(len(J)):
                                    res[I[i], J[j]] = Y2[I[i], J[j]]

                        else:
                            tau = np.log(d) * np.sqrt(0.1 * 2**(1 - min(t, u)) / n)
                            U, s, Vh = np.linalg.svd(M)
                            l = len(s[s >= tau])
                            H = np.dot(U[:, :l] * s[:l], Vh[:l, :])

                            for i in range(len(I)):
                                for j in range(len(J)):
                                    res[I[i], J[j]] = H[i, j]

        res[res < 0.] = 0.

        if np.sum(res) == 0:
            return (Y1 + Y2) / 2

        return res / np.sum(res)

    def fit(self, X):
        """
        Fit categorical dataset to discrete probability matrix estimator

        Parameters:
        ----

        X: nd.array of size (n_samples,2)
        A numpy array with 2 categorical variables (int, float or str)


        return:
        ----
        self : object
        Returns the instance itself.

        """

        if not isinstance(X, np.ndarray):
            raise TypeError(f"Input X should be a nd.array, not a {type(X)}")

        if X.shape[0] == 0:
            raise ValueError("X is an empty array")

        if X.shape[1] != 2:
            raise ValueError(f"Input X should have shape (nb_samples,2), not (nb_samples,{X.shape[1]})")

        if self.alpha < 0:
            raise ValueError(f"alpha should be positive")

        if type(self.alpha) not in (int, float):
            raise ValueError(f"alpha should an int or float, not {type(self.alpha)}")

        self.probability_matrix = self._compute_matrix(X)
        return None

    def sample(self, n_samples=1000):
        """
        Sample discrete data from a multinomial distribution with low_rank probability matrix P


        Parameters
        -------
        n_samples : int, default=1
        Number of samples to draw from distribution


        Returns
        -------
        sample: nd.array of shape (n_samples,)
        Samples drawn from discrete distribution with probability matrix P

        """

        # Reshape probability_matrix
        P = self.probability_matrix
        p = P.flatten()

        # Sample 2D multinomial data with probability matrix
        nrow, ncol = P.shape
        samples = np.random.multinomial(n=1, pvals=p, size=n_samples).reshape((n_samples, nrow, ncol))
        samples = np.argwhere(samples == 1)[:, 1:]

        # Map values of samples to the labels of original data
        dict_d1, dict_d2 = [dict(zip(np.arange(len(k)), k)) for k in self.keys]
        samples_ = np.array([[dict_d1[i], dict_d2[j]] for i, j in samples])

        return samples_
