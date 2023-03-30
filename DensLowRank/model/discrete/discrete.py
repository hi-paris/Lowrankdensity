"""
Low-rank probability matrix estimator for discrete distributions

Author : Laurène David
"""

import numpy as np
import pandas as pd
from scipy.stats.contingency import crosstab



class Discrete:

    Cbar = 1 # class variable

    def __init__(self,alpha=0.1):

        """ 
        alpha: float, default = 0.1 
        A tuning parameter, which gives the level of precision of the estimation.
        Should be a positive value 

        """

        if alpha < 0:
            raise ValueError(f"alpha should be positive")
        
        if type(alpha) not in (int,float):
            raise ValueError(f"alpha should an int or float, not {type(alpha)}")
        
        self.alpha = alpha 

        

    def create_histograms(self,X):

        """
        Compute two histograms from dataset with two categorical/discrete variables.
        Each histogram is a two-way frequency table of both variables's attributes
        and is computed with one half of the data. 

        return : 
        ----
        Y1: nd.array
        A normalized histogram of the first half of the data.

        Y2: nd.array
        A normalized histogram of the second half of the data.

        """

        split_index = int(len(X)/2)

        # create histogram with the first half of the data
        data_1 = X[:split_index,:]
        _, Y1 = crosstab(data_1[:,0], data_1[:,1])
        Y1 = Y1/len(data_1)

        # create histogram with the second half of the data
        data_2 = X[split_index:,:]
        _, Y2 = crosstab(data_2[:,0], data_2[:,1])
        Y2 = Y2/len(data_2)


        return Y1, Y2
    

    
    def fit(self,X=None,Y1=None,Y2=None,n=None,continuous_case=False):
        # X, Y1, Y2, n = None for when Discrete model is used individually or used in Continuous model

        '''
        Compute an matrix estimation of the joint multinomial probability of categorical data with two varibales.

        Parameters:
        ----

        X: nd.array of size (nb_samples,2) 
        A numpy array with 2 categorical columns (int, float or str)

        
        return:
        ----
        res/np.sum(res): numpy.ndarray
        The estimated probability matrix of a multinomial distribution

        '''

        # Check TypeError/ValueError of model input 
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Input X should be a nd.array, not a {type(X)}")
        
        if X.shape[0] == 0:
            raise ValueError("X is an empty array")
        
        if X.shape[1] != 2:
            raise ValueError(f"Input X should have shape (nb_samples,2), not (nb_samples,{X.shape[1]})") 



        # Seperate case where discrete function used in continuous case
        # (In continuous_case, n, Y1, Y2 will be defined in arguments of function) 
        if continuous_case == False:
            n = X.shape[0]
            Y1, Y2 = self.create_histograms(X)
        
        # Model constants 
        d = np.max(np.shape(Y1))
        Cbar = 1
        cstar = self.alpha/10 
        

        # Specific case 
        if (n <= d * np.log(d)):
            return ((Y1 + Y2) / 2)
        

        # Compute p, q variables with the first histogram Y1
        p = np.sum(Y1, axis=1)
        q = np.sum(Y1, axis=0)

        # Estimate probability matrix P with histogram Y2 and p,q
        res = np.zeros(np.shape(Y1))
        T = int(np.log(d) / np.log(2))

        for t in range(T + 1):
            if (t < T):
                I = np.argwhere((p <= 2**(-t)) & (p > 2**(-t - 1)))
            else:
                I = np.argwhere((p <= 2**(-t)))

            for u in range(T + 1):
                if (u < T):
                    J = np.argwhere((q <= 2**(-u)) & (q > 2**(-u - 1)))
                else:
                    J = np.argwhere(q <= 2**(-u))

                M = np.zeros((len(I), len(J)))

                row_indices = np.zeros(Y2.shape[0], dtype=bool)
                row_indices[I] = True
                col_indices = np.zeros(Y2.shape[1], dtype=bool)
                col_indices[J] = True
                M = Y2[row_indices, :][:, col_indices]

                if (np.sum(M) < 2 * Cbar * self.alpha * np.log(d) / (n * np.log(2))):
                    for i in range(len(I)):  # +1
                        for j in range(len(J)):
                            res[I[i], J[j]] = Y2[I[i], J[j]]

                else:
                    # Truncated SVD with threshold tau for singular value
                    tau = np.log(d) * np.sqrt(cstar * 2**(1 - min(t, u)) / n)
                    U, s, Vh = np.linalg.svd(M)
                    l = len(s[s >= tau])
                    H = np.dot(U[:, :l] * s[:l], Vh[:l, :])
                    
                    for i in range(len(I)):  # +2
                        for j in range(len(J)):
                            res[I[i], J[j]] = H[i, j]

        return (res / np.sum(res))
    




    # def sample(self, n_samples=1, random_state=None):
    #     """Generate random multinomial samples from probability matrix P

    #     Parameters 
    #     ----------
    #     n_samples : int, default=1
    #         Number of samples to generate 

    #     random_state : int, RandomState instance or None, default=None
    #         Determines random number generation used to generate
    #         random samples. Pass an int for reproducible results
    #         across multiple function calls.     


    #     Returns
    #     -------
    #     X : array-like of shape (n_samples, n_features)
    #         List of samples.
    #     """

    #     P = self.fit()

    #     return np.random.multinomial()

    


## Test of function with dataset ##

# Kaggle dataset: https://www.kaggle.com/datasets/jasleensondhi/hair-eye-color
# df = pd.read_csv("HairEyeColor.csv")
# X = df[["Hair","Eye"]].to_numpy()
# P_hat = Discrete(alpha=0.01).fit(X)
# print(P_hat)

