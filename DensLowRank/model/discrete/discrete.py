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
        self.alpha = alpha 
        self.P = None

        

    def _compute_histograms(self,X):

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

        # create histogram with the second half of the data
        data_2 = X[split_index:,:]
        _, Y2 = crosstab(data_2[:,0], data_2[:,1])


        return Y1/len(data_1), Y2/len(data_2)
    



    def _compute_matrix(self,X=None,Y1=None,Y2=None,n=None):

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
        
        # Model constants 
        d = np.max(np.shape(Y1))
        cstar = self.alpha/10 
        Cbar = self.Cbar

        # Case 1
        if (n <= d * np.log(d)):
            self.P = (Y1 + Y2) / 2
        
        # Case 2
        p = np.sum(Y1, axis=1)
        q = np.sum(Y1, axis=0)
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
                    tau = np.log(d) * np.sqrt(cstar * 2**(1 - min(t, u)) / n)
                    U, s, Vh = np.linalg.svd(M)
                    l = len(s[s >= tau])
                    H = np.dot(U[:, :l] * s[:l], Vh[:l, :])
                    
                    for i in range(len(I)):  # +2
                        for j in range(len(J)):
                            res[I[i], J[j]] = H[i, j]
        
        return res/np.sum(res)


    
    def fit(self,X=None,Y1=None,Y2=None,n=None,continuous_case=False):
        # X, Y1, Y2, n = None for when Discrete model is used individually or used in Continuous model

        '''
        Fit categorical dataset to discrete probability matrix estimator

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
        

        # Check TypeError/ValueError on alpha 
        if self.alpha < 0:
            raise ValueError(f"alpha should be positive")
        
        if type(self.alpha) not in (int,float):
            raise ValueError(f"alpha should an int or float, not {type(self.alpha)}")


        # Seperate case where discrete function used in continuous function
        if continuous_case == False:
            n = X.shape[0]
            Y1, Y2 = self._compute_histograms(X)
            self.P = self._compute_matrix(n=n,Y1=Y1,Y2=Y2)
        
        else:
            self.P = self._compute_matrix()
        
        return self
    



    def probability_matrix(self):
        """
        Class method to return Probability matrix 

        Parameters 
        -------   


        Returns
        -------
        sample: array-like of shape (n_samples, n_features)
            List of samples.
        """
        
        P = self.P        
        return P

    


## Test of function with dataset ##

# Kaggle dataset: https://www.kaggle.com/datasets/jasleensondhi/hair-eye-color
# df = pd.read_csv(r"C:\Users\LaurèneDAVID\Documents\Projects\Dimension_Reduction\HairEyeColor.csv")
# X = df[["Hair","Eye"]].to_numpy()

# model = Discrete(alpha=0.01)
# model.fit(X)
# P = model.probability_matrix()

# print(P)

