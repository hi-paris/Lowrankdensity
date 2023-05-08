"""
Generate sample data with a discrete distribution and continuous distribution
"""

import warnings
import numpy as np 
import math 
import pandas as pd
from os.path import join



def generate_lowrank_discrete(n_samples=1000,K=2,d1=50,d2=50):
    """
    Generate 2D discrete samples from a low rank probability matrix.
    
    This function starts by computing a low rank matrix 'P' by sampling a uniform distribution.
    Samples are then drawn from a multinomial discrete distribution using 'P' as the probability matrix.


    Parameters 
    ----------
    n_samples : int, default = 1000
    Number of discrete 2D samples to generate

    K : int, default = 2
    Rank of the probability matrix  

    d1 : int, default = 50
    Number of classes in the first discrete variable
    
    d2 : int, default = 50
    Number of classes in the second discrete variable 


    Return 
    --------
    samples : nd.array of shape (n_samples,2)
    List of samples
    
    """

    if n_samples > 50000:
        warnings.warn("Choosing n_samples > 50000 might slow down data generation.")

    # if generate_lowrank_discrete.__default__ == tuple(locals().values()):
    #     pass

    u, v = np.random.rand(d1,K), np.random.rand(d2,K)
    res = (1/K)*(u @ v.T)
    P = res/np.sum(res)
    p = np.reshape(P,d1*d2)
    
    samples = np.random.multinomial(n_samples,p,size=2).T 
    # ?? n != n_samples in np.random.multinomial

    return samples




def generate_lowrank_continuous(n_samples=1000,K=2):
    """
    Generate 2D low rank continuous samples. 

    Sample are drawn from a joint continuous distribution whose density function is defined
    as a sum of K products between two beta density functions f_k,g_k with k=1,...,K.
    

    Parameters 
    ----------
    n_samples : int, default = 1000
    Number of samples to generate

    K : int, default = 2
    Rank of the samples 

    Return 
    --------
    samples : nd.array of shape (n_samples,2)
    List of samples
    
    """

    # if generate_lowrank_discrete.__default__ == tuple(locals().values()):
    #     path = join("",'datasets/samples_continuous.csv')
    #     samples = pd.read_csv(path).to_numpy()
    #     return samples

    if n_samples > 50000:
        warnings.warn("Choosing n_samples > 50000 might slow down data generation.")

    a1, b1 = np.linspace(1,10,K), np.linspace(3,10,K)
    a2, b2 = np.linspace(2,15,K), np.linspace(4,25,K) 
    
    f = np.array([np.random.beta(a=a,b=b,size=n_samples) for a,b in zip(a1,b1)]).T
    g = np.array([np.random.beta(a=a,b=b,size=n_samples) for a,b in zip(a2,b2)]).T
    
    u = np.random.randint(low=0,high=K-1,size=n_samples)
    samples = np.array([[f[i,j],g[i,j]] for i,j in enumerate(u)])
        
    return samples
    

