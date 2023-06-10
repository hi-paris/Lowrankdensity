"""
Generate low rank samples for discrete and continuous distributions

Author: Laurène DAVID
"""

import warnings
import numpy as np 


def generate_lowrank_discrete(n_samples=5000,K=2,d1=10,d2=10):
    """
    Generate 2D discrete samples from a low rank probability matrix.
    
    This function starts by computing a low rank matrix 'P' by sampling a uniform distribution.
    Samples are then drawn from a multinomial discrete distribution using 'P' as the probability matrix.


    Parameters 
    ----------
    n_samples : int, default 5000
    Number of discrete 2D samples to generate

    K : int, default 2
    Rank of the probability matrix used to generate multinomial samples 

    d1 : int, default 10
    Number of classes in the first discrete variable
    
    d2 : int, default 10
    Number of classes in the second discrete variable 


    Return 
    --------
    samples : nd.array of shape (n_samples,2)
    List of samples
    
    """
    if type(n_samples)!= int:
        raise TypeError(f"n_samples should be an int value, not {type(n_samples)}")
    
    if n_samples < 0:
        raise ValueError("n_samples can only take positive values")

    if (type(d1)!= int) or (type(d2)!= int):
        raise TypeError("d1 and d2 can only take int values") 

    if ((d1 > 100) or (d1 < 2)) or ((d2 > 100) or (d2 < 2)):
        raise ValueError("The number of classes d1 and d2 should be between 2 and 100")
    
    if K > n_samples**(1/4):
        warnings.warn("The low rank estimator won't perform better than a classic histogram estimator for samples generated with a rank K > n_samples**(1/4)") 


    # Compute a low rank probability matrix
    u, v = np.random.rand(d1,K), np.random.rand(d2,K)
    res = (1/K)*(u @ v.T)
    P = res/np.sum(res) 

    # Reshape matrix and sample multinomial distribution with it
    p = P.flatten()
    samples = np.random.multinomial(1,p,size=n_samples).reshape((n_samples,d1,d2))
    samples = np.argwhere(samples==1)[:,1:]
    
    return samples




def generate_lowrank_continuous(n_samples=5000,K=2):
    """
    Generate 2D low rank continuous samples. 

    Samples are drawn from a joint continuous distribution with density f.
    The density function f is defined as a sum of K products between two beta density functions f_k*g_k with k=1,...,K.
    Each f_1,...,f_K,g_1,...,g_K is a beta distribution.
    

    Parameters 
    ----------
    n_samples : int, default=5000
    Number of samples to generate

    K : int, default=2 
    Rank of the distribution used to generate samples 

    Return 
    --------
    samples : nd.array of shape (n_samples,2)
    List of samples
    
    """
    if type(n_samples)!= int:
        raise TypeError(f"n_samples should be an integer")
    
    if n_samples < 0:
        raise ValueError("n_samples can only take positive values")
    
    if K <= 0:
        raise ValueError("The rank of probability matrix P used to generate data can only take positive")
    
    if type(K) != int:
        raise TypeError("The rank of probability matrix P used to generate data should be an integer")

    if K > n_samples**(1/4):
        warnings.warn("The low rank estimator won't perform better than a classic histogram estimator for samples generated with a rank K > n_samples**(1/4)") 


    a1, b1 = np.linspace(1,10,K), np.linspace(3,10,K)
    a2, b2 = np.linspace(2,15,K), np.linspace(4,25,K) 
    
    f = np.array([np.random.beta(a=a,b=b,size=n_samples) for a,b in zip(a1,b1)]).T
    g = np.array([np.random.beta(a=a,b=b,size=n_samples) for a,b in zip(a2,b2)]).T
    
    u = np.random.randint(low=0,high=K,size=n_samples)
    samples = np.array([[f[i,j],g[i,j]] for i,j in enumerate(u)])
        
    return samples


