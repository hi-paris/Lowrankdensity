"""
Plot 3D interactive surface plots to visualize low rank or other density estimator functions.

Author: Laurène DAVID
"""

import numpy as np 
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_3d_lowrank_density(continuous_model,x,y):
    """ 
    Plot a 3D surface plot of the low rank density estimator f(x,y)

    Parameters
    ---------
    continuous_model : estimator instance
    Fitted Low rank continuous model
    
    x : nd.array 
    The first dimension x to compute the density estimator f(x,y)

    y : nd.array
    The second dimension x to compute the density estimator f(x,y)

    
    Return 
    ---------
    display : plotly figure


    Example 
    ---------
    # Import continuous model and functions
    >>> import numpy as np
    >>> from lowrankdensity.datasets import generate_lowrank_continuous
    >>> from lowrankdensity.models import Continuous
    >>> from lowrankdenisty.viz import plot_lowrank_density

    # Generate low rank continuous samples
    >>> X = generate_lowrank_continuous()
    
    # Fit samples to the Continuous model
    >>> model = Continuous(alpha=0.1)
    >>> model.fit(X)

    # Plot the 3D surface density plot 
    >>> x, y = np.linspace(0,1,100), np.linspace(0,1,100)
    >>> plot_3d_lowrank_density(continuous_model=model,x=x,y=y)

    """

    pdf_mat = continuous_model.pdf(x,y)

    fig = go.Figure(data=[go.Surface(z=pdf_mat, x=x, y=y)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="lightblue", project_z=True))

    fig.update_layout(title_text='Low rank density estimator', autosize=False,
                    width=800, height=800,
                    margin=dict(l=65, r=50, b=65, t=90))
    fig.show()



def plot_multiple_3d_densities(continuous_model,x,y,mat_estimator2,plot2_title):
    """
    Plot two 3D surface plots of the low rank density estimator f(x,y) and another density estimator
    to compare both methods.
    

    Parameters
    ----------
    continuous_model : estimator instance
    Fitted Low rank continuous model

    x : nd.array 
    The first dimension x to compute the density estimator f(x,y)

    y : nd.array
    The second dimension x to compute the density estimator f(x,y)
    
    mat_estimator2 : nd.array of shape (len(x),len(y))
    A matrix with the values of the estimator function over x and y
    The matrix should be computed using the same x and y arguments as plot_multiple_3d_density.


    Return 
    ----------
    display : 2 subplots of Plotly figures

    
    Example 
    ----------
    # Import continuous model and functions
    >>> import numpy as np
    >>> from lowrankdensity.datasets import generate_lowrank_continuous
    >>> from lowrankdensity.models import Continuous
    >>> from lowrankdenisty.viz import plot_multiple_densities
    >>> from sklearn.neighbors import KernelDensity

    # Generate low rank continuous samples
    >>> X = generate_lowrank_continuous()
    
    # Fit samples to the Continuous model
    >>> model = Continuous(alpha=0.1)
    >>> model.fit(X)
    
    # Compute mat_estimator2 with sklearn's KDE estimator
    >>> from sklearn.neighbors import KernelDensity
    >>> kde = KernelDensity(bandwidth=(X.shape[0])**(-1/3), kernel='tophat')
    >>> kde.fit(continuous_samples)
    
    >>> density_funs_kde = lambda x,y: np.exp(kde.score_samples(np.array([[x,y]])))
    >>> x, y = np.linspace(0,1,100), np.linspace(0,1,100)
    >>> mat_kde = np.array([density_funs_kde(i,j) for i in x for j in y]).reshape((len(x),len(y)))

    # Plot both 3D surface density plots (low rank and KDE) 
    >>> plot_multiple_3d_density(continuous_model=model, x=x, y=y,mat_estimator2=mat_kde)

    """

    mat_lowrank = continuous_model.pdf(x,y)

    if (isinstance(mat_estimator2,np.ndarray) == False) or (mat_estimator2.ndim != 2):
        raise TypeError("mat_estimator2 should be a 2D numpy array (matrix)")

    if mat_estimator2.shape != (len(x),len(y)):
        raise TypeError(f"The shape of mat_estimator should be (len(x),len(y)), not {mat_estimator2.shape}")
    
    cmin = int(min(np.min(mat_lowrank),np.min(mat_estimator2)))
    cmax = int(max(np.max(mat_lowrank),np.max(mat_estimator2)))

    fig = make_subplots(rows=1, cols=2,
                    specs=[[{'is_3d': True}, {'is_3d': True}]],
                    subplot_titles=['Low rank estimator', plot2_title]
                    )

    fig.add_trace(go.Surface(x=x, y=y, z=mat_lowrank,colorbar_x=-0.07, cmin=cmin, cmax=cmax), 1, 1)
    fig.add_trace(go.Surface(x=x, y=y, z=mat_estimator2, cmin=cmin, cmax=cmax), 1, 2)
    fig.update_layout(width=1200, height=700)
    fig.show()