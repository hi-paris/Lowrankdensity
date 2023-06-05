"""
Plot a 2D interactive histogram for low rank samples generated with Discrete or Continuous models.

Author: Laurène DAVID
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_2d_histogram(samples, 
                      text_auto=False, histnorm=None, 
                      marginal_x="histogram", marginal_y="histogram", 
                      var_names=["var1","var2"]):
    """

    Plot a 2D interactive histogram of samples generated with lowrankdensity's Discrete or Continuous model.
    The plot shows a 2D histogram as well as marginal 1D histograms for each dimension.
    
    The goal of this visualization is to study the distribution of 2D samples generated 
    with a low rank discrete or continuous density estimator.

    This plot was generated using plotly's density_heatmap() function. 
    For more information on plot_2d_histogram's arguments, here is a link to plotly's API:
    https://plotly.com/python-api-reference/generated/plotly.express.density_heatmap.html
    

    Parameters 
    ----------
    samples : nd.array of shape (n_samples,2)
    Samples used to plot 2D histogram 

    text_auto : bool (True or False), default=False 
    If True, show var1 and var2 as well as count result.
    If discrete samples are used, the plot will show the exact values of var1 and var2
    If continuous samples are used, the plot will show a range of values for var1 and var2

    histnorm : str, default=None
    Normalization used on the grid values of the 2D histogram. 
    One of 'percent', 'probability', 'density', 'probability density' or None

    marginal_x : str, default="histogram"
    Type of visualization used to represent marginal 1D distribution
    One of 'rug', 'box', 'violin', 'histogram', None
    If set, a horizontal subplot is drawn above the main plot, visualizing the x-distribution.
    If None, no marginal plot will appear for x

    marginal_y : str, default="histogram"
    Type of visualization used to represent marginal 1D distribution
    One of 'rug', 'box', 'violin', 'histogram', None
    If set, a horizontal subplot is drawn above the main plot, visualizing the x-distribution.
    If None, no marginal plot will appear for y

    var_names : list of str, default=["var1", "var2"]
    Variable names used for the plot's labels/legend

    
    Return
    ----------
    display : plotly figure


    Example 
    ---------
    # Import discrete model and functions
    >>> from lowrankdensity.datasets import generate_lowrank_discrete
    >>> from lowrankdensity.models import Discrete
    >>> from lowrankdenisty.viz import plot_2d_histogram

    # Generate low rank discrete samples
    >>> X = generate_lowrank_discrete()
    
    # Compute new samples with the discrete lowrank model
    >>> model = Discrete(alpha=0.1)
    >>> model.fit(X)
    >>> new_samples = model.sample(n_samples=1000)

    # Plot a 2D histogram of newly generated samples 
    >>> plot_2d_histogram(samples=new_samples)

    
    """
    if samples.shape[1] != 2:
        ValueError("This function can only plot the distribution of 2D arrays")

    if (isinstance(samples,np.ndarray) == False) or (isinstance(samples,np.ndarray) == False):
        TypeError("Samples should be numpy arrays")

    if len(var_names) != 2:
        ValueError("The length of var_names should be equal to 2")

    x,y = samples[:,0], samples[:,1]
    labels={"x": var_names[0], "y": var_names[1]}
    
    fig = px.density_heatmap(x=x, y=y,
                             text_auto=text_auto,
                             histnorm=histnorm,
                             marginal_x=marginal_x, 
                             marginal_y=marginal_y,
                             labels=labels)
    fig.update_layout(title_text='2D histogram', autosize=False, width=1000, height=800)
    fig.show()



def plot_2d_contour_histogram(samples, histnorm=None, var_names=["var1","var2"]):
    """
    Plot a 2D interactive contour histogram of samples generated with lowrankdensity's Discrete or Continuous model.
    
    The goal of this visualization is to study the distribution of 2D samples generated 
    with a low rank discrete or continuous density estimator.

    This plot was generated using plotly's density_heatmap() function. 
    For more information on plot_2d_histogram's arguments, here is a link to plotly's API:
    https://plotly.com/python-api-reference/generated/plotly.express.density_heatmap.html
    

    Parameters 
    ----------
    samples : nd.array of shape (n_samples,2)
    Samples used to plot 2D histogram 

    histnorm : str, default None
    Normalization used on the grid values of the 2D histogram. 
    One of 'percent', 'probability', 'density', 'probability density' or None

    var_names : list of str, default ["var1", "var2"]
    Variable names used for the plot's labels/legend

    
    Return
    ----------
    display : plotly figure


    Example 
    ---------
    # Import discrete model and functions
    >>> from lowrankdensity.datasets import generate_lowrank_discrete
    >>> from lowrankdensity.models import Discrete
    >>> from lowrankdenisty.viz import plot_2d_histogram

    # Generate low rank discrete samples
    >>> X = generate_lowrank_discrete()
    
    # Compute new samples with the discrete lowrank model
    >>> model = Discrete(alpha=0.1)
    >>> model.fit(X)
    >>> new_samples = model.sample(n_samples=1000)

    # Plot a 2D histogram of newly generated samples 
    >>> plot_2d_contour_histogram(samples=new_samples)
    
    """

    if samples.shape[1] != 2:
        ValueError("This function can only plot the distribution of 2D arrays")

    if (isinstance(samples,np.ndarray) == False) or (isinstance(samples,np.ndarray) == False):
        TypeError("Samples should be numpy arrays")

    if len(var_names) != 2:
        ValueError("The length of var_names should be equal to 2")

    x,y = samples[:,0], samples[:,1]
    labels={"x": var_names[0], "y": var_names[1]}
    
    fig = px.density_contour(x=x, y=y, labels=labels, histnorm=histnorm)
    fig.update_layout(title_text='2D contour histogram plot', autosize=False, width=800, height=600)
    fig.update_traces(contours_coloring="fill", contours_showlabels = True, line=dict(color="Black"))
    fig.show()