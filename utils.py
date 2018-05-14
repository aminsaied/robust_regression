#!/usr/bin/env python3
"""Helper functions for noisy linear regression model.

Provides methods for reading data from a csv file, unpacking a dataframe and
creating plots.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT

DATA_DIR = './'
DATA_EXT = 'csv'

def load_data(data_dir=DATA_DIR, data_ext=DATA_EXT):
    """Load data at the specified directory."""
    data = {}
    for f in os.listdir(data_dir):
        if f.endswith(data_ext):
            name = f.split('.')[0]
            df = pd.read_csv(f)
            data[name] = df
    return data

def from_dataframe(df):
    """Unpack sample data from dataframe.

    Reads the feature/response data from dataframe and prepares corresponding
    design matrix by adding a column of 1s to account for the intercept.

    Args:
        df, DataFrame: Column 'x' of features, column 'y' of responses.

    Return:
        Tuple (y, X) of response/feature arrays (resp.).
    """
    X = sm.add_constant(np.array(df['x']))
    y = np.array(df['y']).reshape(-1,1)
    return y, X

def huber(t=1.345):
    """Wrapper for HuberT norm."""
    return sm.robust.norms.HuberT(t)

def display_results(experiment, norms, nus, filename):
    """Helper method that plots summary of simulation experiment."""
    # plot the results of the experiment
    fig, axes = plt.subplots(2,2, figsize=(15.0, 10.0))

    # plot large scale overview
    _create_line_plot(experiment['a'], nus, norms, axes[0][0], 'slope')
    _create_line_plot(experiment['b'], nus, norms, axes[0][1], 'intercept')

    # plot more detailed errors exluding OLS estimate
    _create_bar_plot(experiment['a'], nus[1::2], norms, axes[1][0])
    _create_bar_plot(experiment['b'], nus[1::2], norms, axes[1][1])

    # add title and save
    plt.savefig(filename, dpi=300)

def _create_line_plot(experiment_param, nus, norms, ax, subtitle):
    """Line plot of errors."""
    for name in sorted(norms):
        errors = [experiment_param[nu][name] for nu in nus]
        ax.plot(nus, errors, label=name)

    ax.legend()
    ax.set_xticks(nus[1::2])
    ax.set_xticklabels(nus[1::2])
    ax.set_ylabel('Average error (%)', fontsize=15)
    ax.set_ylim([0,5])
    ax.set_title('Estimating {}\n'.format(subtitle), fontsize=15)

def _create_bar_plot(experiment_param, nus, norms, ax, width=0.2):
    """Grouped bar plot of errors."""
    # make pretty
    M = len(nus)
    ind = 2*np.arange(M)

    # create grouped bar plot
    i = 0
    for name in sorted(norms):
        if name == 'ols': continue
        tmp = []
        for nu in nus:
            tmp.append(experiment_param[nu][name])

        ax.bar(ind + i*width, tmp, width, label=name)
        i += 1

    ax.set_xticks((ind + width / 2)+0.2)
    ax.set_xticklabels(nus)
    ax.set_xlabel('Degrees of freedom', fontsize=15)
    ax.set_ylabel('Average error (%)', fontsize=15)
    ax.legend()

def _create_data_plot(df, norms):
    """Scatter plot of data with estimates computed by norms.

    Args:
        df, DataFrame: The samples from the noisy linear model.
        norms, dict: Mapping names (str) to norms (functions).

    Return:
        Matplotlib plot of the data with fitted models.
    """
    # plot data
    ax = df.plot.scatter('x', 'y')

    # plot estimates
    for name, norm in norms.items():
        model = sm.RLM(y, X, M=norm)
        results = model.fit()
        estimate = results.params
        _plot_model(params=estimate,
                   label=name,
                   range_=(df['x'].min(), df['x'].max()))

    # make pretty
    plt.rcParams['figure.figsize'] = (15.0, 10.0)
    ax.legend()
    plt.show()

def _plot_model(params, label, range_=None):
    """Helper funtion plottine line of fit."""
    b, a = params
    if range_ is None:
        x = np.linspace(0,1)
    else:
        u, v = range_
        x = np.linspace(u,v)
    y = a*x + b
    return plt.plot(x, y, label=label)
