#!/usr/bin/env python3
"""Simulate robust estimation of linear model with heavy-tailed residuals.

Script running simulation of various techniques (either M-estimates or the
ECME algorithm) used to estimate the parameters of a one-dimensional linear
model with t-distributed residuals.

Experiment 1 - compares a family of Huber norms across a range of simulations
with various degrees of freedom. The OLS estimate is computed, demonstrating
its susceptability to outliers.

Experiment 2 - compares estimates obtained from the EMCE algorithm against
M-estimates with Huber norms corresponding to tuning parameters 1 and 4.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.robust.norms import LeastSquares

from utils import huber

def simulate_data(N, intercept, slope, nu, sigma2=1, seed=None):
    """Simulate noisy linear model with t-distributed residuals.

    Generates `N` samples from a one-dimensional linear regression with
    residuals drawn from a t-distribution with `nu` degrees of freedom, and
    scaling-parameter `sigma2`. The true parameters of the linear model are
    specified by the `intercept` and `slope` parameters.

    Args:
        N, int: Number of samples.
        intercept, float: The intercept of the linear model.
        slope, float: The slope of the linear model.
        nu, float (>0): The degrees of freedom of the t-distribution.
        sigma2, float (>0): The scale-parameter of the t-distribution.
        seed, int: Set random seed for repeatability.

    Return:
        DataFrame containing N samples from noisy linear model.
    """
    np.random.seed(seed)

    # x ~ Uniform(0,1)
    interval = np.linspace(0,1, num=2*N)
    sample = np.random.choice(interval, size=N, replace=False)
    df = pd.DataFrame({"x": sample})

    # generate y values using linear model
    linear_map = lambda x: intercept + slope*x
    df['y'] = linear_map(df['x']) + sigma2*np.random.standard_t(nu, N)

    return df

def norm_comparison(norms, nus, n_reps=100):
    """Experiment comparing norms over a range of degrees of freedom.

    We compare the estimates for a noisy linear model using M-estimation
    with provided `norms`.

    Args:
        norms, dict: Mapping names (str) to norms (functions).
        nus, iter: Iterator of values for the degrees of freedom.
        n_reps, int (default 100): Number of times experiment is repeated.

    Return:
        Results of the simulation recording average percentage errors.
    """
    errors = { name : {'a':[], 'b':[]} for name in norms}

    for nu in nus:

        tmp_errors = { name : {'a':[], 'b':[]} for name in norms}

        # repeat experiment
        for _ in range(n_reps):

                # generate random data
                a, b = 10*np.random.randn(), 10*np.random.randn()
                data = simulate_data(100, b, a, nu)
                X = sm.add_constant(np.array(data['x']))
                y = np.array(data['y'])

                for name, norm in sorted(norms.items()):

                    # estimate params
                    model = sm.RLM(y, X, M=norm)
                    results = model.fit()
                    b_, a_ = results.params

                    # record percentage errors
                    tmp_errors[name]['a'].append(np.abs((a - a_)/a))
                    tmp_errors[name]['b'].append(np.abs((b - b_)/b))

        # compute average errors
        for name in errors:
            for coeff in errors[name]:
                errors[name][coeff].append(np.mean(tmp_errors[name][coeff]))

    return errors

def emce_comparison(nus, n_reps=100):
    """Simulation comparing ECME algorithm with M-estimates.

    We compare the estimates obtained by the ECME algorithm against two Huber
    M-estimates with tuning parameters 1 and 4.

    Args:
        nus, iter: Iterator of values for the degrees of freedom.
        n_reps, int (default 100): Number of times experiment is repeated.

    Return:
        Results of the simulation recording average percentage errors.
    """
    models = ['ecme', 'huber1', 'huber4']
    errors = { model : {'a':[], 'b':[]} for model in models}

    for nu in nus:

        tmp_errors = { model : {'a':[], 'b':[]} for model in models}

        for _ in range(n_reps):
            a = 10*np.random.randn()
            b = 10*np.random.randn()
            sigma2 = 2*np.random.rand()
            df = simulation.simulate_data(100, b, a, nu, sigma2)

            y, X = from_dataframe(df)

            model = ECME(y, X, compare=True, use_sigma2=True)
            model.fit()

            # slope
            tmp_errors['ecme']['b'].append(np.abs((model.B[0]-b)/b))
            tmp_errors['huber1']['b'].append(np.abs((model.B_huber_1[0]-b)/b))
            tmp_errors['huber4']['b'].append(np.abs((model.B_huber_4[0]-b)/b))

            # intercept
            tmp_errors['ecme']['a'].append(abs((model.B[1] - a)/a))
            tmp_errors['huber1']['a'].append(np.abs((model.B_huber_1[1]-a)/a))
            tmp_errors['huber4']['a'].append(np.abs((model.B_huber_4[1]-a)/a))

        # compute average errors
        for name in errors:
            for coeff in errors[name]:
                errors[name][coeff].append(np.mean(tmp_errors[name][coeff]))

    return errors

if __name__ == "__main__":

    # range of values for degrees of freedom
    nus = np.arange(20)/2 + 0.5

    # experiment 1 - compare tuning parameters for Huber norm
    norms = {}
    for t in range(1,5):
        name = 'huber-{}'.format(t)
        norms[name] = huber(t)
    norms['ols'] = LeastSquares()

    experiment1 = norm_comparison(norms, nus)

    # create and save figure
    display_results(experiment1, norms, nus, filename='experiment_1.png')

    # experiment 2 - compare ECME algorithm with M-estimates
    experiment2 = emce_comparison(nus)

    # create and save figure
    display_results(experiment2, norms, nus, filename='experiment_2.png')
