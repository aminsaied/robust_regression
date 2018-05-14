#!/usr/bin/env python3
"""Implements the EM-type-algorithm of Liu-Rubin.

Provides class `ECME` estimating the parameters of a one-dimensional linear
regression with heavy-tailed noise. In particular, noise is assumed to be
drawn from a t-distribution, the parameters of which are also to be estimated.

If run as a script, will compute the ECME estimates of the datasets at
directory specified in :meth:`load_data` method (see :mod:`utils` module).
"""
import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.optimize import bisect
import statsmodels.api as sm

from utils import load_data, from_dataframe

OUTPUT_FILENAME = 'results.csv'

class ECME(object):
    """Implements modified version of ECME algorithn of Liu-Rubin.
    """
    def __init__(self, y, X, compare=False, use_sigma2=False):
        """Initialize ECME algorithm with default parameters.

        Args:
            y, array: Response variable data.
            X, array: Design matrix.
            compare, bool: If True, computes M-estimates.
            use_sigma2, bool: If True, introduces sigma2 parameter.
        """
        # set data
        self.y = y                                  # response
        self.X = X                                  # covariate

        # initialize model parameters
        self.w = None                               # weights
        self.B = self.ols_estimate(y, X)            # regressors
        self.mu = np.dot(X, self.B)                 # prediction
        self.sigma2 = 1                             # variance
        self.nu = 10                                # degrees of freedom

        self.use_sigma2 = use_sigma2
        if compare:
            self._comparison()

    def fit(self, n_iterations=100):
        """Run ECME-algorithm to estimate model parameters.

        Args:
            n_iterations, int: Number of iterations to run the algorithm.
        """
        for _ in range(n_iterations):
            # E-step
            self.estimate_weights()

            # CM-step-1
            self.estimate_B()
            self.estimate_mu()

            if self.use_sigma2:
                self.estimate_sigma2()

            # CM-step-2
            self.estimate_nu()

    def estimate_nu(self):
        """One-dimensional search for nu maximising likelihood."""
        def f(nu, w):
            return 1 - digamma(nu/2.) + np.log(nu/2.) + np.mean(np.log(w)-w)

        def g(nu, w):
            adj = digamma((nu+1)/2.) - np.log((nu+1)/2.)
            return f(nu, w) + adj

        if np.sign(g(1e-100, self.w)) == np.sign(g(10, self.w)):
            self.nu = 10
        else:
            self.nu = bisect(g, a=1e-100, b=10, args=(self.w))

    def estimate_weights(self):
        """Estimate weights associated to each sample."""
        numer = (self.nu + 1) * self.sigma2
        denom = self.nu * self.sigma2
        denom += (self.y - self.mu)**2
        self.w = numer / denom

    def estimate_B(self):
        """Weighted version of the normal equations."""
        weighted_Xt = np.dot(self.X.T, np.diag(self.w.flatten()))
        weighted_XtX = np.dot(weighted_Xt, self.X)
        weighted_XtX_inv = np.linalg.inv(weighted_XtX)
        weighted_Xty = np.dot(weighted_Xt, self.y)
        self.B = np.dot(weighted_XtX_inv, weighted_Xty)

    def estimate_sigma2(self):
        """Weighted version of normal MLE for sigma2."""
        weighted_error2 = np.multiply(self.w, (self.y - self.mu)**2)
        self.sigma2 = np.mean(weighted_error2)

    def estimate_mu(self):
        """Current estimate for mu."""
        self.mu = np.dot(self.X, self.B)

    @staticmethod
    def ols_estimate(y, X):
        """Compute OLS estimate used as start point of ecme_algorithm."""
        model = sm.OLS(y, X)
        results = model.fit()
        return results.params.reshape(-1,1)

    def _comparison(self):
        """Computes M-estimates for comparison."""
        model = sm.RLM(self.y, self.X, M=sm.robust.norms.HuberT(1))
        results = model.fit()
        self.B_huber_1 = results.params.reshape(-1,1)

        model = sm.RLM(self.y, self.X, M=sm.robust.norms.HuberT(4))
        results = model.fit()
        self.B_huber_4 = results.params.reshape(-1,1)

        self.B_ols = self.ols_estimate(self.y, self.X)

if __name__ == '__main__':

    data = load_data()

    results = pd.DataFrame(columns=['filename','a','b'])

    for filename, df in data.items():

        print(filename)
        print("Compute ECME-estimate.")
        y, X = from_dataframe(df)
        model = ECME(y, X, use_sigma2=False)
        model.fit(n_iterations=100)
        b, a = model.B

        # add estimate to results
        row = {'filename':filename, 'a':float(a), 'b':float(b)}
        results = results.append(row, ignore_index=True)

    # save as .csv
    results = results.sort_values(by=['filename'])
    results.to_csv(OUTPUT_FILENAME)
