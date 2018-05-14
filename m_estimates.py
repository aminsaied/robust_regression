#!/usr/bin/env python3
"""Script computing M-estimates for heavy-tailed linear model with Huber norm.

Estimates the parameters of the five provided datasets using M-estimation.
Computes the M-estimate corresponding to the Huber norm with default tuning
parameter t = 1.345.
"""
import pandas as pd
import statsmodels.api as sm

from utils import load_data, from_dataframe, huber

OUTPUT_FILENAME = 'results/results_M.csv'

def m_estimate(df, norm, verbose=False):
    """Compute M-estimate of linear model with specified norm.

    Args:
        df, DataFrame: Contains datapoints.
        norm, statsmodels.robust.norms: M-estimation function.
        verbose, bool: Output model report.

    Return:
        Array of estimates for intercept and slope of linear model
    """
    # prepare data
    y, X = from_dataframe(df)

    # estimate params
    model = sm.RLM(y, X, M=norm)
    results = model.fit()
    estimate = results.params

    if verbose:
        print(results.summary2())

    return estimate

if __name__ == '__main__':

    data = load_data()

    results = pd.DataFrame(columns=['filename','a','b'])

    for filename, df in data.items():

        print(filename)
        print("Compute M-estimate using Huber-norm reweighting.")
        b, a = m_estimate(df, norm=huber(), verbose=True)

        # add estimate to results
        row = {'filename':filename, 'a':a, 'b':b}
        results = results.append(row, ignore_index=True)

    # save as .csv
    results = results.sort_values(by=['filename'])
    results.to_csv(OUTPUT_FILENAME)
