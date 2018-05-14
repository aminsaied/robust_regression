# Robust Regression
The main content of this library is the implementation of the ECME algorithm of Liu-Rubin for robust linear regression models. This technique is used to predict the parameters from the data provided. In addition, Huber's M-estimates are implemented as an alternative robust method.

Here is the structure of this library:

- `em_algorithm`: Implements the ECME algorithm used to predict the parameters of the five provided samples. This is the technique selected in the final analysis.
- `m_estimates`: Computes the M-estimates with the Huber norm. This is an alternative robust method for parameter estimation. This is not used in the final analysis.
- `simulation`: Runs simulations comparing the various estimation techniques.
- `utils`: Provides helper methods used in the other modules.
