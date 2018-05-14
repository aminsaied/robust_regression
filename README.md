# Robust Regression
Implements techniques used to predict the parameters from the linear models in which the residuals are heavy-tailed: we make the assumption that they are drawn from a t-distribution with possibly unknown degrees of freedom and scaling-parameter.

The main content of this library is the implementation of the ECME algorithm of Liu-Rubin for robust linear regression models. This technique estimates not only the coefficients of the linear model, but also the degrees of freedom and scaling-parameter of the residual's t-distribution. In addition, Huber's M-estimates are implemented as an alternative robust method.

Here is the structure of this library:

- `em_algorithm`: Implements the ECME algorithm (based on Liu-Rubin's more general algorithm).
- `m_estimates`: Computes the M-estimates with the Huber norm. This is an alternative robust method for parameter estimation.
- `simulation`: Runs simulations comparing the various estimation techniques.
- `utils`: Provides helper methods used in the other modules.

See the accompanying report `report/heavy_tailed_residuals.pdf` for more details.
