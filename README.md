# Efficient tensor implementation of vector autoregressive (VAR) model in Python
The VAR model of order $P\geq 1$ has the form:

$$y_t = A_1 y_{t-1} + \ldots + A_P y_{t-P} + \varepsilon_t$$,

where the $y_t's are observed time series with $y_t\in\mathbb{R}^N$ and the $\varepsilon_t$'s are $N$-dimensional, centred and with finite variance, independent and identically distributed variables (called innovations). 
The $A_t$'s matrices are $N\times N$ transition matrices we want to estimate. The estimation process is both time and memory expensive. 
This package aims to implement the VAR model in a more efficient way by using tensor decomposition described in [@wang2020highdimensional].

