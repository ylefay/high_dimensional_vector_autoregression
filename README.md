# Efficient Tensor Implementation of Vector Autoregression (VAR) Model in JAX

This project provides an efficient implementation of the **Vector Autoregression (VAR)** model using tensor decomposition techniques. The VAR model is widely used in time series analysis to capture linear interdependencies among multiple time series.

The VAR model of order $P \geq 1$ is defined as:

$$
y_t = A_1 y_{t-1} + \ldots + A_P y_{t-P} + \varepsilon_t
$$

where:
- $y_t \in \mathbb{R}^N$ are observed time series,
- $\varepsilon_t$ are $N$-dimensional innovations (centered, finite variance, i.i.d.),
- $A_t$ are $N \times N$ transition matrices to be estimated.

However, in a high-dimensional setting, whether it’s in $N$ or in $P$, the number of interest variables grows like $N^2P$, thus making the regression procedure both computationally and memory-expensive. [Wang et al. (2020)](https://arxiv.org/abs/1909.06624) introduce a rearrangement of the VAR model which leverages a tensor decomposition technique to cast the regression problem into a lower dimensional space, critically depending on the structure of the $A_i$, i.e., their sparsity and their spanned vector spaces.

Traditional estimation of VAR models can be computationally expensive in terms of both time and memory. This package addresses these challenges by leveraging **tensor decomposition** methods, as described in [Wang et al. (2020)](https://arxiv.org/abs/1909.06624).

---

## Why is this useful?

Vector Autoregression (VAR) models are essential tools for analyzing multivariate time series data. They are widely used in various industries, including:

### Finance:
- **Portfolio Optimization:** Model the relationships between asset returns to optimize investment strategies.
- **Risk Management:** Forecast financial risks by analyzing interdependencies between economic indicators.
- **Macroeconomic Forecasting:** Predict economic variables such as GDP, inflation, and interest rates.

### Pharmaceuticals:
- **Drug Interaction Analysis:** Model the relationships between different biological markers over time.
- **Clinical Trial Forecasting:** Predict patient outcomes based on longitudinal data.

### Technology:
- **Sensor Data Analysis:** Analyze interdependencies in data from IoT devices or sensors.
- **User Behavior Modeling:** Predict user engagement or churn based on historical data.

### Supply Chain and Logistics:
- **Demand Forecasting:** Model the relationships between supply chain variables to optimize inventory levels.
- **Delivery Route Optimization:** Predict delivery times and optimize logistics networks.

---

## What’s Included?

This package provides the following features:
- **Alternating Least Squares (ALS) Algorithm:** For VAR estimation via tensor decomposition.
- **SHORR Algorithm:** Lasso-penalized regression for VAR estimation via tensor decomposition.
- **Higher-Order Singular Value Decomposition (HOSVD):** For efficient tensor decomposition.
- **Sparse and Orthogonal Regression Subroutines:** For improved model interpretability.
- **Sampling Procedures:** Tools to generate synthetic VAR data for testing and validation.
- **Tensor Algebra Utilities:** Helper functions for tensor operations.

---

## How to Use This Package

### Installation:
This package requires `jax` and `numpy`. You can install it using:
```bash
pip install -e .
```

## References

- Di Wang, Yao Zheng, Heng Lian, and Guodong Li,  
  **"High-dimensional vector autoregressive time series modeling via tensor decomposition"**,  
  [https://arxiv.org/abs/1909.06624](https://arxiv.org/abs/1909.06624)

- Rongjie Lai and Stanley Osher,  
  **"A splitting method for orthogonality-constrained problems"**,  
  *Journal of Scientific Computing*, 2014.  
  [https://doi.org/10.1007/s10915-013-9740-x](https://doi.org/10.1007/s10915-013-9740-x)

For a complete list of references, see `ref.bib`.