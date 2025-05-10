import jax
from jax import lax
import jax.numpy as jnp
from typing import Tuple, Optional
from ._data_prep import _validate_and_dropna, _check_group_levels, _df_to_jax
from jax.scipy.stats import t as student_t
import jax.scipy.special as spec

@jax.jit
def _one_way_anova(
    measure: jnp.ndarray,
    labels: jnp.ndarray
) -> Tuple[float, float]:
    """
    Perform a one-way ANOVA F-test using JAX.

    :param measure: 1D JAX array of observations.
    :param labels: 1D JAX integer array of group labels (0..k-1).
    :return: Tuple of (F-statistic, p-value).
    """
    overall_mean = jnp.mean(measure)
    ss_between = jnp.sum(jax.vmap(lambda l: measure[labels==l].shape[0] * (jnp.mean(measure[labels==l]) - overall_mean)**2)(jnp.unique(labels)))
    ss_within = jnp.sum(jax.vmap(lambda l: jnp.sum((measure[labels==l] - jnp.mean(measure[labels==l]))**2))(jnp.unique(labels)))
    k = jnp.unique(labels).shape[0]
    n = measure.shape[0]
    df_between = k - 1
    df_within = n - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within
    # p-value via F CDF (uses jax.scipy placeholder)
    p_value = 1 - jax.scipy.stats.f.cdf(f_stat, df_between, df_within)
    return f_stat, p_value


@jax.jit
def _t_test(
    measure: jnp.ndarray,
    labels: jnp.ndarray,
    equal_var: bool = True
) -> Tuple[float, float]:
    """
    Two‐sample t‐test (independent groups) using mask‐based sums
    and an explicit t‐distribution CDF via the regularized incomplete beta.

    :param measure: 1D array of all observations.
    :param labels: 1D integer array, same length, values 0 or 1 for group.
    :param equal_var: True for Student’s t (pooled), False for Welch’s t.
    :return: (t_statistic, two‐sided p_value).
    """
    # Boolean masks
    mask0 = labels == 0
    mask1 = labels == 1

    # Sample sizes
    n0 = jnp.sum(mask0)
    n1 = jnp.sum(mask1)

    # Means
    mean0 = jnp.sum(measure * mask0) / n0
    mean1 = jnp.sum(measure * mask1) / n1

    # Unbiased variances (ddof=1)
    var0 = jnp.sum(mask0 * (measure - mean0) ** 2) / (n0 - 1)
    var1 = jnp.sum(mask1 * (measure - mean1) ** 2) / (n1 - 1)

    # Difference in means
    diff = mean0 - mean1

    # Pooled variance SE & float df
    pooled_var = ((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2)
    se_pooled = jnp.sqrt(pooled_var * (1.0 / n0 + 1.0 / n1))
    df_pooled = jnp.array(n0 + n1 - 2, dtype=measure.dtype)

    # Welch SE & df
    v0n = var0 / n0
    v1n = var1 / n1
    se_welch = jnp.sqrt(v0n + v1n)
    df_num = (v0n + v1n) ** 2
    df_den = (v0n ** 2) / (n0 - 1) + (v1n ** 2) / (n1 - 1)
    df_welch = df_num / df_den

    # Pick SE & df via lax.cond on traced equal_var
    se, df = lax.cond(
        equal_var,
        lambda _: (se_pooled, df_pooled),
        lambda _: (se_welch, df_welch),
        operand=None,
    )

    # t‐statistic
    t_stat = diff / se

    # Student‐t CDF via regularized incomplete beta
    def t_cdf(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # x assumed ≥ 0 here
        z = v / (v + x**2)
        ib = spec.betainc(v / 2.0, 0.5, z)
        # for positive x: F = 1 - 0.5*I; for negative we'd do 0.5*I
        return 1.0 - 0.5 * ib

    cdf_abs = t_cdf(jnp.abs(t_stat), df)
    p_val = 2.0 * (1.0 - cdf_abs)

    return t_stat, p_val

@jax.jit
def _linear_regression(
    X: jnp.ndarray,
    y: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fit ordinary least‐squares regression via closed‐form solution.

    :param X: Design matrix of shape (n_samples, n_features), must include intercept column if desired.
    :param y: Outcome vector of shape (n_samples,).
    :return: A tuple of four 1D arrays, each of length n_features:
        - betas: estimated regression coefficients.
        - ses: standard errors of each coefficient.
        - t_stats: t‐statistics (beta / se).
        - p_values: two‐sided p‐values from Student’s t distribution.
    """
    n, p = X.shape

    # β̂ = (XᵀX)⁻¹ Xᵀ y
    xtx = X.T @ X
    xtx_inv = jnp.linalg.inv(xtx)
    betas = xtx_inv @ (X.T @ y)

    # residuals and estimate of σ²
    resid = y - (X @ betas)
    df = n - p
    sigma2 = jnp.sum(resid ** 2) / df

    # covariance matrix and SEs
    cov_beta = xtx_inv * sigma2
    ses = jnp.sqrt(jnp.diag(cov_beta))

    # t‐statistics and two‐sided p‐values
    t_stats = betas / ses
    p_values = 2 * (1.0 - student_t.cdf(jnp.abs(t_stats), df))

    return betas, ses, t_stats, p_values