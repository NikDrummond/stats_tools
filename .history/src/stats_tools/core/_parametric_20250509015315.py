import jax
jax.config.update("jax_enable_x64", True)

from jax import lax, device_get
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
from ._data_prep import _validate_and_dropna, _check_group_levels, _df_to_jax
from jax.scipy.stats import t as student_t
import jax.scipy.special as spec

def _anova_ssb_body(
    i: int,
    acc: float,
    labels: jnp.ndarray,
    measure: jnp.ndarray,
    grand_mean: float
) -> float:
    """
    Single‐step body for SS_between in one‐way ANOVA.

    :param i: group index (0 … k-1)
    :param acc: running sum of SS_between
    :param labels: integer‐coded labels array
    :param measure: outcome values array
    :param grand_mean: overall mean of measure
    :return: updated SS_between
    """
    mask = labels == i
    n_i = jnp.sum(mask)
    mean_i = jnp.sum(measure * mask) / n_i
    return acc + n_i * (mean_i - grand_mean) ** 2


@jax.jit
def _one_way_anova(
    labels: jnp.ndarray,
    measure: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    One‐way ANOVA (jitted) using a top‐level helper for the loop body.

    :param labels: 1D integer codes in {0,1,…,k-1}.
    :param measure: 1D observations.
    :return: (F_statistic, p_value) where p is the upper‐tail F CDF.
    """
    n = measure.size
    k = jnp.max(labels).astype(int) + 1

    # grand mean and total SS
    grand_mean = jnp.mean(measure)
    ss_total  = jnp.sum((measure - grand_mean) ** 2)

    # build a body function with labels, measure, grand_mean baked in
    body_fn = partial(
        _anova_ssb_body,
        labels=labels,
        measure=measure,
        grand_mean=grand_mean,
    )

    # between‐groups sum of squares
    ss_between = lax.fori_loop(0, k, body_fn, 0.0)
    ss_within  = ss_total - ss_between

    df_between = k - 1
    df_within  = n - k

    msb = ss_between / df_between
    msw = ss_within  / df_within

    # F-statistic
    F = msb / msw

    # p-value via the upper‐tail of the F distribution
    z   = (df_between * F) / (df_between * F + df_within)
    cdf = spec.betainc(df_between / 2.0, df_within / 2.0, z)
    p   = 1.0 - cdf

    return F, p

@jax.jit
def t_cdf(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    CDF of the Student‐t distribution for x >= 0, via the regularized incomplete beta.

    :param x: nonnegative t‐statistic(s).
    :param v: degrees of freedom (float or array of floats).
    :return: Student‐t CDF evaluated at x.
    """
    z = v / (v + x**2)
    ib = spec.betainc(v * 0.5, 0.5, z)
    return 1.0 - 0.5 * ib

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

    # t-statistic
    t_stat = diff / se

    # two-sided p-value using our top-level t_cdf
    cdf = t_cdf(jnp.abs(t_stat), df)
    p_val = 2.0 * (1.0 - cdf)

    return t_stat, p_val

@jax.jit
def _linear_regression_kernel(
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
    df = n - p

    # 1) Build xtx and right‐hand‐side
    xtx = X.T @ X           # (p, p)
    rhs = X.T @ y           # (p,)

    # 2) Solve for betas without ever inverting explicitly
    betas = jnp.linalg.solve(xtx, rhs)

    # 3) Residual variance
    resid  = y - (X @ betas)
    sigma2 = jnp.sum(resid**2) / df

    # 4) Compute (XᵀX)⁻¹ by solving against identity
    I      = jnp.eye(p, dtype=X.dtype)
    xtx_inv = jnp.linalg.solve(xtx, I)

    # 5) Covariance and standard errors
    cov_beta = xtx_inv * sigma2
    ses      = jnp.sqrt(jnp.diag(cov_beta))

    # 6) t‐statistics
    t_stats = betas / ses

    # 7) two‐sided p‐values via t_cdf (as before)
    df_arr   = jnp.array(df, dtype=t_stats.dtype)
    p_values = 2.0 * (1.0 - t_cdf(jnp.abs(t_stats), df_arr))

    return betas, ses, t_stats, p_values

def _linear_regression(
    X: jnp.ndarray,
    y: jnp.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Host-side wrapper around the JIT OLS kernel that returns NumPy arrays.

    :param X: design matrix (n_samples, n_features)
    :param y: outcome vector (n_samples,)
    :return: tuple of four NumPy 1D arrays: (betas, ses, t_stats, p_values)
    """
    betas, ses, t_stats, p_values = _linear_regression_kernel(X, y)
    # Move from device (JAX) to host (NumPy)
    betas, ses, t_stats, p_values = device_get((betas, ses, t_stats, p_values))
    return betas, ses, t_stats, p_values

