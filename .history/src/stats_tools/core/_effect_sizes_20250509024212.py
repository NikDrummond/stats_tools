# src/stats_tools/core/_effect_sizes.py

from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
import jax.scipy.special as spec
from jax import lax, random
from functools import partial
import numpy as np
from scipy.stats import t as _t_dist  # for analytic t‐quantiles

# ───────────────────────────────────────────────────────────────────────────────
# Top‐level loop bodies
# ───────────────────────────────────────────────────────────────────────────────

def _eta_sq_body(
    i: int,
    acc: float,
    groups: jnp.ndarray,
    data: jnp.ndarray,
    grand_mean: float
) -> float:
    """
    Body for η²: add n_i*(mean_i - grand_mean)**2 for group i.
    """
    mask = groups == i
    n_i = jnp.sum(mask)
    mean_i = jnp.sum(data * mask) / n_i
    return acc + n_i * (mean_i - grand_mean) ** 2

def _eps_within_body(
    i: int,
    acc: float,
    groups: jnp.ndarray,
    data: jnp.ndarray
) -> float:
    """
    Body for SS_within in ε²: add Σ mask_i * (x - mean_i)**2 for group i.
    """
    mask = groups == i
    # group mean via masked sum
    n_i = jnp.sum(mask)
    mean_i = jnp.sum(data * mask) / n_i
    # sum of squared deviations masked
    ss_i = jnp.sum(mask * (data - mean_i) ** 2)
    return acc + ss_i

# ───────────────────────────────────────────────────────────────────────────────
# Effect‐size estimators (point estimates)
# ───────────────────────────────────────────────────────────────────────────────

@jax.jit
def cohens_d(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """
    Unbiased Cohen's d (with Hedges' correction).
    """
    n1, n2 = x1.size, x2.size
    m1, m2 = jnp.mean(x1), jnp.mean(x2)
    s1 = jnp.var(x1, ddof=1)
    s2 = jnp.var(x2, ddof=1)
    s_pooled = jnp.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
    d = (m1 - m2) / s_pooled
    df = n1 + n2 - 2
    J = 1 - (3.0 / (4.0*df - 1.0))
    return d * J

@jax.jit
def eta_squared(groups: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
    """
    η² = SS_between / SS_total for one‐way ANOVA.
    """
    grand_mean = jnp.mean(data)
    ss_total = jnp.sum((data - grand_mean) ** 2)
    k = jnp.max(groups).astype(int) + 1

    body_fn = partial(_eta_sq_body, groups=groups, data=data, grand_mean=grand_mean)
    ss_between = lax.fori_loop(0, k, body_fn, 0.0)
    return ss_between / ss_total

@jax.jit
def epsilon_squared(groups: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
    """
    ε² estimator for one‐way ANOVA.
    """
    n = data.size
    grand_mean = jnp.mean(data)
    ss_total = jnp.sum((data - grand_mean) ** 2)
    k = jnp.max(groups).astype(int) + 1

    within_fn = partial(_eps_within_body, groups=groups, data=data)
    ss_within = lax.fori_loop(0, k, within_fn, 0.0)
    df_between = k - 1
    ms_within = jnp.var(data, ddof=1)
    ss_between = ss_total - ss_within
    num = ss_between - df_between * ms_within
    den = ss_total + ms_within
    # avoid 0/0: if den==0, return 0.0
    eps2 = num / den
    return jnp.where(den > 0, eps2, 0.0)

@jax.jit
def rank_biserial(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """
    Rank‐biserial correlation for two independent samples.
    """
    all_x = jnp.concatenate([x1, x2])
    ranks = jnp.argsort(jnp.argsort(all_x)) + 1
    r1 = ranks[: x1.size]
    U = jnp.sum(r1) - x1.size*(x1.size+1)/2.0
    n1, n2 = x1.size, x2.size
    return (2.0 * U) / (n1 * n2) - 1.0

@jax.jit
def standardized_beta(X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Standardized regression coefficient (Pearson r).
    """
    Xs = (X - jnp.mean(X)) / jnp.std(X, ddof=1)
    ys = (y - jnp.mean(y)) / jnp.std(y, ddof=1)
    return jnp.sum(Xs * ys) / (X.size - 1)

@jax.jit
def partial_r_squared(
    X: jnp.ndarray,
    y: jnp.ndarray,
    covariates: jnp.ndarray
) -> jnp.ndarray:
    """
    Partial R² of X controlling for covariates via RSS reduction.
    """
    X_full = jnp.column_stack([covariates, X])
    beta_full = jnp.linalg.lstsq(X_full, y, rcond=None)[0]
    resid_full = y - X_full @ beta_full
    ss_full = jnp.sum(resid_full**2)

    beta_red = jnp.linalg.lstsq(covariates, y, rcond=None)[0]
    resid_red = y - covariates @ beta_red
    ss_red = jnp.sum(resid_red**2)

    return (ss_red - ss_full) / ss_red

# ───────────────────────────────────────────────────────────────────────────────
# Confidence‐interval machinery
# ───────────────────────────────────────────────────────────────────────────────

def analytic_ci(
    est: float,
    se: float,
    alpha: float = 0.05,
    df: Optional[int] = None
) -> Tuple[float, float]:
    """
    Analytic two‐sided CI: normal if df=None, else Student’s t.
    """
    if df is None:
        # normal quantile via inverse error function
        z = float(jnp.sqrt(2.0) * spec.erfinv(1.0 - alpha))
    else:
        # Student's t quantile from SciPy
        z = float(_t_dist.ppf(1.0 - alpha/2.0, df))
    return est - z * se, est + z * se
