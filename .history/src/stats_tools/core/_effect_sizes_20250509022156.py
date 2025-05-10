# stats_module/core/_effect_sizes.py

from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import t

# ───────────────────────────────────────────────────────────────────────────────
# Effect‐size estimators (point estimates)
# ───────────────────────────────────────────────────────────────────────────────

@jax.jit
def cohens_d(x1: jnp.ndarray, x2: jnp.ndarray) -> float:
    """
    Compute unbiased Cohen's d between two independent samples.
    
    :param x1: 1D array of sample 1.
    :param x2: 1D array of sample 2.
    :return: Cohen's d with Hedges' correction.
    """
    n1, n2 = x1.size, x2.size
    m1, m2 = jnp.mean(x1), jnp.mean(x2)
    s1, s2 = jnp.var(x1, ddof=1), jnp.var(x2, ddof=1)
    # pooled SD
    s_pooled = jnp.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
    d = (m1 - m2) / s_pooled
    # Hedges' g correction factor
    df = n1 + n2 - 2
    J = 1 - (3 / (4*df - 1))
    return d * J

@jax.jit
def eta_squared(groups: jnp.ndarray, data: jnp.ndarray) -> float:
    """
    Compute eta-squared (η²) for one-way ANOVA.
    
    :param groups: integer-coded group labels.
    :param data: 1D outcome values.
    :return: η² = SS_between / SS_total.
    """
    grand_mean = jnp.mean(data)
    ss_total = jnp.sum((data - grand_mean) ** 2)
    def ssb(acc, g):
        mask = groups == g
        n_g = jnp.sum(mask)
        return acc + n_g * (jnp.mean(data[mask]) - grand_mean) ** 2
    ss_between = jax.lax.fori_loop(0, jnp.unique(groups).size, lambda i, acc: ssb(acc, jnp.unique(groups)[i]), 0.0)
    return ss_between / ss_total

@jax.jit
def epsilon_squared(groups: jnp.ndarray, data: jnp.ndarray) -> float:
    """
    Compute epsilon-squared (ε²) for one-way ANOVA.
    
    :param groups: integer-coded group labels.
    :param data: 1D outcome values.
    :return: ε² estimate.
    """
    n = data.size
    grand_mean = jnp.mean(data)
    ss_total = jnp.sum((data - grand_mean) ** 2)
    ss_within = 0.0
    for g in jnp.unique(groups):
        mask = groups == g
        ss_within += jnp.sum((data[mask] - jnp.mean(data[mask]))**2)
    df_between = jnp.unique(groups).size - 1
    ss_between = ss_total - ss_within
    return (ss_between - df_between * jnp.var(data, ddof=1)) / (ss_total + jnp.var(data, ddof=1))

@jax.jit
def rank_biserial(x1: jnp.ndarray, x2: jnp.ndarray) -> float:
    """
    Compute rank-biserial correlation for two groups.
    
    :param x1: sample 1 values.
    :param x2: sample 2 values.
    :return: rank-biserial r.
    """
    all_x = jnp.concatenate([x1, x2])
    ranks = jnp.argsort(jnp.argsort(all_x)) + 1
    r1 = ranks[: x1.size]
    r2 = ranks[x1.size :]
    U = jnp.sum(r1) - x1.size*(x1.size+1)/2
    n1, n2 = x1.size, x2.size
    return 1 - (2 * U) / (n1 * n2)

@jax.jit
def standardized_beta(X: jnp.ndarray, y: jnp.ndarray) -> float:
    """
    Compute standardized regression coefficient (β) of X predicting y.
    
    :param X: 1D predictor array.
    :param y: 1D outcome array.
    :return: standardized β.
    """
    Xs = (X - jnp.mean(X)) / jnp.std(X, ddof=1)
    ys = (y - jnp.mean(y)) / jnp.std(y, ddof=1)
    return jnp.sum(Xs * ys) / (X.size - 1)

@jax.jit
def partial_r_squared(X: jnp.ndarray, y: jnp.ndarray, covariates: jnp.ndarray) -> float:
    """
    Compute partial R² of X controlling for covariates.
    
    :param X: 1D predictor.
    :param y: 1D outcome.
    :param covariates: 2D array of additional regressors (n_samples × n_covariates).
    :return: partial R².
    """
    # full model
    X_full = jnp.column_stack([covariates, X])
    beta_full = jnp.linalg.lstsq(X_full, y, rcond=None)[0]
    resid_full = y - X_full @ beta_full
    ss_res_full = jnp.sum(resid_full**2)
    # reduced model
    beta_red = jnp.linalg.lstsq(covariates, y, rcond=None)[0]
    resid_red = y - covariates @ beta_red
    ss_res_red = jnp.sum(resid_red**2)
    return (ss_res_red - ss_res_full) / ss_res_red


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
    Compute analytic (normal or t-based) CI for an estimate.
    
    :param est: point estimate.
    :param se: standard error.
    :param alpha: two-sided error rate.
    :param df: degrees of freedom; if None, use normal quantile.
    :return: (lower, upper).
    """    
    if df is None:
        z = jax.scipy.stats.norm.ppf(1 - alpha/2)
    else:
        z = t.ppf(1 - alpha/2, df)
    return (est - z * se, est + z * se)


def bootstrap_ci(
    effect_func: Callable[..., float],
    data_args: Tuple,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """
    Nonparametric bootstrap percentile CI for any effect-size function.
    
    :param effect_func: function returning a scalar effect-size given data.
    :param data_args: tuple of JAX arrays or numpy arrays to bootstrap;
                      first dimension is sample size.
    :param n_bootstraps: number of bootstrap resamples.
    :param alpha: two-sided error rate.
    :param random_state: seed for reproducibility.
    :return: (lower, upper) percentile CI.
    """
    key = random.PRNGKey(random_state or 0)
    n = data_args[0].shape[0]
    def one_boot(key):
        idx = random.randint(key, (n,), 0, n)
        sampled = tuple(arg[idx] for arg in data_args)
        return effect_func(*sampled)
    keys = random.split(key, n_bootstraps)
    boots = jax.vmap(one_boot)(keys)
    lower = jnp.percentile(boots, 100 * (alpha/2))
    upper = jnp.percentile(boots, 100 * (1 - alpha/2))
    return float(lower), float(upper)


def permutation_ci(
    effect_func: Callable[..., float],
    data_args: Tuple[jnp.ndarray, jnp.ndarray],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """
    Permutation-based CI by inverting null distribution of effect sizes.
    
    :param effect_func: effect-size function taking (data, labels) or similar.
    :param data_args: typically (data, labels).
    :param n_permutations: number of label-shuffles.
    :param alpha: two-sided error rate.
    :param random_state: seed.
    :return: (lower, upper) bounds.
    """
    key = random.PRNGKey(random_state or 0)
    data, labels = data_args
    def one_perm(key):
        perm = random.permutation(key, labels)
        return effect_func(data, perm)
    keys = random.split(key, n_permutations)
    null_dist = jax.vmap(one_perm)(keys)
    lower = jnp.percentile(null_dist, 100 * (alpha/2))
    upper = jnp.percentile(null_dist, 100 * (1 - alpha/2))
    return float(lower), float(upper)
