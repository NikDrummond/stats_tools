from typing import Tuple
import jax
import jax.numpy as jnp

@jax.jit
def trimmed_mean(x: jnp.ndarray, proportion_to_cut: float = 0.2) -> float:
    """
    Compute the α‐trimmed mean of a 1D array.
    
    :param x: 1D array of observations.
    :param proportion_to_cut: fraction (0 ≤ α < 0.5) to remove from each tail.
    :return: trimmed mean.
    :raises ValueError: if proportion_to_cut not in [0, 0.5).
    """
    if not (0 <= proportion_to_cut < 0.5):
        raise ValueError("proportion_to_cut must be in [0, 0.5)")
    n = x.shape[0]
    k = jnp.floor(proportion_to_cut * n).astype(int)
    x_sorted = jnp.sort(x)
    return jnp.mean(x_sorted[k : n - k])

@jax.jit
def huber_m_estimator(x: jnp.ndarray, c: float = 1.345, max_iter: int = 50) -> Tuple[float, float]:
    """
    Compute Huber’s M‐estimator location and scale for a 1D array.
    
    :param x: 1D array of observations.
    :param c: tuning constant; default 1.345 for 95% efficiency under normality.
    :param max_iter: maximum IRLS iterations.
    :return: (mu, sigma) robust location and scale.
    """
    # initial estimates
    mu = jnp.median(x)
    sigma = jnp.median(jnp.abs(x - mu)) / 0.6745

    def body(state):
        mu, sigma = state
        resid = (x - mu) / sigma
        w = jnp.where(jnp.abs(resid) <= c, 1.0, c / jnp.abs(resid))
        mu_new = jnp.sum(w * x) / jnp.sum(w)
        sigma_new = jnp.sqrt(jnp.sum(w * (x - mu_new) ** 2) / jnp.sum(w))
        return mu_new, sigma_new

    mu_final, sigma_final = jax.lax.fori_loop(0, max_iter, lambda i, s: body(s), (mu, sigma))
    return mu_final, sigma_final

@jax.jit
def robust_cohens_d(x1: jnp.ndarray, x2: jnp.ndarray, proportion_to_cut: float = 0.2) -> float:
    """
    Compute robust Cohen’s d based on trimmed means and Winsorized pooled SD.
    
    :param x1: sample 1 array.
    :param x2: sample 2 array.
    :param proportion_to_cut: trimming fraction for both samples.
    :return: unbiased effect‐size estimate.
    """
    m1 = trimmed_mean(x1, proportion_to_cut)
    m2 = trimmed_mean(x2, proportion_to_cut)
    # Winsorize both samples
    def winsorize(x):
        k = jnp.floor(proportion_to_cut * x.shape[0]).astype(int)
        x_s = jnp.sort(x)
        lower, upper = x_s[k], x_s[-k-1]
        return jnp.clip(x, lower, upper)
    w1, w2 = winsorize(x1), winsorize(x2)
    s_pool = jnp.sqrt(((w1.size - 1) * jnp.var(w1) + (w2.size - 1) * jnp.var(w2)) / (w1.size + w2.size - 2))
    return (m1 - m2) / s_pool

@jax.jit
def robust_eta_squared(groups: jnp.ndarray, data: jnp.ndarray, proportion_to_cut: float = 0.2) -> float:
    """
    Estimate η² for multiple groups using trimmed‐means approach.
    
    :param groups: integer‐coded group labels.
    :param data: outcome values.
    :param proportion_to_cut: trimming fraction.
    :return: robust η².
    """
    overall_trim = trimmed_mean(data, proportion_to_cut)
    ss_between = 0.0
    ss_total = jnp.sum((data - overall_trim) ** 2)
    for level in jnp.unique(groups):
        mask = groups == level
        grp_data = data[mask]
        m_trim = trimmed_mean(grp_data, proportion_to_cut)
        ss_between += grp_data.size * (m_trim - overall_trim) ** 2
    return ss_between / ss_total

@jax.jit
def robust_se_huber(x: jnp.ndarray, c: float = 1.345) -> float:
    """
    Compute robust standard error of the mean using Huber weights.
    
    :param x: 1D array of observations.
    :param c: Huber tuning constant.
    :return: robust SE for location estimate.
    """
    mu, sigma = huber_m_estimator(x, c)
    resid = (x - mu) / sigma
    w = jnp.where(jnp.abs(resid) <= c, 1.0, c / jnp.abs(resid))
    var_est = jnp.sum(w * (x - mu) ** 2) / (jnp.sum(w) ** 2)
    return jnp.sqrt(var_est * x.size)

# You can add more robust‐SEs, plus CI via percentile‐bootstrap wrappers in effect_sizes.
