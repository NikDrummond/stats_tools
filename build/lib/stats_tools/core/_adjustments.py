# stats_module/core/_adjustments.py

from typing import Literal
import jax
import jax.numpy as jnp

_PAdjustMethod = Literal["none", "bonferroni", "holm", "fdr"]


@jax.jit
def bonferroni(p_values: jnp.ndarray) -> jnp.ndarray:
    """
    Bonferroni correction: p_adj = min(p * m, 1).

    :param p_values: 1D array of raw p-values.
    :return: 1D array of Bonferroni-adjusted p-values.
    """
    m = p_values.size
    return jnp.minimum(p_values * m, 1.0)


@jax.jit
def fdr_bh(p_values: jnp.ndarray) -> jnp.ndarray:
    """
    Benjamini–Hochberg FDR correction.

    :param p_values: 1D array of raw p-values.
    :return: 1D array of FDR-adjusted p-values (monotonic, ≤1).
    """
    m = p_values.size
    # sort p ascending
    sort_idx = jnp.argsort(p_values)
    p_sorted = p_values[sort_idx]
    # BH formula
    ranks = jnp.arange(1, m + 1)
    q = p_sorted * m / ranks
    # enforce monotonicity: cummin from the end
    q_rev = jnp.minimum.accumulate(q[::-1])[::-1]
    # cap at 1 and invert sort
    q_rev = jnp.minimum(q_rev, 1.0)
    inv_idx = jnp.argsort(sort_idx)
    return q_rev[inv_idx]


@jax.jit
def holm(p_values: jnp.ndarray) -> jnp.ndarray:
    """
    Holm–Bonferroni step-down correction.

    :param p_values: 1D array of raw p-values.
    :return: 1D array of Holm-adjusted p-values (monotonic, ≤1).
    """
    m = p_values.size
    # sort p ascending
    sort_idx = jnp.argsort(p_values)
    p_sorted = p_values[sort_idx]
    # multipliers = m, m-1, ..., 1
    multipliers = m - jnp.arange(m)
    adj = jnp.minimum(p_sorted * multipliers, 1.0)
    # enforce non-decreasing across sorted array
    adj_monotonic = jnp.maximum.accumulate(adj)
    # invert sort
    inv_idx = jnp.argsort(sort_idx)
    return adj_monotonic[inv_idx]


def adjust_pvalues(
    p_values: jnp.ndarray,
    method: _PAdjustMethod = "none"
) -> jnp.ndarray:
    """
    Dispatch to multiple-comparison correction.

    :param p_values: 1D array of raw p-values.
    :param method: one of 'none', 'bonferroni', 'holm', 'fdr'.
    :return: 1D array of adjusted p-values.
    :raises ValueError: if method is unrecognized.
    """
    if method == "none":
        return p_values
    elif method == "bonferroni":
        return bonferroni(p_values)
    elif method == "holm":
        return holm(p_values)
    elif method == "fdr":
        return fdr_bh(p_values)
    else:
        raise ValueError(f"Unknown p-value adjustment method: {method}")
