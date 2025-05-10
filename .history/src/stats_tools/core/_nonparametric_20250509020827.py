# src/stats_tools/core/_nonparametric.py

import jax
import jax.numpy as jnp
import jax.scipy.special as spec
from jax import lax
from functools import partial
from typing import Tuple

from ._data_prep import _validate_and_dropna, _check_group_levels, _df_to_jax

# ────────────────────────────────────────────────────────────────────────────────
# Helper for Kruskal–Wallis: group sum‐of‐ranks term
# ────────────────────────────────────────────────────────────────────────────────

def _kw_group_term(
    i: int,
    acc: float,
    labels: jnp.ndarray,
    ranks: jnp.ndarray,
) -> float:
    """
    Accumulate term for group i in Kruskal–Wallis:
      Σ(ranks)**2 / n_i
    """
    mask = labels == i
    sum_r = jnp.sum(ranks * mask)
    n_i   = jnp.sum(mask)
    return acc + (sum_r ** 2) / n_i


# ────────────────────────────────────────────────────────────────────────────────
# Nonparametric kernels
# ────────────────────────────────────────────────────────────────────────────────

@jax.jit
def _rank_data(data: jnp.ndarray) -> jnp.ndarray:
    """
    Assign ranks to the data via competition ranking (ties get distinct ranks),
    implemented purely in JAX.

    :param data: 1D JAX array of observations.
    :return: 1D JAX array of integer ranks 1..n.
    """
    # double argsort gives competition ranks starting at 0
    ranks0 = jnp.argsort(jnp.argsort(data))
    return ranks0 + 1


@jax.jit
def _kruskal_wallis(
    measure: jnp.ndarray,
    labels: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Kruskal–Wallis H‐test for independent samples, via JAX.

    :param measure: 1D observations array.
    :param labels: 1D integer codes in {0,…,k-1}.
    :return: (H-statistic, p-value) using χ²‐approximation.
    """
    # rank all observations
    ranks = _rank_data(measure)
    n     = measure.shape[0]

    # number of groups = max(label)+1
    k = jnp.max(labels).astype(int) + 1

    # sum term via fori_loop
    term_fn  = partial(_kw_group_term, labels=labels, ranks=ranks)
    sum_term = lax.fori_loop(0, k, term_fn, 0.0)

    # Kruskal–Wallis H
    H = (12.0 / (n * (n + 1))) * sum_term - 3.0 * (n + 1)

    # degrees of freedom
    df = k - 1

    # chi2 survival = regularized upper‐incomplete gamma
    # p = Q(df/2, H/2)
    p = spec.gammaincc(df / 2.0, H / 2.0)

    return H, p


@jax.jit
def _rank_biserial(
    measure: jnp.ndarray,
    labels: jnp.ndarray
) -> jnp.ndarray:
    """
    Rank‐biserial correlation for two independent groups.

    :param measure: 1D observations.
    :param labels: 1D integer codes 0 or 1.
    :return: rank‐biserial correlation in [-1, 1].
    """
    ranks = _rank_data(measure)
    sum1  = jnp.sum(ranks * (labels == 1))
    n1    = jnp.sum(labels == 1)
    U     = sum1 - n1 * (n1 + 1) / 2.0
    n0    = jnp.sum(labels == 0)
    return 1.0 - (2.0 * U) / (n1 * n0)
