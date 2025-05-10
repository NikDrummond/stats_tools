import jax
import jax.numpy as jnp
import pandas as pd
from typing import Tuple, Dict, Any
from ._data_prep import _validate_and_dropna, _check_group_levels, _df_to_jax

@jax.jit
def _rank_data(
    data: jnp.ndarray
) -> jnp.ndarray:
    """
    Assign ranks to the data, average ties.

    :param data: 1D JAX array of observations.
    :return: 1D JAX array of ranks.
    """
    # Convert to numpy for ranking then back to JAX
    import numpy as onp
    ranks = onp.argsort(onp.argsort(onp.array(data))) + 1
    return jnp.array(ranks)

@jax.jit
def _kruskal_wallis(
    measure: jnp.ndarray,
    labels: jnp.ndarray
) -> Tuple[float, float]:
    """
    Perform Kruskal–Wallis H-test for independent samples.

    :param measure: 1D JAX array of observations.
    :param labels: 1D JAX integer array of group labels.
    :return: Tuple of (H-statistic, p-value).
    """
    # Rank all data
    ranks = _rank_data(measure)
    n = ranks.shape[0]
    unique_labels = jnp.unique(labels)

    # Compute H statistic
    def group_stat(l):
        grp_ranks = ranks[labels == l]
        ni = grp_ranks.shape[0]
        return (jnp.sum(grp_ranks)**2) / ni
    sum_term = jnp.sum(jax.vmap(group_stat)(unique_labels))
    h = (12 / (n * (n + 1)) * sum_term) - 3 * (n + 1)

    # p-value via chi2 CDF
    df = unique_labels.shape[0] - 1
    p_value = 1 - jax.scipy.stats.chi2.cdf(h, df)
    return h, p_value

@jax.jit
def _rank_biserial(
    measure: jnp.ndarray,
    labels: jnp.ndarray
) -> float:
    """
    Compute rank-biserial correlation for two groups.

    :param measure: 1D JAX array of observations, two groups.
    :param labels: 1D JAX integer array with values 0 or 1.
    :return: Rank-biserial correlation coefficient.
    """
    ranks = _rank_data(measure)
    r1 = ranks[labels == 1]
    r0 = ranks[labels == 0]
    n1, n0 = r1.shape[0], r0.shape[0]
    u_stat = jnp.sum(r1) - n1 * (n1 + 1) / 2
    return 1 - (2 * u_stat) / (n1 * n0)