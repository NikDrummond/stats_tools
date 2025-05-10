import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional


def _shuffle_labels(
    labels: jnp.ndarray,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Return a new array of labels randomly permuted.

    :param labels: 1D JAX integer array of group labels.
    :param key: JAX PRNGKey for reproducibility.
    :return: Shuffled labels array.
    """
    return jax.random.permutation(key, labels)


def _null_distribution(
    stat_func: Callable[[jnp.ndarray, jnp.ndarray], Tuple[float, float]],
    measure: jnp.ndarray,
    labels: jnp.ndarray,
    n_permutations: int,
    seed: int
) -> jnp.ndarray:
    """
    Generate null distribution of test statistics by permuting labels.

    :param stat_func: JAX-jitted function returning (statistic, p-value).
    :param measure: 1D JAX array of observations.
    :param labels: 1D JAX integer array of original labels.
    :param n_permutations: Number of permutations.
    :param seed: Integer seed for PRNG.
    :return: 1D JAX array of permuted statistics.
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_permutations)
    # Compute statistic for each permuted label set
    def perm_stat(k):
        perm_labels = jax.random.permutation(k, labels)
        stat, _ = stat_func(measure, perm_labels)
        return stat
    return jax.vmap(perm_stat)(keys)


def empirical_p_value(
    observed: float,
    null_dist: jnp.ndarray,
    alternative: str = 'two-sided'
) -> float:
    """
    Compute empirical p-value given null distribution.

    :param observed: Observed test statistic.
    :param null_dist: 1D JAX array of null distribution statistics.
    :param alternative: 'two-sided', 'greater', or 'less'.
    :return: Empirical p-value.
    """
    if alternative == 'two-sided':
        return jnp.mean(jnp.abs(null_dist) >= jnp.abs(observed))
    elif alternative == 'greater':
        return jnp.mean(null_dist >= observed)
    elif alternative == 'less':
        return jnp.mean(null_dist <= observed)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")


def bootstrap_ci(
    effect_func: Callable[..., float],
    data_args: Tuple,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """
    Percentile‐bootstrap CI on effect_func(data_args).
    """
    key = random.PRNGKey(random_state or 0)
    n = data_args[0].shape[0]

    def one_boot(key):
        idx = random.randint(key, (n,), 0, n)
        sampled = tuple(arg[idx] for arg in data_args)
        return effect_func(*sampled)

    keys = random.split(key, n_bootstraps)
    boots = jax.vmap(one_boot)(keys)
    lower = jnp.percentile(boots, 100*(alpha/2.0))
    upper = jnp.percentile(boots, 100*(1.0 - alpha/2.0))
    return float(lower), float(upper)

def permutation_ci(
    effect_func: Callable[..., float],
    data_args: Tuple[jnp.ndarray, jnp.ndarray],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """
    Permutation‐based percentile CI via label shuffling.
    """
    key = random.PRNGKey(random_state or 0)
    data, labels = data_args

    def one_perm(key):
        perm = random.permutation(key, labels)
        return effect_func(data, perm)

    keys = random.split(key, n_permutations)
    nulls = jax.vmap(one_perm)(keys)
    lower = jnp.percentile(nulls, 100*(alpha/2.0))
    upper = jnp.percentile(nulls, 100*(1.0 - alpha/2.0))
    return float(lower), float(upper)