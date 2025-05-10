import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from ._data_prep import _validate_and_dropna, _check_group_levels, _df_to_jax
from jax.scipy.stats import t as student_t

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
    Perform an independent two-sample t-test using JAX.

    :param measure: 1D JAX array of observations.
    :param labels: 1D JAX integer array of group labels (0..1).
    :param equal_var: Assume equal variances if True; Welch's otherwise.
    :return: Tuple of (t-statistic, p-value).
    """
    x = measure[labels == 0]
    y = measure[labels == 1]
    n1, n2 = x.shape[0], y.shape[0]
    m1, m2 = jnp.mean(x), jnp.mean(y)
    v1, v2 = jnp.var(x, ddof=1), jnp.var(y, ddof=1)
    if equal_var:
        pooled = ((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)
        se = jnp.sqrt(pooled * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        se = jnp.sqrt(v1/n1 + v2/n2)
        df = (v1/n1 + v2/n2)**2 / (((v1/n1)**2)/(n1-1) + ((v2/n2)**2)/(n2-1))
    t_stat = (m1 - m2) / se
    p_value = 2 * (1 - jax.scipy.stats.t.cdf(jnp.abs(t_stat), df))
    return t_stat, p_value