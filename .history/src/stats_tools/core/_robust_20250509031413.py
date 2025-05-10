# src/stats_tools/core/_robust.py

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

# ───────────────────────────────────────────────────────────────────────────────
# Top-level helpers
# ───────────────────────────────────────────────────────────────────────────────

def _winsorize(
    x: jnp.ndarray,
    proportion_to_cut: float
) -> jnp.ndarray:
    n = x.shape[0]
    k = jnp.floor(proportion_to_cut * n).astype(int)
    x_s = jnp.sort(x)
    low, high = x_s[k], x_s[-k-1]
    return jnp.clip(x, low, high)


def _robust_eta_body(
    i: int,
    acc: float,
    groups: jnp.ndarray,
    data: jnp.ndarray,
    prop: float,
    overall_trim: float
) -> float:
    mask = groups == i
    grp_data = data[mask]  # mask is static per iteration
    # trimmed‐mean on grp_data
    n_i = grp_data.size
    k = jnp.floor(prop * n_i).astype(int)
    trimmed = jnp.mean(jnp.sort(grp_data)[k : n_i - k])
    return acc + n_i * (trimmed - overall_trim) ** 2


def _huber_loop_body(
    state: Tuple[jnp.ndarray, jnp.ndarray],
    _dummy: None,
    x: jnp.ndarray,
    c: float
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], None]:
    """
    Single IRLS step for Huber’s M-estimator.
    :param state: (mu, sigma)
    :param _dummy: unused scan element
    :param x: data vector
    :param c: Huber tuning constant
    :returns: (new_state, None)
    """
    mu, sigma = state
    resid = (x - mu) / sigma
    w = jnp.where(jnp.abs(resid) <= c, 1.0, c / jnp.abs(resid))
    mu_new = jnp.sum(w * x) / jnp.sum(w)
    sigma_new = jnp.sqrt(jnp.sum(w * (x - mu_new) ** 2) / jnp.sum(w))
    return (mu_new, sigma_new), None


# ───────────────────────────────────────────────────────────────────────────────
# Robust estimators
# ───────────────────────────────────────────────────────────────────────────────

+@partial(jax.jit, static_argnums=(1,))
+def trimmed_mean(x: jnp.ndarray, proportion_to_cut: float = 0.2) -> jnp.ndarray:
+    """
+    Compute the α‐trimmed mean of a 1D array via boolean masking.
+    """
+    # sanity check on the static trim fraction
+    if not (0.0 <= proportion_to_cut < 0.5):
+        raise ValueError("proportion_to_cut must be in [0, 0.5)")
+
+    n = x.shape[0]
+    k = jnp.floor(proportion_to_cut * n).astype(int)
+    x_sorted = jnp.sort(x)
+
+    # build a mask [0,1,...,n-1] selecting indices in [k, n-k)
+    idx = jnp.arange(n)
+    mask = (idx >= k) & (idx < (n - k))
+    selected = x_sorted * mask
+    # sum of selected plus zeros elsewhere, divided by count (n - 2*k)
+    count = n - 2 * k
+    return jnp.sum(selected) / count

@jax.jit
def huber_m_estimator(
    x: jnp.ndarray,
    c: float = 1.345,
    max_iter: int = 50
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Huber’s M‐estimator location and scale.
    """
    mu0 = jnp.median(x)
    sigma0 = jnp.median(jnp.abs(x - mu0)) / 0.6745

    # partial out x and c into the scan body
    body_fn = partial(_huber_loop_body, x=x, c=c)

    # run IRLS for max_iter iterations
    (mu_final, sigma_final), _ = lax.scan(
        body_fn,
        (mu0, sigma0),
        None,
        length=max_iter
    )
    return mu_final, sigma_final


@jax.jit
def robust_cohens_d(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    proportion_to_cut: float = 0.2
) -> jnp.ndarray:
    m1 = trimmed_mean(x1, proportion_to_cut)
    m2 = trimmed_mean(x2, proportion_to_cut)
    w1 = _winsorize(x1, proportion_to_cut)
    w2 = _winsorize(x2, proportion_to_cut)
    n1, n2 = w1.size, w2.size
    v1 = jnp.var(w1, ddof=1)
    v2 = jnp.var(w2, ddof=1)
    s_pool = jnp.sqrt(((n1 - 1)*v1 + (n2 - 1)*v2) / (n1 + n2 - 2))
    return (m1 - m2) / s_pool


@jax.jit
def robust_eta_squared(
    groups: jnp.ndarray,
    data: jnp.ndarray,
    proportion_to_cut: float = 0.2
) -> jnp.ndarray:
    overall_trim = trimmed_mean(data, proportion_to_cut)
    k = jnp.max(groups).astype(int) + 1
    body_fn = partial(
        _robust_eta_body,
        groups=groups,
        data=data,
        prop=proportion_to_cut,
        overall_trim=overall_trim
    )
    ss_between = lax.fori_loop(0, k, body_fn, 0.0)
    ss_total   = jnp.sum((data - overall_trim) ** 2)
    return ss_between / ss_total


@jax.jit
def robust_se_huber(x: jnp.ndarray, c: float = 1.345) -> jnp.ndarray:
    mu, sigma = huber_m_estimator(x, c)
    resid = (x - mu) / sigma
    w = jnp.where(jnp.abs(resid) <= c, 1.0, c / jnp.abs(resid))
    var_est = jnp.sum(w * (x - mu) ** 2) / (jnp.sum(w) ** 2)
    return jnp.sqrt(var_est * x.size)
