import numpy as np
import pytest
import jax.numpy as jnp
from scipy import stats

from stats_tools.core._effect_sizes import (
    cohens_d, eta_squared, epsilon_squared,
    rank_biserial, standardized_beta, partial_r_squared,
    analytic_ci, bootstrap_ci, permutation_ci
)

def test_cohens_d_zero_difference():
    x = jnp.array([0.0, 1.0, 2.0])
    # same sample vs itself → d=0
    assert float(cohens_d(x, x)) == pytest.approx(0.0, abs=1e-12)

def test_cohens_d_known_value():
    # two small samples with known d
    a = jnp.array([0.0, 0.0, 1.0, 1.0])
    b = jnp.array([1.0, 1.0, 2.0, 2.0])
    # pooled sd = 1, raw d = mean(a)-mean(b) = -1
    # with Hedges’ J ~ 1 => d≈-1
    d = float(cohens_d(a, b))
    assert pytest.approx(d, rel=1e-6) == -1.0

def test_eta_squared_against_manual():
    # three groups with means [0,1,2], equal sizes, σ=0
    groups = jnp.array([0]*5 + [1]*5 + [2]*5)
    data = jnp.array([0.0]*5 + [1.0]*5 + [2.0]*5)
    # SS_between = Σ n (μ_g - μ)^2 = 5*((0-1)^2+(1-1)^2+(2-1)^2)=5*2=10
    # SS_total = Σ (x-μ)^2 = same = 10
    assert float(eta_squared(groups, data)) == pytest.approx(1.0, abs=1e-12)

def test_epsilon_squared_edge_case():
    # identical data → SS_between=0 → ε² negative or zero; manual ≈0
    groups = jnp.array([0,0,1,1])
    data = jnp.array([1.0,1.0,1.0,1.0])
    eps2 = float(epsilon_squared(groups, data))
    assert eps2 == pytest.approx(0.0, abs=1e-12)

def test_rank_biserial_simple():
    # reuse perfect‐separation test
    measure = jnp.array([1,2,3, 10,11,12])
    labels  = jnp.array([0,0,0, 1,1,1])
    assert float(rank_biserial(measure, labels)) == pytest.approx(1.0, abs=1e-12)

def test_standardized_beta_correlation():
    # generate data with known correlation 0.8
    rng = np.random.RandomState(0)
    X = rng.randn(100)
    y = 0.8 * X + rng.randn(100) * 0.2
    beta = float(standardized_beta(jnp.array(X), jnp.array(y)))
    # standardized_beta ≈ Pearson r
    r, _ = stats.pearsonr(X, y)
    assert pytest.approx(beta, rel=1e-2) == r

def test_partial_r_squared_simple():
    # y = x + noise; controlling for nothing (covariates empty)
    rng = np.random.RandomState(0)
    n = 50
    x = rng.randn(n)
    y = 2*x + rng.randn(n)*0.1
    # no covariates: partial R² ≈ R² of simple regression
    pr2 = float(partial_r_squared(jnp.array(x), jnp.array(y), jnp.empty((n,0))))
    r2 = stats.linregress(x,y).rvalue**2
    assert pytest.approx(pr2, rel=1e-2) == r2

def test_analytic_ci_normal():
    # for est=0, se=1, α=0.05 → z≈1.96
    lo, hi = analytic_ci(0.0, 1.0, alpha=0.05, df=None)
    assert lo < -1.95 and hi > 1.95

def test_bootstrap_ci_constant_effect():
    # effect function returns constant
    def eff_fn(x, g): return 0.5
    data = jnp.array([1,2,3,4])
    labels = jnp.array([0,1,0,1])
    lo, hi = bootstrap_ci(eff_fn, (data, labels), n_bootstraps=50, random_state=1)
    assert lo == pytest.approx(0.5, abs=1e-6)
    assert hi == pytest.approx(0.5, abs=1e-6)

def test_permutation_ci_constant_effect():
    def eff_fn(x, g): return 0.7
    data = jnp.array([1,2,3,4])
    labels = jnp.array([0,1,0,1])
    lo, hi = permutation_ci(eff_fn, (data, labels), n_permutations=50, random_state=2)
    assert lo == pytest.approx(0.7, abs=1e-6)
    assert hi == pytest.approx(0.7, abs=1e-6)
