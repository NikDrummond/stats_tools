import numpy as np
import pytest
from scipy.stats import f_oneway
import jax.numpy as jnp
from stats_tools.core._parametric import _linear_regression, _one_way_anova. _t_test

def test_linear_regression_perfect_fit():
    # y = 2*x + 1 exactly
    x = np.linspace(0,5,6)
    y = 1 + 2*x
    # design matrix [1, x]
    X = np.vstack([np.ones_like(x), x]).T
    betas, ses, t_stats, p_vals = _linear_regression(X, y)
    # coefficients should be exactly [1, 2]
    assert pytest.approx(betas[0], abs=1e-6) == 1.0
    assert pytest.approx(betas[1], abs=1e-6) == 2.0
    # standard errors should be effectively zero
    assert ses[0] < 1e-6
    assert ses[1] < 1e-6
    # t-stats -> infinite or very large, p-values -> zero
    assert p_vals[1] < 1e-10

def test_linear_regression_intercept_only():
    # intercept-only model y = const
    y = np.array([5.0,5.0,5.0,5.0])
    X = np.ones((4,1))
    betas, ses, t_stats, p_vals = _linear_regression(X, y)
    assert pytest.approx(betas[0]) == 5.0
    assert ses[0] < 1e-6
    # df=3, so t_stat is infinite and p-val=0
    assert p_vals[0] < 1e-10

def test_one_way_anova_three_groups():
    # Group A mean=0, B mean=1, C mean=2, all σ=1
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, size=30)
    b = rng.normal(1, 1, size=30)
    c = rng.normal(2, 1, size=30)

    data = np.concatenate([a, b, c])
    # integer codes 0,1,2
    groups = np.concatenate([
        np.zeros_like(a, dtype=int),
        np.ones_like(b, dtype=int),
        np.full_like(c, 2, dtype=int),
    ])

    # our ANOVA
    F_jax, p_jax = _one_way_anova(jnp.array(groups), jnp.array(data))

    # reference from SciPy (two‐sided test)
    F_ref, p_ref = f_oneway(a, b, c)

    # compare F
    assert pytest.approx(float(F_jax), rel=1e-6) == F_ref
    # compare p (we’re both two‐sided)
    assert pytest.approx(float(p_jax), rel=1e-6) == p_ref

def _prep_ttest_data(a: np.ndarray, b: np.ndarray):
    """
    Helper to build the flat measure + label arrays for _t_test.
    """
    measure = np.concatenate([a, b])
    labels  = np.concatenate([np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])
    return jnp.array(measure), jnp.array(labels)

@pytest.mark.parametrize("equal_var", [True, False])
def test_t_test_against_scipy(equal_var):
    # small samples with known difference
    rng = np.random.RandomState(42)
    a = rng.normal(loc=0.0, scale=1.0, size=20)
    b = rng.normal(loc=0.5, scale=1.5, size=25)

    # get JAX result
    measure_jax, labels_jax = _prep_ttest_data(a, b)
    t_jax, p_jax = _t_test(measure_jax, labels_jax, equal_var=equal_var)
    t_jax, p_jax = float(t_jax), float(p_jax)

    # reference from SciPy
    t_ref, p_ref = ttest_ind(a, b, equal_var=equal_var)

    # two‐sided comparison
    assert pytest.approx(t_jax, rel=1e-6) == t_ref
    assert pytest.approx(p_jax, rel=1e-6) == p_ref

def test_t_test_one_sided_behaviour():
    # identical samples should give p_two_sided=1, so one‐sided = 0.5
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    measure_jax, labels_jax = _prep_ttest_data(a, b)

    # two‐sided
    t2, p2 = _t_test(measure_jax, labels_jax, equal_var=True)
    assert float(p2) == pytest.approx(1.0, abs=1e-8)