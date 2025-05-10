import numpy as np
import pytest
import jax.numpy as jnp
from scipy.stats import trim_mean

from stats_tools.core._robust import (
    trimmed_mean,
    huber_m_estimator,
    robust_cohens_d,
    robust_eta_squared,
    robust_se_huber,
)

def test_trimmed_mean_no_trim():
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # α=0 => ordinary mean
    assert float(trimmed_mean(x, proportion_to_cut=0.0)) == pytest.approx(3.0)

def test_trimmed_mean_20pct():
    x = np.arange(1, 11)  # 1..10
    # scipy trim_mean cuts 20% each tail => cuts 2 values → mean of 3..8
    expected = trim_mean(x, 0.2)
    result = float(trimmed_mean(jnp.array(x), proportion_to_cut=0.2))
    assert result == pytest.approx(expected, rel=1e-6)

def test_trimmed_mean_invalid_alpha():
    x = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        trimmed_mean(x, proportion_to_cut=0.5)
    with pytest.raises(ValueError):
        trimmed_mean(x, proportion_to_cut=-0.1)

def test_huber_m_estimator_no_outliers():
    # data from a normal distribution
    rng = np.random.RandomState(0)
    x = rng.normal(0, 1, size=1000)
    mu, sigma = huber_m_estimator(jnp.array(x), c=1.345, max_iter=20)
    # should be close to sample median and scaled MAD
    assert float(mu) == pytest.approx(np.median(x), rel=1e-2)
    # sigma is approx MAD/0.6745
    mad = np.median(np.abs(x - np.median(x)))
    expected_sigma = mad / 0.6745
    assert float(sigma) == pytest.approx(expected_sigma, rel=1e-2)

def test_huber_m_estimator_with_outliers():
    # add extreme outliers
    rng = np.random.RandomState(1)
    x = rng.normal(0, 1, size=1000)
    x[::50] += 50
    mu, sigma = huber_m_estimator(jnp.array(x), c=1.345, max_iter=50)
    # robust location should still be near zero, not pulled to outliers
    assert abs(float(mu)) < 1.0
    # robust scale should be reasonable
    assert float(sigma) < 10.0

def test_robust_cohens_d_vs_standard():
    # two samples differing by 1
    a = np.zeros(50)
    b = np.ones(50)
    d_std = (a.mean() - b.mean()) / np.sqrt(((a.var(ddof=1)*(len(a)-1) +
                                             b.var(ddof=1)*(len(b)-1)) /
                                            (len(a)+len(b)-2)))
    # robust_cohens_d on identical clean data should ≈ standard d
    d_r = float(robust_cohens_d(jnp.array(a), jnp.array(b), proportion_to_cut=0.0))
    assert d_r == pytest.approx(d_std, rel=1e-6)

def test_robust_eta_squared_equal_groups():
    # three groups with equal values => η²=1
    data = jnp.array([0.0]*10 + [1.0]*10 + [2.0]*10)
    groups = jnp.array([0]*10 + [1]*10 + [2]*10)
    e2 = float(robust_eta_squared(groups, data, proportion_to_cut=0.0))
    assert e2 == pytest.approx(1.0, abs=1e-12)

def test_robust_eta_squared_identical_data():
    # all data identical => SS_between=0 => η²=0
    data = jnp.array([5.0]*20)
    groups = jnp.array([0]*10 + [1]*10)
    e2 = float(robust_eta_squared(groups, data, proportion_to_cut=0.2))
    assert e2 == pytest.approx(0.0, abs=1e-12)

def test_robust_se_huber_consistency():
    # on clean data, robust_se_huber ≈ std(x)/sqrt(n)
    rng = np.random.RandomState(2)
    x = rng.normal(0, 2, size=500)
    se_classic = np.std(x, ddof=1) / np.sqrt(len(x))
    se_robust = float(robust_se_huber(jnp.array(x), c=1.345))
    assert se_robust == pytest.approx(se_classic, rel=1e-2)

