import numpy as np
import pytest

import jax.numpy as jnp
from scipy.stats import kruskal, rankdata
from stats_tools.core._nonparametric import _rank_data, _kruskal_wallis, _rank_biserial

def test_rank_data_simple_no_ties():
    data = jnp.array([10, 20, 15, 30])
    ranks = _rank_data(data)
    # expected competition ranks: smallest=1, next=2, ...
    assert np.array_equal(np.array(ranks), np.array([1, 3, 2, 4]))

def test_rank_data_with_ties():
    data = jnp.array([5, 5, 10, 10, 1])
    # current impl does not average ties:
    # values [1,5,5,10,10] → competition ranks 1,2,3,4,5
    ranks = _rank_data(data)
    # assert set(np.array(ranks)) == {1,2,3,4,5}
    # if you switch to average‐tie, compare against scipy:
    assert np.allclose(np.array(ranks), rankdata(np.array(data), method="average"))

@pytest.mark.parametrize("sizes", [(10,10), (5,7)])
def test_kruskal_wallis_against_scipy(sizes):
    n1, n2 = sizes
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, size=n1)
    b = rng.normal(1, 1, size=n2)

    data = np.concatenate([a,b])
    labels = np.concatenate([np.zeros(n1,dtype=int), np.ones(n2,dtype=int)])

    H_jax, p_jax = _kruskal_wallis(jnp.array(data), jnp.array(labels))
    H_ref, p_ref = kruskal(a, b)

    assert pytest.approx(float(H_jax), rel=1e-6) == H_ref
    assert pytest.approx(float(p_jax), rel=1e-6) == p_ref

def test_kruskal_wallis_identical_groups():
    data = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    labels = jnp.array([0,0,1,1,2,2])
    H, p = _kruskal_wallis(data, labels)
    # if all ranks within groups are equal, H=0 and p=1
    assert float(H) == pytest.approx(0.0, abs=1e-12)
    assert float(p) == pytest.approx(1.0, abs=1e-12)

def test_rank_biserial_perfect_separation():
    # all group1 > group0
    measure = jnp.array([1,2,3, 10,11,12])
    labels  = jnp.array([0,0,0, 1,1,1])
    rbc = _rank_biserial(measure, labels)
    assert float(rbc) == pytest.approx(1.0, abs=1e-12)

+def test_rank_biserial_theoretical_value():
+    # alternating labels on ascending data => expected rbc = 1/3
+    measure = jnp.array([1,2,3,4,5,6])
+    labels  = jnp.array([0,1,0,1,0,1])
+    rbc = _rank_biserial(measure, labels)
+    assert float(rbc) == pytest.approx(1/3, rel=1e-6)