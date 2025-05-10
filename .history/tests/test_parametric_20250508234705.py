import numpy as np
import pytest
from stats_tools.core._parametric import _linear_regression

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
