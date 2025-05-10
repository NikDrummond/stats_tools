import pandas as pd
import numpy as np
import pytest
from stats_tools.api import compare_groups, regress_with_groups

def make_group_df():
    # two groups with different means
    np.random.seed(0)
    return pd.DataFrame({
        "score": np.concatenate([np.random.normal(0,1,20), np.random.normal(1,1,20)]),
        "grp": ["A"]*20 + ["B"]*20
    })

def test_compare_groups_ttest():
    df = make_group_df()
    res = compare_groups(df, measure="score", group="grp",
                         method="parametric", robust=False,
                         ci=False, alternative = )
    assert res.method == "parametric"
    assert res.effect_size > 0  # B > A
    assert 0 < res.p_value < 0.1

def test_regress_with_groups_simple():
    # y = 3*x + noise
    df = pd.DataFrame({
        "y": [3*i for i in range(10)],
        "x": list(range(10))
    })
    res = regress_with_groups(df, outcome="y", predictor="x",
                              method="parametric", ci=False)
    assert pytest.approx(res.coef, rel=1e-3) == 3.0
    assert res.p_value < 1e-6
