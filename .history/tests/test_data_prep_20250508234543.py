import pandas as pd
import pytest
import numpy as np

from stats_tools.core._data_prep import _validate_and_dropna, _check_group_levels, _df_to_jax

def test_validate_and_dropna_warns_and_drops():
    df = pd.DataFrame({
        "x": [1.0, np.nan, 3.0, 4.0],
        "g": ["A", "A", "B", None]
    })
    # we expect rows 1 and 3 to be dropped => 2 dropped
    with pytest.warns(UserWarning) as record:
        df_clean, n_dropped = _validate_and_dropna(df, ["x","g"])
    assert n_dropped == 2
    assert "2 rows removed" in str(record[0].message)
    # remaining DataFrame has no NAs
    assert not df_clean.isnull().any().any()

def test_validate_and_dropna_missing_column():
    df = pd.DataFrame({"x":[1,2,3]})
    with pytest.raises(ValueError):
        _validate_and_dropna(df, ["x","y"])

def test_check_group_levels_too_few():
    # only one unique level
    arr = pd.Categorical(["A","A"]).codes
    with pytest.raises(ValueError):
        _check_group_levels(arr, "g")
