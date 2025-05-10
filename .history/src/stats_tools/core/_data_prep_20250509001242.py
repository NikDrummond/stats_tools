import pandas as pd
import jax.numpy as jnp
import warnings
from typing import Tuple, List, Dict, Any


def _validate_and_dropna(
    df: pd.DataFrame,
    columns: List[str]
) -> Tuple[pd.DataFrame, int]:
    """
    Validate that the DataFrame contains the specified columns and drop rows with NA in those columns.

    :param df: Input pandas DataFrame.
    :param columns: List of column names to validate and drop NAs from.
    :return: Tuple of (cleaned DataFrame, number of rows dropped).
    :raises ValueError: If any required column is missing.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Column(s) {missing} not found")

    initial_n = len(df)
    df_clean = df.dropna(subset=columns)
    n_dropped = initial_n - len(df_clean)
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} rows containing missing values in {columns}",
            UserWarning
        )
    return df_clean, n_dropped


def _check_group_levels(
    df: pd.DataFrame,
    group_col: str
) -> None:
    """
    Ensure that the grouping column has at least two unique levels.

    :param df: pandas DataFrame after NA removal.
    :param group_col: Name of the grouping column.
    :raises ValueError: If fewer than two distinct levels are present.
    """
    levels = df[group_col].unique()
    if levels.shape[0] < 2:
        raise ValueError(f"Grouping column '{group_col}' must have ≥2 levels")


def _df_to_jax(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """
    Convert a long‐format DataFrame into JAX arrays for grouping and values.

    :param df: DataFrame already cleaned of NAs in the target columns.
    :param group_col: Name of the grouping column.
    :param value_col: Name of the numeric outcome column.
    :return: (groups_jax, values_jax, metadata) where
      - groups_jax: integer codes for each row’s group, as jnp.ndarray
      - values_jax: the numeric values, as jnp.ndarray
      - metadata: dict containing the mapping { category: code }
    """
    # make sure columns exist
    for col in (group_col, value_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

    # turn groups into integer codes
    cat = pd.Categorical(df[group_col])
    codes = cat.codes  # -1 only if unseen, but we dropped NA already
    values = df[value_col].to_numpy()

    mapping = {cat.categories[i]: int(i) for i in range(len(cat.categories))}

    return jnp.array(codes), jnp.array(values), {"mapping": mapping}