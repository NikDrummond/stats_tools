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
    measure: str,
    group: str
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """
    Convert pandas DataFrame columns to JAX arrays for analysis.

    :param df: pandas DataFrame after validation and NA dropping.
    :param measure: Name of the numeric outcome column.
    :param group: Name of the grouping column.
    :return: Tuple of
        - measure_arr: JAX array of shape (n_samples,)
        - label_arr: JAX integer array of group labels (0..k-1)
        - metadata: dict with mapping info (e.g., group levels)
    """
    measure_arr = jnp.array(df[measure].to_numpy())
    labels, levels = pd.factorize(df[group], sort=True)
    label_arr = jnp.array(labels)
    metadata = {"levels": list(levels), "n_groups": len(levels)}
    return measure_arr, label_arr, metadata