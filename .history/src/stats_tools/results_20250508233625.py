# src/stats_tools/results.py

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict

import pandas as pd
import jax.numpy as jnp


@dataclass
class GroupTestResult:
    """
    Container for group‐comparison results.
    """
    method: str
    robust: bool
    n_permutations: int
    p_adjust_method: str
    n_dropped: int
    statistic: float
    p_value: float
    p_value_adj: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    permutation_stats: Optional[jnp.ndarray]
    metadata: Dict[str, Any]

    def summary(self) -> str:
        """
        Return a concise multi‐line summary of the test results.
        """
        lines = [
            f"Method: {self.method}",
            f"Robust: {self.robust}",
            f"Rows dropped: {self.n_dropped}",
            f"Statistic: {self.statistic:.4f}",
            f"P-value: {self.p_value:.4f}",
        ]
        if self.p_adjust_method and self.p_adjust_method != "none":
            lines.append(f"Adjusted p-value ({self.p_adjust_method}): {self.p_value_adj:.4f}")
        lines.append(f"Effect size: {self.effect_size:.4f}")
        if not (math.isnan(self.ci_lower) or math.isnan(self.ci_upper)):
            lines.append(f"CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the results as a pandas DataFrame (one row).
        """
        data = asdict(self)
        # Convert ndarray to list for DataFrame compatibility
        if self.permutation_stats is not None:
            data["permutation_stats"] = self.permutation_stats.tolist()
        return pd.DataFrame([data])


@dataclass
class RegressionTestResult:
    """
    Container for regression results.
    """
    outcome: str
    predictor: str
    group: Optional[str]
    method: str
    robust: bool
    n_permutations: int
    p_adjust_method: str
    n_dropped: int
    coef: float
    se: float
    t_stat: float
    p_value: float
    p_value_adj: Optional[float]
    effect_size: float
    ci_lower: float
    ci_upper: float
    permutation_stats: Optional[jnp.ndarray]
    metadata: Dict[str, Any]

    def summary(self) -> str:
        """
        Return a concise multi‐line summary of the regression results.
        """
        lines = [
            f"Outcome: {self.outcome}",
            f"Predictor: {self.predictor}",
        ]
        if self.group:
            lines.append(f"Group factor: {self.group}")
        lines.extend([
            f"Method: {self.method}",
            f"Robust: {self.robust}",
            f"Rows dropped: {self.n_dropped}",
            f"Coefficient: {self.coef:.4f} (SE: {self.se:.4f})",
            f"t-statistic: {self.t_stat:.4f}",
            f"P-value: {self.p_value:.4f}",
        ])
        if self.p_adjust_method and self.p_adjust_method != "none" and self.p_value_adj is not None:
            lines.append(f"Adjusted p-value ({self.p_adjust_method}): {self.p_value_adj:.4f}")
        lines.append(f"Effect size: {self.effect_size:.4f}")
        if not (math.isnan(self.ci_lower) or math.isnan(self.ci_upper)):
            lines.append(f"CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the results as a pandas DataFrame (one row).
        """
        data = asdict(self)
        if self.permutation_stats is not None:
            data["permutation_stats"] = self.permutation_stats.tolist()
        return pd.DataFrame([data])
