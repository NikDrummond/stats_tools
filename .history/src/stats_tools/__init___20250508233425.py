# src/stats_tools/__init__.py

"""
stats_tools
===========

A high‐performance Python package for group‐comparison and covariate testing
on “long‐format” pandas DataFrames. Features:

- Parametric (t‐tests, ANOVA) and nonparametric (rank) tests
- Robust analogues (trimmed means, Huber M) and permutation methods
- Effect‐size estimation with analytic, bootstrap, or permutation CIs
- Built‐in multiple‐comparison corrections (Bonferroni, Holm, FDR)
- Unified JAX backend for speed and automatic differentiation
"""

__version__ = "0.1.0"

# High-level API
from .api import compare_groups, regress_with_groups

# Result classes
from .results import GroupTestResult, RegressionTestResult

__all__ = [
    "compare_groups",
    "regress_with_groups",
    "GroupTestResult",
    "RegressionTestResult",
]
