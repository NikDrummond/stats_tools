"""
Core statistical kernels for stats_module:
  - Data preparation (_data_prep)
  - Parametric tests and regression (_parametric)
  - Nonparametric tests (_nonparametric)
  - Permutation routines (_permutation)
  - Robust estimators (_robust)
  - Effect‐size calculations and CIs (_effect_sizes)
  - Multiple‐comparison adjustments (_adjustments)
"""

# Expose the submodules
__all__ = [
    "_data_prep",
    "_parametric",
    "_nonparametric",
    "_permutation",
    "_robust",
    "_effect_sizes",
    "_adjustments",
]

# And (optionally) re-export the most commonly used functions:
from ._data_prep        import _validate_and_dropna, _check_group_levels, _df_to_jax
from ._parametric       import _one_way_anova, _t_test, _linear_regression
from ._nonparametric    import _rank_data, _kruskal_wallis, _rank_biserial
from ._permutation      import _shuffle_labels, _null_distribution, empirical_p_value
from ._robust           import trimmed_mean, huber_m_estimator, robust_cohens_d, robust_eta_squared, robust_se_huber
from ._effect_sizes     import (
    cohens_d,
    eta_squared,
    epsilon_squared,
    rank_biserial,
    standardized_beta,
    partial_r_squared,
    analytic_ci,
    bootstrap_ci,
    permutation_ci,
)
from ._adjustments      import bonferroni, holm, fdr_bh, adjust_pvalues
