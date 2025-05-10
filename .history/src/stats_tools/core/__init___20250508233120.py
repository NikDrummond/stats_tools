"""
stats_module.core
-----------------
Core statistical kernels for data preparation, testing, and effect‐size estimation.

Submodules:
  - _data_prep      : data validation, NA‐handling, pandas→JAX conversion
  - _parametric     : one‐way ANOVA, two‐sample t‐test, linear regression
  - _nonparametric  : rank‐based kernels (ranks, Kruskal–Wallis, rank‐biserial)
  - _permutation    : label/residual shuffling, null‐distribution builder, empirical p-values
  - _robust         : trimmed means, Huber M‐estimator, robust effect sizes & SEs
  - _effect_sizes   : point estimators (Cohen’s d, η², etc.) plus CI wrappers
  - _adjustments    : Bonferroni, Holm, Benjamini–Hochberg corrections
"""

__all__ = [
    # submodules
    "_data_prep",
    "_parametric",
    "_nonparametric",
    "_permutation",
    "_robust",
    "_effect_sizes",
    "_adjustments",
]

# re-export key functions for convenient import
from ._data_prep       import _validate_and_dropna, _check_group_levels, _df_to_jax
from ._parametric      import _one_way_anova, _t_test, _linear_regression
from ._nonparametric   import _rank_data, _kruskal_wallis, _rank_biserial
from ._permutation     import _shuffle_labels, _null_distribution, empirical_p_value
from ._robust          import trimmed_mean, huber_m_estimator, robust_cohens_d, \
                              robust_eta_squared, robust_se_huber
from ._effect_sizes    import (
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
from ._adjustments     import bonferroni, holm, fdr_bh, adjust_pvalues
