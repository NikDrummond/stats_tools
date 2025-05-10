# stats_module/api.py

from typing import Optional, Tuple, Union, Literal, Dict, Any
import warnings

import pandas as pd
import jax.numpy as jnp
import jax.random as jrand

from .core._data_prep import _validate_and_dropna, _check_group_levels, _df_to_jax
from .core._parametric import _one_way_anova, _t_test, _linear_regression
from .core._nonparametric import _kruskal_wallis
from .core._permutation import _null_distribution, empirical_p_value
from .core._robust import robust_cohens_d, robust_eta_squared
from .core._effect_sizes import (
    cohens_d,
    eta_squared,
    epsilon_squared,
    rank_biserial,
    analytic_ci,
    bootstrap_ci,
    permutation_ci,
    standardized_beta
)
from .core._adjustments import adjust_pvalues
from .results import GroupTestResult, RegressionTestResult


_PMethod = Literal["parametric", "rank", "permutation"]
_CIMethod = Literal["analytic", "bootstrap", "permutation"]
_PAdjust = Literal["none", "bonferroni", "holm", "fdr"]
_Alternative = Literal["two-sided", "greater", "less"]

def compare_groups(
    df: pd.DataFrame,
    measure: str,
    group: str,
    method: _PMethod = "parametric",
    robust: bool = False,
    n_permutations: int = 0,
    p_adjust: _PAdjust = "none",
    ci: bool = True,
    ci_method: _CIMethod = "analytic",
    alternative: _Alternative = "two-sided",
    random_state: Optional[int] = None,
) -> GroupTestResult:
    """
    Compare means/ranks across two or more groups.
    
    :param df: long‐format DataFrame.
    :param measure: name of the outcome column.
    :param group: name of the grouping column.
    :param method: "parametric" (ANOVA/t-test), "rank", or "permutation".
    :param robust: whether to use trimmed/Huber robust analogues.
    :param n_permutations: number of permutations if method="permutation".
    :param p_adjust: multiple-comparison correction for >2 groups.
    :param ci: whether to compute CI for the effect size.
    :param ci_method: "analytic", "bootstrap", or "permutation".
    :param random_state: seed for reproducibility.
    :return: GroupTestResult with statistics, p-values, effect sizes, CIs, etc.
    :raises ValueError: if inputs invalid.
    """
    # 1. validate columns & drop NA
    df_clean, n_dropped = _validate_and_dropna(df, [measure, group])
    if measure not in df_clean or group not in df_clean:
        raise ValueError(f"Column '{measure}' or '{group}' not found")
    # 2. extract arrays
    groups_jax, data_jax, metadata = _df_to_jax(df_clean, group_col=group, value_col=measure)
    # 3. check group levels
    _check_group_levels(groups_jax, group_name=group)
    k = int(jnp.unique(groups_jax).size)
    # 4. compute statistic & raw p-value
    if method == "parametric":
        if k == 2:
# compute two-sided t and p
            stat_jax, p_two = _t_test(data_jax, groups_jax, equal_var=not robust)
            stat = float(stat_jax)
            # convert to requested alternative
            if alternative == "two-sided":
                p_jax = p_two
            elif alternative == "greater":
                # P(T ≥ t_obs) = p_two/2 when t_obs ≥ 0, else = 1 - p_two/2
                p_jax = jnp.where(stat_jax >= 0, p_two * 0.5, 1.0 - p_two * 0.5)
            elif alternative == "less":
                # P(T ≤ t_obs) = p_two/2 when t_obs ≤ 0, else = 1 - p_two/2
                p_jax = jnp.where(stat_jax <= 0, p_two * 0.5, 1.0 - p_two * 0.5)
            else:
                raise ValueError(f"Unknown alternative: {alternative}")
            p_val = float(p_jax)

            # effect‐size
            x1 = data_jax[groups_jax == 0]
            x2 = data_jax[groups_jax == 1]
            eff = robust_cohens_d(x2, x1) if robust else cohens_d(x2, x1)
        else:
            stat, p_val = _one_way_anova(groups_jax, data_jax)
            eff = robust_eta_squared(groups_jax, data_jax) if robust else eta_squared(groups_jax, data_jax)
    elif method == "rank":
        stat, p_val = _kruskal_wallis(groups_jax, data_jax)
        if k == 2:
            x1, x2 = data_jax[groups_jax == 0], data_jax[groups_jax == 1]
            eff = rank_biserial(x2, x1)
        else:
            eff = epsilon_squared(groups_jax, data_jax)
    elif method == "permutation":
        # build null distribution of the same stat‐function
        stat_fn = (
            (lambda d, g: _t_test(d[g==0], d[g==1], equal_var=not robust)[0])
            if k == 2 and method == "parametric" else
            (lambda d, g: _one_way_anova(g, d)[0])
        )
        perm_stats = _null_distribution(stat_fn, data_jax, groups_jax,
                                        n_perm=n_permutations,
                                        seed=random_state or 0)
        p_val = empirical_p_value(stat_fn, data_jax, groups_jax,
                                  n_perm=n_permutations,
                                  seed=random_state or 0)
        stat = float(jnp.mean(perm_stats))  # or observed stat
        eff = jnp.nan  # effect size via separate call if desired
    else:
        raise ValueError(f"Unknown method: {method}")
    # 5. adjust p-values
    p_adj_array = adjust_pvalues(jnp.array([p_val]), method=p_adjust)
    p_val_adj = float(p_adj_array[0])
    # 6. compute CI for effect size
    ci_lower = ci_upper = jnp.nan
    if ci:
        if ci_method == "analytic":
            # analytic requires an SE; assume _effect_sizes provides an se function
            # here we use a placeholder se=1.0 / sqrt(n) for demo
            se_eff = 1.0 / jnp.sqrt(data_jax.size)
            ci_lower, ci_upper = analytic_ci(float(eff), float(se_eff))
        elif ci_method == "bootstrap":
            ci_lower, ci_upper = bootstrap_ci(
                lambda d, g: float(
                    (robust and robust_eta_squared(g, d)) or eta_squared(g, d)
                ),
                (data_jax, groups_jax),
                n_bootstraps=1000,
                random_state=random_state,
            )
        elif ci_method == "permutation":
            ci_lower, ci_upper = permutation_ci(
                lambda d, g: float(
                    (robust and robust_eta_squared(g, d)) or eta_squared(g, d)
                ),
                (data_jax, groups_jax),
                n_permutations=n_permutations or 1000,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown ci_method: {ci_method}")
    # 7. assemble result
    return GroupTestResult(
        method=method,
        robust=robust,
        n_permutations=n_permutations,
        p_adjust_method=p_adjust,
        n_dropped=n_dropped,
        statistic=float(stat),
        p_value=float(p_val),
        effect_size=float(eff),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        permutation_stats=(perm_stats if method=="permutation" else None),
        metadata={**metadata, "measure": measure, "group": group},
        p_value_adj=p_val_adj,
    )



def regress_with_groups(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    group: Optional[str] = None,
    method: _PMethod = "parametric",
    robust: bool = False,                    # not yet implemented
    n_permutations: int = 0,
    p_adjust: _PAdjust = "none",             # reserved for p-adjusting multiple terms
    ci: bool = True,
    ci_method: _CIMethod = "analytic",
    random_state: Optional[int] = None,
) -> RegressionTestResult:
    """
    Fit an OLS model (optionally with group dummies and interactions) and test the
    predictor term. Supports parametric inference, and permutation testing.
    """
    # 1. Validate & drop NAs
    cols = [outcome, predictor] + ([group] if group else [])
    df_clean, n_dropped = _validate_and_dropna(df, cols)
    for col in cols:
        if col not in df_clean:
            raise ValueError(f"Column '{col}' not found")

    # 2. Build design matrix
    X_pd = pd.DataFrame({"__intercept": 1.0, predictor: df_clean[predictor].astype(float)})
    if group:
        dummies = pd.get_dummies(df_clean[group], prefix="g", drop_first=True)
        X_pd = pd.concat([X_pd, dummies], axis=1)
        # interaction terms
        for col in dummies.columns:
            X_pd[f"ix_{col}"] = X_pd[predictor] * X_pd[col]

    X = jnp.asarray(X_pd.values)
    y = jnp.asarray(df_clean[outcome].astype(float).values)

    # 3. Fit via closed‐form OLS
    betas, ses, t_stats, p_values = _linear_regression(X, y)
    # predictor is at column index 1
    coef    = float(betas[1])
    se_coef = float(ses[1])
    t_stat  = float(t_stats[1])
    p_val   = float(p_values[1])

    # 4. Effect size: standardized β
    eff = float(standardized_beta(X[:, 1], y))

    # 5. Permutation inference (on t-stat) if requested
    perm_stats = None
    if method == "permutation" and n_permutations > 0:
        stat_fn = lambda Xmat, yvec: float(_linear_regression(Xmat, yvec)[2][1])
        perm_stats = _null_distribution(stat_fn, X, y,
                                        n_perm=n_permutations,
                                        seed=random_state or 0)
        p_val = empirical_p_value(stat_fn, X, y,
                                  n_perm=n_permutations,
                                  seed=random_state or 0)

    # 6. CI for the coefficient (or you could CI on effect size similarly)
    ci_lower = ci_upper = jnp.nan
    if ci:
        if ci_method == "analytic":
            df_resid = X.shape[0] - X.shape[1]
            ci_lower, ci_upper = analytic_ci(coef, se_coef, df=df_resid)
        elif ci_method == "bootstrap":
            ci_lower, ci_upper = bootstrap_ci(
                lambda Xmat, yvec: float(_linear_regression(Xmat, yvec)[0][1]),
                (X, y),
                n_bootstraps=1000,
                random_state=random_state,
            )
        elif ci_method == "permutation":
            ci_lower, ci_upper = permutation_ci(
                lambda Xmat, yvec: float(_linear_regression(Xmat, yvec)[0][1]),
                (X, y),
                n_permutations=n_permutations or 1000,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown ci_method: {ci_method}")

    # 7. Assemble result
    return RegressionTestResult(
        outcome=outcome,
        predictor=predictor,
        group=group,
        method=method,
        robust=robust,
        n_permutations=n_permutations,
        p_adjust_method=p_adjust,
        n_dropped=n_dropped,
        coef=coef,
        se=se_coef,
        t_stat=t_stat,
        p_value=p_val,
        p_value_adj=None,                   # fill in later if you add multi-term adjustment
        effect_size=eff,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        permutation_stats=perm_stats,
        metadata={"design_matrix_cols": list(X_pd.columns)},
    )
