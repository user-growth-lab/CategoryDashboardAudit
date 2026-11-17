"""P8: paired-user robustness audit and non-causal gap accounting.

This module tightens the KuaiRand comparison in several ways:

* a fixed-point scope with the same users and videos in both traffic buckets;
* a synchronized paired-user bootstrap (one multiplicity vector for both),
  including direct estimator-change intervals;
* exact symmetric item-mix/response-context decomposition, followed by
  absolute and centered Shapley *accounting*;
* sensitivity to the production coverage threshold, item-support restriction,
  and cell-equal versus exposure-weighted aggregation targets;
* a four-way early/late fixed point with common users and videos;
* outcome sensitivity for watch ratio, click, and long-view indicators; and
* sensitivity to official first-level category confidence >= 0.8.

Every decomposition below is descriptive accounting.  Neither component is a
causal effect, because traffic bucket can change user state, context, and other
unobserved conditions even after users and videos are paired.

Outputs (when :func:`main` is run):
``results/p8_paired_audit.json`` and
``results/fig8_paired_decomposition.png``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from audit_metrics import (
    CENTER_DEFINITION,
    eligible_production_rows,
    evaluate_category_estimate,
    paired_user_audit_scope,
    symmetric_category_gap_decomposition,
)
from data_utils import (
    RESULTS_DIR,
    load_kuairand_buckets,
    resolve_kuairand_category_path,
)


SEED = 42
B_USERS = 1000
MIN_CATEGORIES = 5
MIN_CATEGORY_SENSITIVITY = (1, 2, 3, 5, 10)
EARLY_DATES = (20220422, 20220430)
LATE_DATES = (20220501, 20220508)
OUTCOMES = ("watch_ratio", "is_click", "long_view")
TOPK = 10
B_TEMPORAL = 1000
ACCOUNTING_WARNING = (
    "Descriptive accounting only, not causal attribution: within-video "
    "residual and item-mix terms are algebraic components of an observed "
    "bucket gap."
)


METRIC_NAMES = (
    "absolute_mae",
    "centered_mae",
    "max_abs_error",
    "rank_corr",
    "top10_overlap",
)
ERROR_METRICS = ("absolute_mae", "centered_mae", "max_abs_error")


def _finite_outcome(frame: pd.DataFrame, outcome_col: str, label: str) -> None:
    if outcome_col not in frame:
        raise ValueError(f"{label} is missing outcome column {outcome_col!r}")
    values = pd.to_numeric(frame[outcome_col], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}.{outcome_col} contains missing/non-finite values")


def _category_metrics(
    estimate: pd.Series,
    reference: pd.Series,
    reference_shares: pd.Series,
    *,
    topk: int = TOPK,
) -> dict[str, float | int]:
    """Return the five consistently named category-vector metrics."""
    raw = evaluate_category_estimate(
        estimate, reference, reference_shares, topk=topk
    )
    effective_k = min(int(topk), len(reference))
    categories = list(reference.index)
    aligned_estimate = estimate.reindex(categories)
    aligned_reference = reference.reindex(categories)
    weights = reference_shares.reindex(categories).astype(float)
    weights /= weights.sum()
    estimate_center = float((aligned_estimate * weights).sum())
    reference_center = float((aligned_reference * weights).sum())
    absolute_error = (aligned_estimate - aligned_reference).abs()
    centered_error = (
        (aligned_estimate - estimate_center)
        - (aligned_reference - reference_center)
    ).abs()
    return {
        "absolute_mae": float(raw["mae"]),
        "centered_mae": float(raw["centered_shape_mae"]),
        "random_share_weighted_absolute_mae": float(
            (absolute_error * weights).sum()
        ),
        "random_share_weighted_centered_mae": float(
            (centered_error * weights).sum()
        ),
        "max_abs_error": float(raw["max_abs_error"]),
        "rank_corr": float(raw["rank_corr"]),
        "top10_overlap": float(raw[f"top{effective_k}_overlap"]),
        "topk_effective": effective_k,
    }


def _user_category_arrays(
    frame: pd.DataFrame,
    users: list,
    categories: list,
    outcome_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned per-user/category sums and counts."""
    _finite_outcome(frame, outcome_col, "bucket")
    user_index = {user: idx for idx, user in enumerate(users)}
    category_index = {cat: idx for idx, cat in enumerate(categories)}
    cells = (
        frame.groupby(["user_id", "cat1_name"], sort=False)[outcome_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    ui = cells["user_id"].map(user_index)
    ci = cells["cat1_name"].map(category_index)
    if ui.isna().any() or ci.isna().any():
        raise ValueError("frame contains users/categories outside aligned support")
    sums = np.zeros((len(users), len(categories)), dtype=float)
    counts = np.zeros_like(sums)
    np.add.at(sums, (ui.astype(int), ci.astype(int)), cells["sum"].to_numpy(float))
    np.add.at(
        counts,
        (ui.astype(int), ci.astype(int)),
        cells["count"].to_numpy(float),
    )
    return sums, counts


def paired_sufficient_statistics(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
) -> dict[str, object]:
    """Create aligned sufficient statistics for synchronized user resampling."""
    random_users = set(random["user_id"].unique())
    standard_users = set(standard["user_id"].unique())
    random_videos = set(random["video_id"].unique())
    standard_videos = set(standard["video_id"].unique())
    if random_users != standard_users:
        raise ValueError("paired bootstrap requires identical user sets")
    if random_videos != standard_videos:
        raise ValueError("paired bootstrap requires identical video sets")
    categories = sorted(
        set(random["cat1_name"].unique()), key=lambda value: str(value)
    )
    if set(categories) != set(standard["cat1_name"].unique()):
        raise ValueError("paired bootstrap requires identical category sets")
    users = sorted(random_users, key=lambda value: str(value))
    random_sums, random_counts = _user_category_arrays(
        random, users, categories, outcome_col
    )
    standard_sums, standard_counts = _user_category_arrays(
        standard, users, categories, outcome_col
    )
    return {
        "users": users,
        "categories": categories,
        "random_sums": random_sums,
        "random_counts": random_counts,
        "standard_sums": standard_sums,
        "standard_counts": standard_counts,
        "outcome": outcome_col,
    }


def estimates_from_user_multiplicity(
    stats: Mapping[str, object], multiplicity: np.ndarray
) -> dict[str, pd.Series]:
    """Evaluate reference, naive, and demean estimates for one paired draw."""
    categories = list(stats["categories"])
    rs = np.asarray(stats["random_sums"], dtype=float)
    rc = np.asarray(stats["random_counts"], dtype=float)
    ss = np.asarray(stats["standard_sums"], dtype=float)
    sc = np.asarray(stats["standard_counts"], dtype=float)
    weights = np.asarray(multiplicity, dtype=float)
    if weights.ndim != 1 or len(weights) != rs.shape[0]:
        raise ValueError("multiplicity must have one entry per paired user")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("multiplicity must be finite and non-negative")
    if weights.sum() <= 0:
        raise ValueError("multiplicity cannot be all zero")

    # Avoid BLAS/Accelerate's vector-matrix path here.  On some macOS NumPy
    # builds it emits spurious divide/overflow/invalid warnings even for these
    # small finite non-negative arrays.  Explicit weighted reduction is both
    # deterministic and numerically transparent for bootstrap multiplicities.
    column_weights = weights[:, None]
    random_sum = np.sum(rs * column_weights, axis=0, dtype=float)
    random_count = np.sum(rc * column_weights, axis=0, dtype=float)
    standard_sum = np.sum(ss * column_weights, axis=0, dtype=float)
    standard_count = np.sum(sc * column_weights, axis=0, dtype=float)
    if (random_count <= 0).any() or (standard_count <= 0).any():
        raise ValueError("resample lost at least one category")

    reference = random_sum / random_count
    reference_shares = random_count / random_count.sum()
    random_grand = float(random_sum.sum() / random_count.sum())
    naive = standard_sum / standard_count

    user_total_count = sc.sum(axis=1)
    if (user_total_count <= 0).any():
        raise ValueError("a paired user has no standard observations")
    user_mean = ss.sum(axis=1) / user_total_count
    observed = sc > 0
    cell_mean = np.divide(ss, sc, out=np.zeros_like(ss), where=observed)
    deviations = cell_mean - user_mean[:, None]
    weighted_observed = weights[:, None] * observed
    cell_denominator = weighted_observed.sum(axis=0)
    if (cell_denominator <= 0).any():
        raise ValueError("resample lost a standard user-category cell support")
    centered_demean = (weights[:, None] * deviations * observed).sum(axis=0)
    centered_demean /= cell_denominator
    demean = centered_demean + random_grand

    # Alternative aggregation target for sensitivity analysis.  It uses the
    # same user-specific row-weighted baselines and the same random reference
    # (including its grand-mean anchor), but averages residuals across
    # production exposure rows rather than equally across observed
    # user-category cells.
    residual_sums = ss - user_mean[:, None] * sc
    row_weighted_centered = np.sum(
        weights[:, None] * residual_sums, axis=0, dtype=float
    ) / standard_count
    demean_row_weighted = row_weighted_centered + random_grand

    index = pd.Index(categories, name="cat1_name")
    return {
        "reference": pd.Series(reference, index=index),
        "reference_shares": pd.Series(reference_shares, index=index),
        "naive": pd.Series(naive, index=index),
        "demean": pd.Series(demean, index=index),
        "demean_centered_cells": pd.Series(centered_demean, index=index),
        "demean_row_weighted": pd.Series(demean_row_weighted, index=index),
        "demean_row_weighted_centered": pd.Series(
            row_weighted_centered, index=index
        ),
    }


def point_paired_metrics(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
    topk: int = TOPK,
) -> dict[str, object]:
    """Point estimates for an already paired scope."""
    stats = paired_sufficient_statistics(
        random, standard, outcome_col=outcome_col
    )
    vectors = estimates_from_user_multiplicity(
        stats, np.ones(len(stats["users"]), dtype=float)
    )
    return {
        "n_users": len(stats["users"]),
        "n_videos": int(random["video_id"].nunique()),
        "n_categories": len(stats["categories"]),
        "naive": _category_metrics(
            vectors["naive"],
            vectors["reference"],
            vectors["reference_shares"],
            topk=topk,
        ),
        "demean": _category_metrics(
            vectors["demean"],
            vectors["reference"],
            vectors["reference_shares"],
            topk=topk,
        ),
        "category_vectors": {
            name: {str(key): float(value) for key, value in series.items()}
            for name, series in vectors.items()
        },
    }


def aggregation_target_sensitivity(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
    topk: int = TOPK,
) -> dict[str, object]:
    """Hold the random target fixed and change only residual aggregation."""
    stats = paired_sufficient_statistics(
        random, standard, outcome_col=outcome_col
    )
    vectors = estimates_from_user_multiplicity(
        stats, np.ones(len(stats["users"]), dtype=float)
    )
    reference = vectors["reference"]
    shares = vectors["reference_shares"]
    cell_equal = vectors["demean"]
    row_weighted = vectors["demean_row_weighted"]
    difference = row_weighted - cell_equal
    rank = cell_equal.rank().corr(row_weighted.rank())
    return {
        "definition": (
            "both estimators subtract each user's row-weighted production "
            "mean and add the identical randomized grand-mean anchor; only "
            "aggregation of production residuals changes from equal observed "
            "user-category cells to exposure rows"
        ),
        "random_reference_target_held_fixed": True,
        "point": {
            "cell_equal_user_category": _category_metrics(
                cell_equal, reference, shares, topk=topk
            ),
            "row_exposure_weighted": _category_metrics(
                row_weighted, reference, shares, topk=topk
            ),
        },
        "vector_comparison": {
            "row_weighted_minus_cell_equal_mae": float(
                difference.abs().mean()
            ),
            "row_weighted_minus_cell_equal_max_abs": float(
                difference.abs().max()
            ),
            "spearman_rank_corr": float(rank),
        },
        "category_vectors": {
            "cell_equal_user_category": {
                str(key): float(value) for key, value in cell_equal.items()
            },
            "row_exposure_weighted": {
                str(key): float(value) for key, value in row_weighted.items()
            },
            "row_weighted_minus_cell_equal": {
                str(key): float(value) for key, value in difference.items()
            },
        },
    }


def _flatten_metrics(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {
        f"{estimator}.{metric}": float(value)
        for estimator in ("naive", "demean")
        for metric, value in metrics[estimator].items()
        if metric != "topk_effective"
    }


def paired_user_bootstrap(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
    n_boot: int = B_USERS,
    seed: int = SEED,
    topk: int = TOPK,
    return_replicates: bool = False,
) -> dict[str, object]:
    """Synchronized paired-user bootstrap with complete-category retries.

    A single multinomial multiplicity vector is applied to each user's rows in
    *both* buckets.  This is the defining pairing; random and standard users
    are never resampled independently.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    stats = paired_sufficient_statistics(
        random, standard, outcome_col=outcome_col
    )
    n_users = len(stats["users"])
    point_vectors = estimates_from_user_multiplicity(
        stats, np.ones(n_users, dtype=float)
    )
    point_metrics = {
        estimator: _category_metrics(
            point_vectors[estimator],
            point_vectors["reference"],
            point_vectors["reference_shares"],
            topk=topk,
        )
        for estimator in ("naive", "demean")
    }

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    attempts = 0
    max_attempts = max(n_boot * 20, n_boot + 100)
    probabilities = np.repeat(1.0 / n_users, n_users)
    while len(rows) < n_boot and attempts < max_attempts:
        attempts += 1
        multiplicity = rng.multinomial(n_users, probabilities)
        try:
            vectors = estimates_from_user_multiplicity(stats, multiplicity)
        except ValueError as exc:
            if "resample lost" in str(exc):
                continue
            raise
        replicate_metrics = {
            estimator: _category_metrics(
                vectors[estimator],
                vectors["reference"],
                vectors["reference_shares"],
                topk=topk,
            )
            for estimator in ("naive", "demean")
        }
        row = _flatten_metrics(replicate_metrics)
        for metric in METRIC_NAMES:
            naive_value = float(replicate_metrics["naive"][metric])
            demean_value = float(replicate_metrics["demean"][metric])
            row[f"difference.{metric}"] = demean_value - naive_value
        for metric in ERROR_METRICS:
            naive_value = float(replicate_metrics["naive"][metric])
            demean_value = float(replicate_metrics["demean"][metric])
            row[f"relative_reduction.{metric}"] = (
                1.0 - demean_value / naive_value
                if naive_value != 0.0
                else float("nan")
            )
        rows.append(row)
    if len(rows) != n_boot:
        raise RuntimeError(
            f"only {len(rows)} complete bootstrap draws after {attempts} attempts"
        )

    replicates = pd.DataFrame(rows)

    def interval(estimator: str, metric: str) -> list[float]:
        values = replicates[f"{estimator}.{metric}"].to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            return [float("nan"), float("nan")]
        return [
            float(np.percentile(finite, 2.5)),
            float(np.percentile(finite, 97.5)),
        ]

    point_difference = {
        metric: float(point_metrics["demean"][metric])
        - float(point_metrics["naive"][metric])
        for metric in METRIC_NAMES
    }
    point_reduction = {
        metric: (
            1.0
            - float(point_metrics["demean"][metric])
            / float(point_metrics["naive"][metric])
            if float(point_metrics["naive"][metric]) != 0.0
            else None
        )
        for metric in ERROR_METRICS
    }

    def column_interval(column: str) -> list[float]:
        values = replicates[column].to_numpy(float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            return [None, None]
        return [
            float(np.percentile(finite, 2.5)),
            float(np.percentile(finite, 97.5)),
        ]

    output: dict[str, object] = {
        "design": {
            "unit": "paired user",
            "same_user_multiplicity_in_both_buckets": True,
            "category_complete_draws": True,
            "n_bootstrap": int(n_boot),
            "attempts": int(attempts),
            "seed": int(seed),
        },
        "point": point_metrics,
        "ci95": {
            estimator: {
                metric: interval(estimator, metric) for metric in METRIC_NAMES
            }
            for estimator in ("naive", "demean")
        },
        "naive_to_demean_metric_change": {
            "definition": "demean minus naive within each synchronized draw",
            "point": point_difference,
            "ci95": {
                metric: column_interval(f"difference.{metric}")
                for metric in METRIC_NAMES
            },
        },
        "relative_error_reduction": {
            "definition": "1 - demean / naive, evaluated within each draw",
            "scale": "fraction",
            "point": point_reduction,
            "ci95": {
                metric: column_interval(f"relative_reduction.{metric}")
                for metric in ERROR_METRICS
            },
        },
    }
    if return_replicates:
        output["replicates"] = rows
    return output


def min_category_sensitivity(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    thresholds: Iterable[int] = MIN_CATEGORY_SENSITIVITY,
    outcome_col: str = "watch_ratio",
    topk: int = TOPK,
) -> dict[str, object]:
    """Rebuild the paired fixed point for each production coverage threshold."""
    requested = tuple(int(value) for value in thresholds)
    if not requested or any(value < 1 for value in requested):
        raise ValueError("thresholds must contain positive integers")
    cells = {}
    base_standard_users = int(standard["user_id"].nunique())
    for threshold in requested:
        paired_random, paired_standard, scope = paired_user_audit_scope(
            random, standard, min_categories=threshold
        )
        cells[str(threshold)] = {
            "scope": scope,
            "paired_users_over_base_standard_users": float(
                paired_standard["user_id"].nunique() / base_standard_users
            ),
            "metrics": point_paired_metrics(
                paired_random,
                paired_standard,
                outcome_col=outcome_col,
                topk=topk,
            ),
        }
    return {
        "definition": (
            "for each threshold, rebuild production eligibility, paired-user "
            "intersection, and common-video fixed point from the same base buckets"
        ),
        "thresholds": list(requested),
        "cells": cells,
    }


def full_vs_common_video_support_sensitivity(
    random_full: pd.DataFrame,
    standard_full: pd.DataFrame,
    random_common: pd.DataFrame,
    standard_common: pd.DataFrame,
    *,
    min_categories: int = MIN_CATEGORIES,
    outcome_col: str = "watch_ratio",
    topk: int = TOPK,
) -> dict[str, object]:
    """Compare full bucket-specific video supports with common-video support.

    Users are fixed to the canonical common-support paired cohort.  The full
    cell retains every valid categorized video seen by those users in each
    bucket, so its video sets may differ and it is evaluated with generic
    category vectors rather than the item-paired bootstrap machinery.
    """
    users = set(random_common["user_id"].unique())
    if users != set(standard_common["user_id"].unique()):
        raise ValueError("common-support inputs must have identical user sets")
    full_random_users = random_full[random_full["user_id"].isin(users)].copy()
    full_standard_users = standard_full[standard_full["user_id"].isin(users)].copy()
    if set(full_random_users["user_id"].unique()) != users or set(
        full_standard_users["user_id"].unique()
    ) != users:
        raise ValueError("full-support inputs are missing a canonical paired user")
    coverage = full_standard_users.groupby("user_id")["cat1_name"].nunique()
    if (coverage < min_categories).any():
        raise ValueError("canonical paired users fail the full-support threshold")

    categories = sorted(
        set(full_random_users["cat1_name"].unique())
        & set(full_standard_users["cat1_name"].unique()),
        key=str,
    )
    full_random_users = full_random_users[
        full_random_users["cat1_name"].isin(categories)
    ].copy()
    full_standard_users = full_standard_users[
        full_standard_users["cat1_name"].isin(categories)
    ].copy()
    common_categories = sorted(random_common["cat1_name"].unique(), key=str)
    full_metrics = _scope_cell_metrics(
        full_random_users,
        full_standard_users,
        categories,
        outcome_col=outcome_col,
        topk=topk,
    )
    common_metrics = _scope_cell_metrics(
        random_common,
        standard_common,
        common_categories,
        outcome_col=outcome_col,
        topk=topk,
    )

    def support(frame):
        return {
            "rows": int(len(frame)),
            "users": int(frame["user_id"].nunique()),
            "videos": int(frame["video_id"].nunique()),
            "categories": int(frame["cat1_name"].nunique()),
        }

    retention = {
        bucket: {
            "row_fraction": float(len(common) / len(full)),
            "video_fraction": float(
                common["video_id"].nunique() / full["video_id"].nunique()
            ),
        }
        for bucket, full, common in (
            ("random", full_random_users, random_common),
            ("standard", full_standard_users, standard_common),
        )
    }
    metric_change = {
        estimator: {
            metric: float(common_metrics[estimator][metric])
            - float(full_metrics[estimator][metric])
            for metric in METRIC_NAMES
        }
        for estimator in ("naive", "demean")
    }
    return {
        "definition": (
            "canonical paired users are held fixed; full support retains each "
            "bucket's categorized videos separately, while common support "
            "restricts both buckets to the canonical shared-video fixed point"
        ),
        "estimand_note": (
            "the full-support cell includes between-bucket video entry/exit; "
            "the common-support cell conditions it away"
        ),
        "full_bucket_specific_support": {
            "random": support(full_random_users),
            "standard": support(full_standard_users),
        },
        "common_video_support": {
            "random": support(random_common),
            "standard": support(standard_common),
        },
        "common_over_full_retention": retention,
        "metrics": {
            "full_bucket_specific_support": full_metrics,
            "common_video_support": common_metrics,
        },
        "common_minus_full_metric_change": metric_change,
    }


def _generic_vectors(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    categories: list,
    outcome_col: str,
) -> dict[str, pd.Series]:
    """Vectors for possibly unpaired cohorts, used only in scope sensitivity."""
    _finite_outcome(random, outcome_col, "random")
    _finite_outcome(standard, outcome_col, "standard")
    reference = random.groupby("cat1_name")[outcome_col].mean().reindex(categories)
    counts = random["cat1_name"].value_counts().reindex(categories)
    naive = standard.groupby("cat1_name")[outcome_col].mean().reindex(categories)
    if reference.isna().any() or counts.isna().any() or naive.isna().any():
        raise ValueError("scope cell does not have complete category support")
    shares = counts / counts.sum()
    anchor = float((reference * shares).sum())

    cells = (
        standard.groupby(["user_id", "cat1_name"])[outcome_col]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    user_sum = cells.groupby("user_id")["sum"].transform("sum")
    user_count = cells.groupby("user_id")["count"].transform("sum")
    cells["demeaned"] = cells["mean"] - user_sum / user_count
    centered = cells.groupby("cat1_name")["demeaned"].mean().reindex(categories)
    if centered.isna().any():
        raise ValueError("scope cell lacks demeaned category support")
    return {
        "reference": reference,
        "reference_shares": shares,
        "naive": naive,
        "demean": centered + anchor,
    }


def _scope_cell_metrics(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    categories: list,
    *,
    outcome_col: str = "watch_ratio",
    topk: int = TOPK,
) -> dict[str, object]:
    vectors = _generic_vectors(random, standard, categories, outcome_col)
    return {
        estimator: _category_metrics(
            vectors[estimator],
            vectors["reference"],
            vectors["reference_shares"],
            topk=topk,
        )
        for estimator in ("naive", "demean")
    }


def two_factor_metric_shapley(
    old_old: float,
    paired_production_all_random: float,
    all_production_paired_random: float,
    paired_paired: float,
) -> dict[str, float]:
    """Shapley-account a two-factor metric change (scope, not causality)."""
    production = 0.5 * (
        (paired_production_all_random - old_old)
        + (paired_paired - all_production_paired_random)
    )
    reference = 0.5 * (
        (all_production_paired_random - old_old)
        + (paired_paired - paired_production_all_random)
    )
    total = paired_paired - old_old
    error = total - production - reference
    if not np.isclose(error, 0.0, rtol=0.0, atol=1e-12, equal_nan=False):
        raise AssertionError(f"scope Shapley accounting failed: {error}")
    return {
        "legacy_metric": float(old_old),
        "paired_metric": float(paired_paired),
        "total_change": float(total),
        "production_cohort_contribution": float(production),
        "random_reference_cohort_contribution": float(reference),
        "reconstruction_error": float(error),
    }


def scope_sensitivity_accounting(
    random_all: pd.DataFrame,
    standard_all: pd.DataFrame,
    random_paired: pd.DataFrame,
    standard_paired: pd.DataFrame,
    *,
    min_categories: int = MIN_CATEGORIES,
    outcome_col: str = "watch_ratio",
    topk: int = TOPK,
) -> dict[str, object]:
    """Compare the paired estimand with the legacy all-random reference.

    All four cells are evaluated on the final paired video/category support.
    The two toggles are (i) eligible-all versus paired production users and
    (ii) all-random versus paired-random users.  Shapley values account for the
    order-independent metric change; they are not causal contributions.
    """
    paired_videos = set(random_paired["video_id"].unique())
    categories = sorted(random_paired["cat1_name"].unique(), key=str)
    all_random_on_v = random_all[
        random_all["video_id"].isin(paired_videos)
        & random_all["cat1_name"].isin(categories)
    ].copy()
    all_standard_on_v = standard_all[
        standard_all["video_id"].isin(paired_videos)
        & standard_all["cat1_name"].isin(categories)
    ].copy()
    legacy_standard, coverage = eligible_production_rows(
        all_standard_on_v, min_categories=min_categories
    )

    cells = {
        "legacy_eligible_production__all_random_reference": _scope_cell_metrics(
            all_random_on_v,
            legacy_standard,
            categories,
            outcome_col=outcome_col,
            topk=topk,
        ),
        "paired_production__all_random_reference": _scope_cell_metrics(
            all_random_on_v,
            standard_paired,
            categories,
            outcome_col=outcome_col,
            topk=topk,
        ),
        "legacy_eligible_production__paired_random_reference": _scope_cell_metrics(
            random_paired,
            legacy_standard,
            categories,
            outcome_col=outcome_col,
            topk=topk,
        ),
        "paired_production__paired_random_reference": _scope_cell_metrics(
            random_paired,
            standard_paired,
            categories,
            outcome_col=outcome_col,
            topk=topk,
        ),
    }
    first = cells["legacy_eligible_production__all_random_reference"]
    second = cells["paired_production__all_random_reference"]
    third = cells["legacy_eligible_production__paired_random_reference"]
    fourth = cells["paired_production__paired_random_reference"]
    metric_names = (
        "absolute_mae",
        "centered_mae",
        "max_abs_error",
        "rank_corr",
        "top10_overlap",
    )
    shapley = {
        estimator: {
            metric: two_factor_metric_shapley(
                first[estimator][metric],
                second[estimator][metric],
                third[estimator][metric],
                fourth[estimator][metric],
            )
            for metric in metric_names
        }
        for estimator in ("naive", "demean")
    }
    return {
        "definition": (
            "four scope cells on final paired V/categories; legacy production "
            "uses all users meeting production coverage, while legacy reference "
            "uses every random-bucket user"
        ),
        "warning": ACCOUNTING_WARNING,
        "support": {
            "videos": len(paired_videos),
            "categories": len(categories),
            "legacy_eligible_production_users": int(
                legacy_standard["user_id"].nunique()
            ),
            "paired_users": int(standard_paired["user_id"].nunique()),
            "all_random_users": int(all_random_on_v["user_id"].nunique()),
            "legacy_coverage_min": int(
                coverage[coverage >= min_categories].min()
            ),
        },
        "metric_cells": cells,
        "metric_change_shapley_accounting": shapley,
    }


def shapley_absolute_accounting(
    first_component: pd.Series,
    second_component: pd.Series,
) -> pd.DataFrame:
    """Allocate ``abs(first + second)`` exactly with two-player Shapley values."""
    second = second_component.reindex(first_component.index)
    if second.isna().any():
        raise ValueError("Shapley components must share complete support")
    first = first_component.astype(float)
    second = second.astype(float)
    total = first + second
    first_value = 0.5 * (
        first.abs() + total.abs() - second.abs()
    )
    second_value = 0.5 * (
        second.abs() + total.abs() - first.abs()
    )
    reconstruction_error = total.abs() - first_value - second_value
    if not np.allclose(
        reconstruction_error.to_numpy(), 0.0, rtol=0.0, atol=1e-12
    ):
        raise AssertionError("absolute Shapley accounting failed")
    return pd.DataFrame(
        {
            "total": total,
            "absolute_total": total.abs(),
            "first_shapley": first_value,
            "second_shapley": second_value,
            "reconstruction_error": reconstruction_error,
        }
    )


def _paired_user_item_sparse_statistics(
    frame: pd.DataFrame,
    users: list,
    videos: list,
    *,
    outcome_col: str,
):
    """Return user-by-video sparse outcome sums and exposure counts."""
    from scipy import sparse

    _finite_outcome(frame, outcome_col, "bucket")
    user_index = {user: idx for idx, user in enumerate(users)}
    video_index = {video: idx for idx, video in enumerate(videos)}
    cells = (
        frame.groupby(["user_id", "video_id"], sort=False)[outcome_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    ui = cells["user_id"].map(user_index)
    vi = cells["video_id"].map(video_index)
    if ui.isna().any() or vi.isna().any():
        raise ValueError("frame contains users/videos outside aligned support")
    shape = (len(users), len(videos))
    indices = (ui.to_numpy(int), vi.to_numpy(int))
    sums = sparse.csr_matrix(
        (cells["sum"].to_numpy(float), indices), shape=shape
    )
    counts = sparse.csr_matrix(
        (cells["count"].to_numpy(float), indices), shape=shape
    )
    return sums, counts


def item_gap_accounting_bootstrap(
    standard: pd.DataFrame,
    random: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
    n_boot: int = B_USERS,
    seed: int = SEED,
    chunk_size: int = 25,
) -> dict[str, object]:
    """Paired positive-multiplier bootstrap for Shapley summary uncertainty.

    Strictly positive user-cluster weights preserve the point analysis' fixed
    video support in every draw.  This avoids declaring an item response to be
    zero merely because a multinomial draw omits the only sampled user who saw
    that item.  One weight vector is applied to both traffic buckets.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    random_users = set(random["user_id"].unique())
    standard_users = set(standard["user_id"].unique())
    random_videos = set(random["video_id"].unique())
    standard_videos = set(standard["video_id"].unique())
    if random_users != standard_users or random_videos != standard_videos:
        raise ValueError("accounting bootstrap requires paired users and videos")

    users = sorted(random_users, key=str)
    videos = sorted(random_videos, key=str)
    random_map = random.drop_duplicates("video_id").set_index("video_id")[
        "cat1_name"
    ]
    standard_map = standard.drop_duplicates("video_id").set_index("video_id")[
        "cat1_name"
    ]
    video_categories = random_map.reindex(videos)
    if video_categories.isna().any() or not video_categories.equals(
        standard_map.reindex(videos)
    ):
        raise ValueError("accounting bootstrap requires one shared video taxonomy")
    categories = sorted(video_categories.unique(), key=str)
    category_index = {category: idx for idx, category in enumerate(categories)}
    video_category_index = video_categories.map(category_index).to_numpy(int)

    from scipy import sparse

    category_indicator = sparse.csr_matrix(
        (
            np.ones(len(videos), dtype=float),
            (np.arange(len(videos)), video_category_index),
        ),
        shape=(len(videos), len(categories)),
    )
    random_sums, random_counts = _paired_user_item_sparse_statistics(
        random, users, videos, outcome_col=outcome_col
    )
    standard_sums, standard_counts = _paired_user_item_sparse_statistics(
        standard, users, videos, outcome_col=outcome_col
    )

    names = (
        "absolute_gap_mae",
        "absolute_response_context_shapley_mean",
        "absolute_item_mix_shapley_mean",
        "absolute_response_context_share",
        "absolute_item_mix_share",
        "centered_gap_mae",
        "centered_response_context_shapley_mean",
        "centered_item_mix_shapley_mean",
        "centered_response_context_share",
        "centered_item_mix_share",
    )
    draws = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    remaining = int(n_boot)
    while remaining:
        batch = min(int(chunk_size), remaining)
        weights = np.maximum(
            rng.exponential(size=(batch, len(users))),
            np.finfo(float).tiny,
        )

        def weighted_video_arrays(sums, counts):
            weighted_sums = np.asarray(sums.T.dot(weights.T), dtype=float).T
            weighted_counts = np.asarray(counts.T.dot(weights.T), dtype=float).T
            if (weighted_counts <= 0).any():  # pragma: no cover - positive weights
                raise AssertionError("positive multiplier lost fixed video support")
            means = weighted_sums / weighted_counts
            category_counts = np.asarray(
                category_indicator.T.dot(weighted_counts.T), dtype=float
            ).T
            shares = weighted_counts / category_counts[:, video_category_index]
            return means, shares, category_counts

        r_mean, r_share, r_category_counts = weighted_video_arrays(
            random_sums, random_counts
        )
        s_mean, s_share, _ = weighted_video_arrays(
            standard_sums, standard_counts
        )
        response_video = 0.5 * (s_share + r_share) * (s_mean - r_mean)
        mix_video = 0.5 * (s_mean + r_mean) * (s_share - r_share)
        response = np.asarray(
            category_indicator.T.dot(response_video.T), dtype=float
        ).T
        item_mix = np.asarray(
            category_indicator.T.dot(mix_video.T), dtype=float
        ).T

        def summarize(first, second, prefix):
            total = first + second
            first_value = 0.5 * (
                np.abs(first) + np.abs(total) - np.abs(second)
            )
            second_value = 0.5 * (
                np.abs(second) + np.abs(total) - np.abs(first)
            )
            gap = np.abs(total).mean(axis=1)
            first_mean = first_value.mean(axis=1)
            second_mean = second_value.mean(axis=1)
            draws[f"{prefix}_gap_mae"].extend(gap.tolist())
            draws[f"{prefix}_response_context_shapley_mean"].extend(
                first_mean.tolist()
            )
            draws[f"{prefix}_item_mix_shapley_mean"].extend(
                second_mean.tolist()
            )
            draws[f"{prefix}_response_context_share"].extend(
                (first_mean / gap).tolist()
            )
            draws[f"{prefix}_item_mix_share"].extend(
                (second_mean / gap).tolist()
            )

        summarize(response, item_mix, "absolute")
        reference_weights = r_category_counts / r_category_counts.sum(
            axis=1, keepdims=True
        )
        centered_response = response - np.sum(
            response * reference_weights, axis=1, keepdims=True
        )
        centered_item_mix = item_mix - np.sum(
            item_mix * reference_weights, axis=1, keepdims=True
        )
        summarize(centered_response, centered_item_mix, "centered")
        remaining -= batch

    def ci(values) -> list[float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):  # pragma: no cover - nonzero empirical gaps expected
            return [float("nan"), float("nan")]
        return [
            float(np.percentile(finite, 2.5)),
            float(np.percentile(finite, 97.5)),
        ]

    return {
        "design": {
            "unit": "paired user cluster",
            "method": "positive exponential multiplier bootstrap",
            "same_user_weight_in_both_buckets": True,
            "strictly_positive_weights_preserve_fixed_video_support": True,
            "n_bootstrap": int(n_boot),
            "seed": int(seed),
        },
        "component_display_names": {
            "response_context": "within-video residual",
            "item_mix": "item mix",
        },
        "ci95": {name: ci(values) for name, values in draws.items()},
    }


def item_gap_accounting(
    standard: pd.DataFrame,
    random: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
    n_boot: int = 0,
    seed: int = SEED,
) -> dict[str, object]:
    """Symmetric item decomposition plus exact absolute/centered accounting."""
    decomposition = symmetric_category_gap_decomposition(
        standard,
        random,
        outcome_col=outcome_col,
    )
    counts = random["cat1_name"].value_counts().reindex(decomposition.index)
    weights = counts / counts.sum()
    response = decomposition["response_context"]
    item_mix = decomposition["item_mix"]
    absolute = shapley_absolute_accounting(response, item_mix)
    centered_response = response - float((response * weights).sum())
    centered_item_mix = item_mix - float((item_mix * weights).sum())
    centered = shapley_absolute_accounting(centered_response, centered_item_mix)

    table = decomposition.copy()
    table["random_exposure_share"] = weights
    table["absolute_gap"] = absolute["absolute_total"]
    table["absolute_response_context_shapley"] = absolute["first_shapley"]
    table["absolute_item_mix_shapley"] = absolute["second_shapley"]
    table["centered_gap"] = centered["total"]
    table["absolute_centered_gap"] = centered["absolute_total"]
    table["centered_response_context_shapley"] = centered["first_shapley"]
    table["centered_item_mix_shapley"] = centered["second_shapley"]
    table["absolute_shapley_reconstruction_error"] = absolute[
        "reconstruction_error"
    ]
    table["centered_shapley_reconstruction_error"] = centered[
        "reconstruction_error"
    ]

    max_identity_error = float(
        max(
            table["reconstruction_error"].abs().max(),
            table["absolute_shapley_reconstruction_error"].abs().max(),
            table["centered_shapley_reconstruction_error"].abs().max(),
        )
    )
    if max_identity_error > 1e-10:
        raise AssertionError(f"item accounting identity error: {max_identity_error}")

    per_category = {
        str(category): {
            column: (int(value) if column == "n_videos" else float(value))
            for column, value in row.items()
        }
        for category, row in table.iterrows()
    }
    output = {
        "definition": (
            "symmetric within-category video residual and exposure-mix identity "
            "(the legacy response_context key denotes the within-video residual); "
            "two-player Shapley values allocate absolute gap after the identity, "
            "before and after fixed random-share centering"
        ),
        "warning": ACCOUNTING_WARNING,
        "component_display_names": {
            "response_context": "within-video residual",
            "item_mix": "item mix",
        },
        "center_definition": CENTER_DEFINITION,
        "summary_unweighted_across_categories": {
            "absolute_gap_mae": float(table["absolute_gap"].mean()),
            "absolute_response_context_shapley_mean": float(
                table["absolute_response_context_shapley"].mean()
            ),
            "absolute_item_mix_shapley_mean": float(
                table["absolute_item_mix_shapley"].mean()
            ),
            "centered_gap_mae": float(table["absolute_centered_gap"].mean()),
            "centered_response_context_shapley_mean": float(
                table["centered_response_context_shapley"].mean()
            ),
            "centered_item_mix_shapley_mean": float(
                table["centered_item_mix_shapley"].mean()
            ),
            "maximum_identity_error": max_identity_error,
        },
        "per_category": per_category,
    }
    summary = output["summary_unweighted_across_categories"]
    absolute_gap = float(summary["absolute_gap_mae"])
    centered_gap = float(summary["centered_gap_mae"])
    summary.update(
        {
            "absolute_response_context_share": float(
                summary["absolute_response_context_shapley_mean"] / absolute_gap
            ),
            "absolute_item_mix_share": float(
                summary["absolute_item_mix_shapley_mean"] / absolute_gap
            ),
            "centered_response_context_share": float(
                summary["centered_response_context_shapley_mean"] / centered_gap
            ),
            "centered_item_mix_share": float(
                summary["centered_item_mix_shapley_mean"] / centered_gap
            ),
        }
    )
    if n_boot:
        output["paired_user_multiplier_bootstrap"] = (
            item_gap_accounting_bootstrap(
                standard,
                random,
                outcome_col=outcome_col,
                n_boot=n_boot,
                seed=seed,
            )
        )
    return output


def _frame_support(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "videos": int(frame["video_id"].nunique()),
        "categories": int(frame["cat1_name"].nunique()),
    }


def four_way_temporal_scope(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    early_dates: tuple[int, int] = EARLY_DATES,
    late_dates: tuple[int, int] = LATE_DATES,
    full_period_min_categories: int = MIN_CATEGORIES,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Iterate to common users/videos in early/late x random/standard frames.

    ``standard`` must already be the full-period eligible paired cohort.  The
    eligibility decision is frozen before splitting: users are *not* required
    to cover ``full_period_min_categories`` in either half.  Imposing a second
    half-period threshold would select on temporal activity and change the
    prespecified full-period estimand.
    """
    if full_period_min_categories < 1:
        raise ValueError("full_period_min_categories must be positive")
    for label, frame in (("random", random), ("standard", standard)):
        needed = {"user_id", "video_id", "cat1_name", "date"}
        missing = needed.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} missing temporal columns: {sorted(missing)}")
    full_coverage = standard.groupby("user_id")["cat1_name"].nunique()
    if (full_coverage < full_period_min_categories).any():
        raise ValueError(
            "standard input is not the frozen full-period eligible cohort"
        )

    periods = {
        "early_random": random[
            pd.to_numeric(random["date"], errors="coerce").between(*early_dates)
        ].copy(),
        "early_standard": standard[
            pd.to_numeric(standard["date"], errors="coerce").between(*early_dates)
        ].copy(),
        "late_random": random[
            pd.to_numeric(random["date"], errors="coerce").between(*late_dates)
        ].copy(),
        "late_standard": standard[
            pd.to_numeric(standard["date"], errors="coerce").between(*late_dates)
        ].copy(),
    }
    if any(frame.empty for frame in periods.values()):
        empty = [name for name, frame in periods.items() if frame.empty]
        raise ValueError(f"empty temporal cells: {empty}")
    original_periods = periods
    initial = {name: _frame_support(frame) for name, frame in periods.items()}

    # The caller supplies the already paired full-period scope.  Start from
    # that full U/V state, not from a sequential four-cell intersection.
    random_users = set(random["user_id"].unique())
    standard_users = set(standard["user_id"].unique())
    random_videos = set(random["video_id"].unique())
    standard_videos = set(standard["video_id"].unique())
    if random_users != standard_users or random_videos != standard_videos:
        raise ValueError(
            "temporal input must be the full-period paired user/video scope"
        )
    common_users = random_users
    common_videos = random_videos

    def restrict(user_set, video_set):
        return {
            name: frame[
                frame["user_id"].isin(user_set)
                & frame["video_id"].isin(video_set)
            ].copy()
            for name, frame in original_periods.items()
        }

    periods = restrict(common_users, common_videos)

    def state_record(state_index, frames, user_set, video_set):
        return {
            "state_index": int(state_index),
            "common_users": len(user_set),
            "common_videos": len(video_set),
            **{f"{name}_rows": len(frame) for name, frame in frames.items()},
        }

    # State 0 is the frozen full-period paired U/V scope.  Every transition
    # computes U_next and V_next from the *same* four restricted strata, then
    # updates both sets together.  This makes the trajectory order invariant.
    iteration_log: list[dict[str, int]] = [
        state_record(0, periods, common_users, common_videos)
    ]
    max_iterations = len(common_users) + len(common_videos) + 2
    stability_check_iteration = None
    for transition in range(1, max_iterations + 1):
        next_users = set.intersection(
            *(set(frame["user_id"].unique()) for frame in periods.values())
        )
        next_videos = set.intersection(
            *(set(frame["video_id"].unique()) for frame in periods.values())
        )
        if not next_users:
            raise ValueError("four-way temporal fixed point lost all users")
        if not next_videos:
            raise ValueError("four-way temporal fixed point lost all videos")
        if next_users == common_users and next_videos == common_videos:
            stability_check_iteration = transition
            break
        common_users, common_videos = next_users, next_videos
        periods = restrict(common_users, common_videos)
        iteration_log.append(
            state_record(transition, periods, common_users, common_videos)
        )
    else:  # pragma: no cover
        raise RuntimeError("four-way temporal scope did not converge")

    user_sets = [set(frame["user_id"].unique()) for frame in periods.values()]
    video_sets = [set(frame["video_id"].unique()) for frame in periods.values()]
    category_sets = [set(frame["cat1_name"].unique()) for frame in periods.values()]
    if any(users != user_sets[0] for users in user_sets[1:]):
        raise AssertionError("temporal cells have different final user sets")
    if any(videos != video_sets[0] for videos in video_sets[1:]):
        raise AssertionError("temporal cells have different final video sets")
    if any(categories != category_sets[0] for categories in category_sets[1:]):
        raise AssertionError("temporal cells have different final category sets")
    metadata = {
        "definition": (
            "four traffic-period cells restricted iteratively to common users "
            "and videos; full-period eligibility is frozen before the split "
            "and no half-period category-coverage filter is applied; U_next "
            "and V_next are computed simultaneously from the same four strata"
        ),
        "early_dates": list(early_dates),
        "late_dates": list(late_dates),
        "full_period_min_categories": int(full_period_min_categories),
        "full_period_eligibility_frozen": True,
        "half_period_eligibility_filter": False,
        "initial": initial,
        "state_log_definition": (
            "state 0 is the full-period paired U/V scope; subsequent entries "
            "are changed simultaneous updates; the unchanged stability check "
            "is reported separately and not duplicated"
        ),
        "changed_updates_to_fixed_point": len(iteration_log) - 1,
        "stability_check_iteration": int(stability_check_iteration),
        "iteration_log": iteration_log,
        "final": {name: _frame_support(frame) for name, frame in periods.items()},
        "invariants": {
            "same_user_set_four_ways": True,
            "same_video_set_four_ways": True,
            "same_category_set_four_ways": True,
            "full_period_eligibility_frozen": True,
            "no_half_period_coverage_filter": True,
        },
    }
    return periods, metadata


def _vector_stability(
    early: pd.Series,
    late: pd.Series,
    *,
    topk: int = TOPK,
) -> dict[str, float | int]:
    late = late.reindex(early.index)
    if late.isna().any():
        raise ValueError("early/late vectors do not share category support")
    difference = late - early
    k = min(topk, len(early))
    return {
        "mean_abs_early_late_change": float(difference.abs().mean()),
        "max_abs_early_late_change": float(difference.abs().max()),
        "rank_corr_early_vs_late": float(early.rank().corr(late.rank())),
        "top10_overlap_early_vs_late": float(
            len(set(early.nlargest(k).index) & set(late.nlargest(k).index)) / k
        ),
        "topk_effective": int(k),
    }


def temporal_paired_analysis(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
    n_boot: int = B_USERS,
    seed: int = SEED,
    full_period_min_categories: int = MIN_CATEGORIES,
    topk: int = TOPK,
) -> dict[str, object]:
    """Four-way fixed-point temporal analysis with one joint paired bootstrap."""
    periods, scope = four_way_temporal_scope(
        random,
        standard,
        full_period_min_categories=full_period_min_categories,
    )
    early_stats = paired_sufficient_statistics(
        periods["early_random"],
        periods["early_standard"],
        outcome_col=outcome_col,
    )
    late_stats = paired_sufficient_statistics(
        periods["late_random"],
        periods["late_standard"],
        outcome_col=outcome_col,
    )
    if early_stats["users"] != late_stats["users"]:
        raise AssertionError("temporal sufficient statistics have different users")
    if early_stats["categories"] != late_stats["categories"]:
        raise AssertionError("temporal sufficient statistics have different categories")
    n_users = len(early_stats["users"])

    def evaluate_vectors(vectors):
        return {
            estimator: _category_metrics(
                vectors[estimator],
                vectors["reference"],
                vectors["reference_shares"],
                topk=topk,
            )
            for estimator in ("naive", "demean")
        }

    unit_multiplicity = np.ones(n_users, dtype=float)
    early_point_vectors = estimates_from_user_multiplicity(
        early_stats, unit_multiplicity
    )
    late_point_vectors = estimates_from_user_multiplicity(
        late_stats, unit_multiplicity
    )
    early_point_metrics = evaluate_vectors(early_point_vectors)
    late_point_metrics = evaluate_vectors(late_point_vectors)
    early_vectors = {
        key: value
        for key, value in early_point_vectors.items()
        if key != "reference_shares"
    }
    late_vectors = {
        key: value
        for key, value in late_point_vectors.items()
        if key != "reference_shares"
    }
    stability = {
        name: _vector_stability(
            early_vectors[name], late_vectors[name], topk=topk
        )
        for name in ("reference", "naive", "demean")
    }
    metric_names = (
        "absolute_mae",
        "centered_mae",
        "max_abs_error",
        "rank_corr",
        "top10_overlap",
    )
    changes = {
        estimator: {
            metric: float(late_point_metrics[estimator][metric])
            - float(early_point_metrics[estimator][metric])
            for metric in metric_names
        }
        for estimator in ("naive", "demean")
    }

    def gap_rank(early_values, late_values):
        value = early_values.rank().corr(late_values.rank())
        return float(value)

    gap_rank_point = {
        estimator: gap_rank(
            early_point_vectors[estimator] - early_point_vectors["reference"],
            late_point_vectors[estimator] - late_point_vectors["reference"],
        )
        for estimator in ("naive", "demean")
    }

    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    probabilities = np.repeat(1.0 / n_users, n_users)
    replicate_rows = []
    attempts = 0
    max_attempts = max(n_boot * 20, n_boot + 100)
    while len(replicate_rows) < n_boot and attempts < max_attempts:
        attempts += 1
        multiplicity = rng.multinomial(n_users, probabilities)
        try:
            early_draw_vectors = estimates_from_user_multiplicity(
                early_stats, multiplicity
            )
            late_draw_vectors = estimates_from_user_multiplicity(
                late_stats, multiplicity
            )
        except ValueError as exc:
            if "resample lost" in str(exc):
                continue
            raise
        early_draw = evaluate_vectors(early_draw_vectors)
        late_draw = evaluate_vectors(late_draw_vectors)
        row = {}
        for estimator in ("naive", "demean"):
            for metric in metric_names:
                early_value = float(early_draw[estimator][metric])
                late_value = float(late_draw[estimator][metric])
                row[f"early.{estimator}.{metric}"] = early_value
                row[f"late.{estimator}.{metric}"] = late_value
                row[f"difference.{estimator}.{metric}"] = late_value - early_value
            row[f"gap_rank.{estimator}"] = gap_rank(
                early_draw_vectors[estimator]
                - early_draw_vectors["reference"],
                late_draw_vectors[estimator]
                - late_draw_vectors["reference"],
            )
        replicate_rows.append(row)
    if len(replicate_rows) != n_boot:
        raise RuntimeError(
            f"only {len(replicate_rows)} temporal draws after {attempts} attempts"
        )
    replicate_frame = pd.DataFrame(replicate_rows)

    def ci(column):
        values = replicate_frame[column].to_numpy(float)
        values = values[np.isfinite(values)]
        if not len(values):
            return [float("nan"), float("nan")]
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    early_ci = {
        estimator: {
            metric: ci(f"early.{estimator}.{metric}") for metric in metric_names
        }
        for estimator in ("naive", "demean")
    }
    late_ci = {
        estimator: {
            metric: ci(f"late.{estimator}.{metric}") for metric in metric_names
        }
        for estimator in ("naive", "demean")
    }
    difference_ci = {
        estimator: {
            metric: ci(f"difference.{estimator}.{metric}")
            for metric in metric_names
        }
        for estimator in ("naive", "demean")
    }
    gap_rank_ci = {
        estimator: ci(f"gap_rank.{estimator}")
        for estimator in ("naive", "demean")
    }
    return {
        "scope": scope,
        "bootstrap_design": {
            "unit": "paired user",
            "same_user_multiplicity_in_all_four_strata": True,
            "n_bootstrap": int(n_boot),
            "attempts": int(attempts),
            "seed": int(seed),
        },
        "early": {"point": early_point_metrics, "ci95": early_ci},
        "late": {"point": late_point_metrics, "ci95": late_ci},
        "early_late_vector_stability": stability,
        "late_minus_early_metric_change": {
            "point": changes,
            "ci95": difference_ci,
        },
        "gap_vector_early_late_rank_corr": {
            "definition": (
                "Spearman correlation across category-wise estimator-minus-"
                "reference gaps in the early and late halves"
            ),
            "point": gap_rank_point,
            "ci95": gap_rank_ci,
        },
    }


def outcome_sensitivity(
    random_paired: pd.DataFrame,
    standard_paired: pd.DataFrame,
    *,
    outcomes: Iterable[str] = OUTCOMES,
    topk: int = TOPK,
) -> dict[str, object]:
    """Evaluate three outcomes on one fixed paired U/V scope."""
    results = {}
    for outcome in outcomes:
        results[outcome] = point_paired_metrics(
            random_paired,
            standard_paired,
            outcome_col=outcome,
            topk=topk,
        )
    return {
        "definition": (
            "same paired users/videos/categories for every outcome; primary "
            "watch_ratio is clipped to [0,1], while watch_ratio_unclipped "
            "retains play_time_ms / duration_ms when requested"
        ),
        "outcomes": results,
    }


def category_confidence_sensitivity(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    category_path: str | Path | None = None,
    threshold: float = 0.8,
    min_categories: int = MIN_CATEGORIES,
    topk: int = TOPK,
) -> dict[str, object]:
    """Rebuild the paired scope using only official labels with prob >= threshold."""
    source = resolve_kuairand_category_path(category_path)
    mapping = pd.read_csv(
        source,
        usecols=["final_video_id", "first_level_category_prob"],
    )
    mapping["final_video_id"] = pd.to_numeric(
        mapping["final_video_id"], errors="coerce"
    )
    mapping["first_level_category_prob"] = pd.to_numeric(
        mapping["first_level_category_prob"], errors="coerce"
    )
    high = mapping[
        mapping["final_video_id"].notna()
        & mapping["first_level_category_prob"].ge(float(threshold))
    ]
    high_ids = set(high["final_video_id"].astype("int64"))
    random_high = random[random["video_id"].isin(high_ids)].copy()
    standard_high = standard[standard["video_id"].isin(high_ids)].copy()
    paired_random, paired_standard, scope = paired_user_audit_scope(
        random_high,
        standard_high,
        min_categories=min_categories,
    )
    return {
        "definition": (
            "official supplementary first_level_category_prob threshold, then "
            "paired fixed-point scope rebuilt from the unpaired base buckets"
        ),
        "threshold": float(threshold),
        "mapping_rows_at_or_above_threshold": int(len(high)),
        "base_videos_retained_random_fraction": float(
            random_high["video_id"].nunique() / random["video_id"].nunique()
        ),
        "scope": scope,
        "metrics": point_paired_metrics(
            paired_random, paired_standard, topk=topk
        ),
    }


def make_figure(
    item_accounting: Mapping[str, object],
    temporal: Mapping[str, object],
    output_path: str | Path,
) -> None:
    """Create a compact two-panel decomposition/temporal figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use the shared released translations so a headless DejaVu render does
    # not silently replace Chinese glyphs with empty boxes.
    from category_labels import english_category_label as category_label

    plt.rcParams["font.family"] = "DejaVu Sans"

    per_category = pd.DataFrame.from_dict(
        item_accounting["per_category"], orient="index"
    )
    shown = per_category.loc[
        per_category["gap"].abs().nlargest(min(10, len(per_category))).index
    ].sort_values("gap")
    y = np.arange(len(shown))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    axes[0].barh(
        y,
        shown["response_context"],
        label="Within-video residual",
        color="#4C78A8",
    )
    axes[0].barh(
        y,
        shown["item_mix"],
        left=shown["response_context"],
        label="Item mix",
        color="#F58518",
    )
    axes[0].axvline(0, color="0.25", linewidth=0.8)
    axes[0].set_yticks(y, [category_label(str(value)) for value in shown.index])
    axes[0].set_xlabel("Observed standard − random gap")
    axes[0].set_title("(a) Symmetric descriptive accounting")
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=8,
    )

    labels = ["Naive\nabsolute", "Naive\ncentered", "Demean\nabsolute", "Demean\ncentered"]
    early = temporal["early"]["point"]
    late = temporal["late"]["point"]
    early_values = [
        early["naive"]["absolute_mae"],
        early["naive"]["centered_mae"],
        early["demean"]["absolute_mae"],
        early["demean"]["centered_mae"],
    ]
    late_values = [
        late["naive"]["absolute_mae"],
        late["naive"]["centered_mae"],
        late["demean"]["absolute_mae"],
        late["demean"]["centered_mae"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    axes[1].bar(x - width / 2, early_values, width, label="Apr 22–30")
    axes[1].bar(x + width / 2, late_values, width, label="May 1–8")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Category MAE")
    axes[1].set_title("(b) Four-way paired temporal audit")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.text(
        0.5,
        0.005,
        "Accounting is descriptive, not causal attribution.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _common_video_base_from_full(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    full_scope: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Reconstruct the loader's common-video base scope without a second read."""
    shared = set(random["video_id"].unique()) & set(standard["video_id"].unique())
    common_random = random[random["video_id"].isin(shared)].copy()
    common_standard = standard[standard["video_id"].isin(shared)].copy()
    categories = sorted(
        set(common_random["cat1_name"].unique())
        & set(common_standard["cat1_name"].unique()),
        key=str,
    )
    common_random = common_random[
        common_random["cat1_name"].isin(categories)
    ].copy()
    common_standard = common_standard[
        common_standard["cat1_name"].isin(categories)
    ].copy()
    scope = dict(full_scope)
    scope.update(
        {
            "common_videos_only": True,
            "n_common_videos": int(len(shared)),
            "n_common_categories": int(len(categories)),
            "random_rows": int(len(common_random)),
            "standard_rows": int(len(common_standard)),
            "random_users": int(common_random["user_id"].nunique()),
            "standard_users": int(common_standard["user_id"].nunique()),
            "random_videos": int(common_random["video_id"].nunique()),
            "standard_videos": int(common_standard["video_id"].nunique()),
        }
    )
    return (
        common_random.reset_index(drop=True),
        common_standard.reset_index(drop=True),
        scope,
    )


def run_p8(
    *,
    n_boot: int = B_USERS,
    temporal_n_boot: int = B_TEMPORAL,
    accounting_n_boot: int | None = None,
    seed: int = SEED,
) -> dict[str, object]:
    """Run the full P8 analysis in memory and return its JSON-ready result."""
    if accounting_n_boot is None:
        accounting_n_boot = n_boot
    random_full, standard_full, full_base_scope = load_kuairand_buckets(
        matched_period=True,
        tab=1,
        common_videos=False,
        extra_cols=("is_click", "long_view"),
    )
    for frame in (random_full, standard_full):
        frame["watch_ratio_unclipped"] = (
            frame["play_time_ms"] / frame["duration_ms"]
        )
    random, standard, base_scope = _common_video_base_from_full(
        random_full, standard_full, full_base_scope
    )
    paired_random, paired_standard, paired_scope = paired_user_audit_scope(
        random, standard, min_categories=MIN_CATEGORIES
    )
    bootstrap = paired_user_bootstrap(
        paired_random,
        paired_standard,
        n_boot=n_boot,
        seed=seed,
    )
    accounting = item_gap_accounting(
        paired_standard,
        paired_random,
        n_boot=accounting_n_boot,
        seed=seed,
    )
    temporal = temporal_paired_analysis(
        paired_random,
        paired_standard,
        n_boot=temporal_n_boot,
        seed=seed,
        full_period_min_categories=MIN_CATEGORIES,
    )
    result = {
        "analysis": "P8 paired-user audit and descriptive accounting",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "accounting_warning": ACCOUNTING_WARNING,
        "base_scope": base_scope,
        "paired_scope": paired_scope,
        "main_paired_metrics_and_bootstrap": bootstrap,
        "min_category_coverage_sensitivity": min_category_sensitivity(
            random, standard
        ),
        "full_vs_common_video_support_sensitivity": (
            full_vs_common_video_support_sensitivity(
                random_full,
                standard_full,
                paired_random,
                paired_standard,
            )
        ),
        "aggregation_target_sensitivity": aggregation_target_sensitivity(
            paired_random, paired_standard
        ),
        "legacy_scope_sensitivity": scope_sensitivity_accounting(
            random, standard, paired_random, paired_standard
        ),
        "item_gap_accounting": accounting,
        "temporal_halves": temporal,
        "outcome_sensitivity": outcome_sensitivity(
            paired_random,
            paired_standard,
            outcomes=(*OUTCOMES, "watch_ratio_unclipped"),
        ),
        "category_confidence_ge_0_8": category_confidence_sensitivity(
            random, standard, threshold=0.8
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=B_USERS)
    parser.add_argument("--temporal-bootstrap", type=int, default=B_TEMPORAL)
    parser.add_argument(
        "--accounting-bootstrap",
        type=int,
        default=None,
        help="defaults to --bootstrap",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = run_p8(
        n_boot=args.bootstrap,
        temporal_n_boot=args.temporal_bootstrap,
        accounting_n_boot=args.accounting_bootstrap,
        seed=args.seed,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "p8_paired_audit.json"
    figure_path = RESULTS_DIR / "fig8_paired_decomposition.png"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
    make_figure(result["item_gap_accounting"], result["temporal_halves"], figure_path)
    print(f"wrote {json_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
