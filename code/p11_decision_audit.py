"""P11: decision-aware audit of category dashboards.

The paired KuaiRand audit ultimately feeds ranked category decisions.  This
module therefore evaluates four dashboard vectors on the canonical paired
user/video scope:

``naive``
    Standard-serving category means.
``shift_only``
    Naive means plus the random-minus-standard grand-mean shift.  This changes
    levels but is ranking-identical to ``naive`` by construction.
``ce``
    The observed-cell-equal user-demeaned estimator from P8.
``rw``
    Its production-row-exposure-weighted sensitivity counterpart.

For k in {5, 10, 20}, reference regret is the average random-insertion outcome
of the reference top-k minus the average random-insertion outcome of the
candidate top-k.  Lower is better and the reference itself has zero regret.
Exact score ties are broken by ascending category string label.  This
makes selected sets reproducible; a boundary tie can lower overlap without
creating positive regret.

Inference uses one multinomial user multiplicity vector in both logs on every
draw, with the same seed and retry rule as P8.  Categories are fixed across
draws.  Nominal pointwise percentile intervals are retained for decision
metrics.  The six primary regret contrasts (CE-minus-naive and RW-minus-naive
at k in {5, 10, 20}) additionally receive a single family-wise standardized
bootstrap max-|t| interval.  The output also contains max-t simultaneous bands
for raw and random-share-centered signed category errors.

Outputs (when :func:`main` is run):
``results/p11_decision_audit.json`` and
``results/fig11_decision_audit.png``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from audit_metrics import CENTER_DEFINITION, paired_user_audit_scope
from data_utils import RESULTS_DIR, load_kuairand_buckets
from p8_paired_audit import (
    B_USERS,
    MIN_CATEGORIES,
    SEED,
    estimates_from_user_multiplicity,
    paired_sufficient_statistics,
)


TOP_KS = (5, 10, 20)
ESTIMATORS = ("naive", "shift_only", "ce", "rw")
RANKERS = ("reference", *ESTIMATORS)
STABLE_IN_THRESHOLD = 0.95
STABLE_OUT_THRESHOLD = 0.05


def _validate_series(values: pd.Series, label: str) -> pd.Series:
    """Return a finite float series with deterministic string labels."""
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    if values.empty:
        raise ValueError(f"{label} cannot be empty")
    if not values.index.is_unique:
        raise ValueError(f"{label} category labels must be unique")
    labels = [str(value) for value in values.index]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{label} category labels must have unique strings")
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{label} must contain only finite values")
    numeric.index = pd.Index(labels, name="cat1_name")
    return numeric


def deterministic_topk(values: pd.Series, k: int) -> list[str]:
    """Select top-k labels, breaking exact score ties lexicographically."""
    values = _validate_series(values, "ranking vector")
    if not isinstance(k, (int, np.integer)) or not 1 <= int(k) <= len(values):
        raise ValueError("k must be an integer between 1 and n_categories")
    ordered = sorted(
        values.items(),
        key=lambda item: (-float(item[1]), str(item[0])),
    )
    return [str(category) for category, _ in ordered[: int(k)]]


def decision_metrics(
    estimate: pd.Series,
    reference: pd.Series,
    *,
    k: int,
) -> dict[str, object]:
    """Compute reference regret and set overlap for one ranked dashboard.

    Regret is non-negative because the reference top-k maximizes the sum of
    reference outcomes among all size-k sets.  A tiny negative floating-point
    residual is clipped to zero; a substantive violation raises an error.
    """
    reference = _validate_series(reference, "reference")
    estimate = _validate_series(estimate, "estimate").reindex(reference.index)
    if estimate.isna().any():
        raise ValueError("estimate and reference must share category support")
    reference_selected = deterministic_topk(reference, k)
    estimate_selected = deterministic_topk(estimate, k)
    reference_value = float(reference.loc[reference_selected].mean())
    selected_value = float(reference.loc[estimate_selected].mean())
    regret = reference_value - selected_value
    if regret < -1e-12:
        raise AssertionError(f"reference regret cannot be negative: {regret}")
    regret = max(0.0, float(regret))
    overlap = len(set(reference_selected) & set(estimate_selected)) / int(k)
    return {
        "k": int(k),
        "selected_categories": estimate_selected,
        "reference_selected_categories": reference_selected,
        "reference_topk_mean": reference_value,
        "selected_set_reference_mean": selected_value,
        "reference_regret": regret,
        "topk_overlap": float(overlap),
    }


def inclusion_classification(
    probability: float,
    *,
    stable_in_threshold: float = STABLE_IN_THRESHOLD,
    stable_out_threshold: float = STABLE_OUT_THRESHOLD,
) -> str:
    """Classify a bootstrap top-k inclusion probability."""
    probability = float(probability)
    if not 0.0 <= stable_out_threshold < stable_in_threshold <= 1.0:
        raise ValueError("stability thresholds must satisfy 0 <= out < in <= 1")
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    if probability >= stable_in_threshold:
        return "stable_in"
    if probability <= stable_out_threshold:
        return "stable_out"
    return "uncertain"


def _standard_grand_mean(
    stats: Mapping[str, object], multiplicity: np.ndarray
) -> float:
    sums = np.asarray(stats["standard_sums"], dtype=float)
    counts = np.asarray(stats["standard_counts"], dtype=float)
    weights = np.asarray(multiplicity, dtype=float)[:, None]
    denominator = float(np.sum(counts * weights, dtype=float))
    if denominator <= 0:
        raise ValueError("resample has no standard rows")
    return float(np.sum(sums * weights, dtype=float) / denominator)


def dashboard_vectors_from_user_multiplicity(
    stats: Mapping[str, object], multiplicity: np.ndarray
) -> dict[str, pd.Series]:
    """Construct the reference and four candidate dashboard vectors."""
    p8 = estimates_from_user_multiplicity(stats, multiplicity)
    reference = _validate_series(p8["reference"], "reference")
    shares = _validate_series(p8["reference_shares"], "reference shares")
    shares = shares.reindex(reference.index)
    shares /= shares.sum()
    random_grand = float((reference * shares).sum())
    standard_grand = _standard_grand_mean(stats, multiplicity)
    naive = _validate_series(p8["naive"], "naive").reindex(reference.index)
    shift_only = naive + (random_grand - standard_grand)
    ce = _validate_series(p8["demean"], "ce").reindex(reference.index)
    rw = _validate_series(p8["demean_row_weighted"], "rw").reindex(
        reference.index
    )
    if any(vector.isna().any() for vector in (shares, naive, ce, rw)):
        raise ValueError("dashboard vectors do not share complete category support")
    return {
        "reference": reference,
        "reference_shares": shares,
        "naive": naive,
        "shift_only": shift_only,
        "ce": ce,
        "rw": rw,
    }


def _centered_signed_error(
    estimate: pd.Series, reference: pd.Series, shares: pd.Series
) -> pd.Series:
    """Signed estimate-reference error after fixed random-share centering."""
    error = estimate - reference
    return error - float((error * shares).sum())


def max_t_simultaneous_band(
    point: Sequence[float],
    bootstrap: np.ndarray,
    categories: Sequence[str],
    *,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Form a single-step max-t simultaneous band over fixed categories.

    Bootstrap standard deviations studentize category errors.  The critical
    value is the ``confidence`` quantile of the maximum absolute centered
    bootstrap t statistic.  Degenerate coordinates receive zero-width bands
    when every replicate equals the point value.
    """
    point_array = np.asarray(point, dtype=float)
    draws = np.asarray(bootstrap, dtype=float)
    labels = [str(value) for value in categories]
    if draws.ndim != 2 or draws.shape != (len(draws), len(point_array)):
        raise ValueError("bootstrap must be a draws-by-categories matrix")
    if len(draws) < 2:
        raise ValueError("max-t bands require at least two bootstrap draws")
    if len(labels) != len(point_array) or len(set(labels)) != len(labels):
        raise ValueError("categories must uniquely match point coordinates")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if not np.isfinite(point_array).all() or not np.isfinite(draws).all():
        raise ValueError("point and bootstrap values must be finite")

    standard_error = draws.std(axis=0, ddof=1)
    deviations = draws - point_array[None, :]
    active = standard_error > np.finfo(float).eps
    t_values = np.zeros_like(deviations)
    t_values[:, active] = (
        np.abs(deviations[:, active]) / standard_error[None, active]
    )
    if (~active).any() and not np.allclose(
        deviations[:, ~active], 0.0, atol=1e-12, rtol=0.0
    ):
        raise ValueError("zero-SE coordinate has nonzero bootstrap deviation")
    max_t = t_values.max(axis=1)
    critical = float(np.quantile(max_t, confidence))
    half_width = critical * standard_error
    lower = point_array - half_width
    upper = point_array + half_width
    return {
        "method": (
            "single-step bootstrap max-t band; bootstrap SD studentization; "
            "critical value is the confidence quantile of max absolute "
            "centered bootstrap t over the fixed category family"
        ),
        "confidence": float(confidence),
        "family_size": len(labels),
        "critical_value": critical,
        "categories_fixed_across_draws": True,
        "per_category": {
            category: {
                "point_signed_error": float(point_array[index]),
                "bootstrap_standard_error": float(standard_error[index]),
                "simultaneous_lower": float(lower[index]),
                "simultaneous_upper": float(upper[index]),
            }
            for index, category in enumerate(labels)
        },
    }


def ranking_stability_certificate(
    estimate: pd.Series,
    reference: pd.Series,
    reference_shares: pd.Series,
    *,
    k: int,
) -> dict[str, object]:
    """Evaluate a sufficient top-k margin condition.

    If the maximum absolute centered signed error is strictly less than half
    the random-reference gap between ranks k and k+1, no outside category can
    cross the boundary.  The condition is sufficient, not necessary.
    """
    reference = _validate_series(reference, "reference")
    estimate = _validate_series(estimate, "estimate").reindex(reference.index)
    shares = _validate_series(reference_shares, "reference shares").reindex(
        reference.index
    )
    if estimate.isna().any() or shares.isna().any():
        raise ValueError("certificate vectors must share category support")
    shares /= shares.sum()
    if not 1 <= int(k) < len(reference):
        raise ValueError("certificate requires 1 <= k < n_categories")

    ordered = deterministic_topk(reference, len(reference))
    kth_category = ordered[int(k) - 1]
    next_category = ordered[int(k)]
    gap = float(reference.loc[kth_category] - reference.loc[next_category])
    if gap < -1e-12:
        raise AssertionError("deterministic reference order is not descending")
    gap = max(gap, 0.0)
    centered_error = _centered_signed_error(estimate, reference, shares)
    max_error = float(centered_error.abs().max())
    half_gap = 0.5 * gap
    condition = bool(max_error < half_gap)
    reference_top = deterministic_topk(reference, int(k))
    estimate_top = deterministic_topk(estimate, int(k))
    return {
        "k": int(k),
        "reference_kth_category": kth_category,
        "reference_k_plus_one_category": next_category,
        "reference_boundary_gap": gap,
        "half_reference_boundary_gap": half_gap,
        "actual_max_abs_centered_signed_error": max_error,
        "condition_satisfied": condition,
        "condition_slack": float(half_gap - max_error),
        "observed_topk_exact": bool(reference_top == estimate_top),
        "observed_topk_set_exact": bool(set(reference_top) == set(estimate_top)),
        "condition_is_sufficient_not_necessary": True,
    }


def _percentile_interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("interval has no finite values")
    return [
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    ]


def max_t_simultaneous_contrast_intervals(
    point: Sequence[float],
    bootstrap: np.ndarray,
    contrast_labels: Sequence[str],
    *,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Construct a single-step family-wise interval for fixed contrasts.

    All contrasts must be evaluated on the same bootstrap draws.  Each
    coordinate is standardized by its across-draw bootstrap standard error;
    the common critical value is the requested quantile of the draw-wise
    maximum absolute standardized deviation from the original-sample point.
    This is the max-|t| (standardized max-statistic) bootstrap analogue used
    here because nested standard errors are not available for each resample.
    """
    point_array = np.asarray(point, dtype=float)
    draws = np.asarray(bootstrap, dtype=float)
    labels = [str(value) for value in contrast_labels]
    if point_array.ndim != 1 or not len(point_array):
        raise ValueError("point must be a non-empty one-dimensional vector")
    if draws.ndim != 2 or draws.shape[1] != len(point_array):
        raise ValueError("bootstrap must be a draws-by-contrasts matrix")
    if len(draws) < 2:
        raise ValueError("simultaneous intervals require at least two draws")
    if len(labels) != len(point_array) or len(set(labels)) != len(labels):
        raise ValueError("contrast labels must uniquely match point coordinates")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if not np.isfinite(point_array).all() or not np.isfinite(draws).all():
        raise ValueError("point and bootstrap values must be finite")

    standard_error = draws.std(axis=0, ddof=1)
    deviations = draws - point_array[None, :]
    active = standard_error > np.finfo(float).eps
    standardized = np.zeros_like(deviations)
    standardized[:, active] = (
        np.abs(deviations[:, active]) / standard_error[None, active]
    )
    if (~active).any() and not np.allclose(
        deviations[:, ~active], 0.0, atol=1e-12, rtol=0.0
    ):
        raise ValueError("zero-SE contrast has nonzero bootstrap deviation")
    max_abs_t = standardized.max(axis=1)
    critical = float(np.quantile(max_abs_t, confidence, method="linear"))
    half_width = critical * standard_error
    lower = point_array - half_width
    upper = point_array + half_width

    per_contrast = {}
    for index, label in enumerate(labels):
        excludes_zero = bool(lower[index] > 0.0 or upper[index] < 0.0)
        if upper[index] < 0.0:
            direction = "lower regret than naive"
        elif lower[index] > 0.0:
            direction = "higher regret than naive"
        else:
            direction = "not family-wise distinguishable from zero"
        per_contrast[label] = {
            "point_regret_contrast": float(point_array[index]),
            "bootstrap_standard_error": float(standard_error[index]),
            "simultaneous_ci95": [
                float(lower[index]),
                float(upper[index]),
            ],
            "excludes_zero": excludes_zero,
            "supported_direction": direction,
        }

    return {
        "method": (
            "single-step paired-user bootstrap standardized max-|t| interval; "
            "each contrast uses its across-draw bootstrap SD and the common "
            "critical value is the confidence quantile of the draw-wise "
            "maximum absolute standardized deviation from the point estimate"
        ),
        "confidence": float(confidence),
        "familywise_error_rate": float(1.0 - confidence),
        "family_size": len(labels),
        "n_shared_bootstrap_draws": int(len(draws)),
        "same_bootstrap_draws_for_all_contrasts": True,
        "critical_value": critical,
        "quantile_method": "linear empirical quantile",
        "contrast_order": labels,
        "per_contrast": per_contrast,
    }


def decision_aware_audit(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
    ks: Sequence[int] = TOP_KS,
    n_boot: int = B_USERS,
    seed: int = SEED,
) -> dict[str, object]:
    """Run the decision audit on an already paired user/video scope."""
    requested_ks = tuple(int(value) for value in ks)
    if not requested_ks or len(set(requested_ks)) != len(requested_ks):
        raise ValueError("ks must be a non-empty sequence of unique integers")
    if n_boot < 2:
        raise ValueError("n_boot must be at least two")
    stats = paired_sufficient_statistics(
        random, standard, outcome_col=outcome_col
    )
    categories = [str(value) for value in stats["categories"]]
    if any(not 1 <= value < len(categories) for value in requested_ks):
        raise ValueError("each k must satisfy 1 <= k < n_categories")
    n_users = len(stats["users"])
    point_vectors = dashboard_vectors_from_user_multiplicity(
        stats, np.ones(n_users, dtype=float)
    )

    point: dict[str, object] = {}
    for k in requested_ks:
        reference_selected = deterministic_topk(point_vectors["reference"], k)
        cell = {
            "reference": {
                "selected_categories": reference_selected,
                "reference_regret": 0.0,
                "topk_overlap": 1.0,
            }
        }
        for estimator in ESTIMATORS:
            cell[estimator] = decision_metrics(
                point_vectors[estimator], point_vectors["reference"], k=k
            )
        if cell["shift_only"]["selected_categories"] != cell["naive"][
            "selected_categories"
        ]:
            raise AssertionError("shift-only must preserve the naive ranking")
        point[str(k)] = cell

    n_categories = len(categories)
    raw_point_errors = {
        estimator: (
            point_vectors[estimator] - point_vectors["reference"]
        ).to_numpy(float)
        for estimator in ESTIMATORS
    }
    centered_point_errors = {
        estimator: _centered_signed_error(
            point_vectors[estimator],
            point_vectors["reference"],
            point_vectors["reference_shares"],
        ).to_numpy(float)
        for estimator in ESTIMATORS
    }
    if not np.allclose(
        centered_point_errors["naive"],
        centered_point_errors["shift_only"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("shift-only and naive centered errors must match")

    regret_draws = {
        str(k): {
            estimator: np.empty(n_boot, dtype=float) for estimator in ESTIMATORS
        }
        for k in requested_ks
    }
    overlap_draws = {
        str(k): {
            estimator: np.empty(n_boot, dtype=float) for estimator in ESTIMATORS
        }
        for k in requested_ks
    }
    inclusion_counts = {
        str(k): {
            ranker: np.zeros(n_categories, dtype=int) for ranker in RANKERS
        }
        for k in requested_ks
    }
    raw_error_draws = {
        estimator: np.empty((n_boot, n_categories), dtype=float)
        for estimator in ESTIMATORS
    }
    centered_error_draws = {
        estimator: np.empty((n_boot, n_categories), dtype=float)
        for estimator in ESTIMATORS
    }
    category_index = {category: index for index, category in enumerate(categories)}

    rng = np.random.default_rng(seed)
    probabilities = np.repeat(1.0 / n_users, n_users)
    attempts = 0
    completed = 0
    max_attempts = max(n_boot * 20, n_boot + 100)
    while completed < n_boot and attempts < max_attempts:
        attempts += 1
        multiplicity = rng.multinomial(n_users, probabilities)
        try:
            vectors = dashboard_vectors_from_user_multiplicity(stats, multiplicity)
        except ValueError as exc:
            if "resample lost" in str(exc):
                continue
            raise

        raw_errors = {
            estimator: vectors[estimator] - vectors["reference"]
            for estimator in ESTIMATORS
        }
        centered_errors = {
            estimator: _centered_signed_error(
                vectors[estimator],
                vectors["reference"],
                vectors["reference_shares"],
            )
            for estimator in ESTIMATORS
        }
        if not np.allclose(
            centered_errors["naive"].to_numpy(float),
            centered_errors["shift_only"].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError("bootstrap shift-only ranking invariant failed")
        for estimator in ESTIMATORS:
            raw_error_draws[estimator][completed] = raw_errors[
                estimator
            ].to_numpy(float)
            centered_error_draws[estimator][completed] = centered_errors[
                estimator
            ].to_numpy(float)

        for k in requested_ks:
            key = str(k)
            selected = {
                ranker: deterministic_topk(vectors[ranker], k)
                for ranker in RANKERS
            }
            # Preserve the mathematical shift invariant exactly in the
            # decision output, even in the presence of floating-point ties.
            selected["shift_only"] = list(selected["naive"])
            reference_value = float(
                vectors["reference"].loc[selected["reference"]].mean()
            )
            reference_set = set(selected["reference"])
            for ranker in RANKERS:
                for category in selected[ranker]:
                    inclusion_counts[key][ranker][category_index[category]] += 1
            for estimator in ESTIMATORS:
                candidate_value = float(
                    vectors["reference"].loc[selected[estimator]].mean()
                )
                regret = reference_value - candidate_value
                if regret < -1e-12:
                    raise AssertionError("bootstrap reference regret is negative")
                regret_draws[key][estimator][completed] = max(0.0, regret)
                overlap_draws[key][estimator][completed] = (
                    len(reference_set & set(selected[estimator])) / k
                )
        completed += 1

    if completed != n_boot:
        raise RuntimeError(
            f"only {completed} complete bootstrap draws after {attempts} attempts"
        )

    ci95: dict[str, object] = {}
    paired_change: dict[str, object] = {}
    for k in requested_ks:
        key = str(k)
        ci95[key] = {
            estimator: {
                "reference_regret": _percentile_interval(
                    regret_draws[key][estimator]
                ),
                "topk_overlap": _percentile_interval(
                    overlap_draws[key][estimator]
                ),
            }
            for estimator in ESTIMATORS
        }
        paired_change[key] = {}
        for estimator in ("shift_only", "ce", "rw"):
            regret_change = (
                regret_draws[key][estimator] - regret_draws[key]["naive"]
            )
            overlap_change = (
                overlap_draws[key][estimator] - overlap_draws[key]["naive"]
            )
            paired_change[key][estimator] = {
                "definition": "estimator minus naive within paired bootstrap draw",
                "interpretation": (
                    "negative reference-regret contrast favors the estimator"
                ),
                "point": {
                    "reference_regret": float(
                        point[key][estimator]["reference_regret"]
                        - point[key]["naive"]["reference_regret"]
                    ),
                    "topk_overlap": float(
                        point[key][estimator]["topk_overlap"]
                        - point[key]["naive"]["topk_overlap"]
                    ),
                },
                "ci95": {
                    "reference_regret": _percentile_interval(regret_change),
                    "topk_overlap": _percentile_interval(overlap_change),
                },
                "ci95_method": "paired-user bootstrap percentile interval",
                "ci95_scope": "nominal pointwise",
                "ci95_multiplicity_adjusted": False,
            }

    # One pre-specified family across both proposed demeaning estimators and
    # every requested decision cutoff.  Column stacking preserves the same
    # paired-user bootstrap draw in all coordinates.
    regret_contrast_labels = [
        f"{estimator}_minus_naive_k{k}"
        for k in requested_ks
        for estimator in ("ce", "rw")
    ]
    regret_contrast_points = np.asarray(
        [
            point[str(k)][estimator]["reference_regret"]
            - point[str(k)]["naive"]["reference_regret"]
            for k in requested_ks
            for estimator in ("ce", "rw")
        ],
        dtype=float,
    )
    regret_contrast_draws = np.column_stack(
        [
            regret_draws[str(k)][estimator] - regret_draws[str(k)]["naive"]
            for k in requested_ks
            for estimator in ("ce", "rw")
        ]
    )
    simultaneous_regret = max_t_simultaneous_contrast_intervals(
        regret_contrast_points,
        regret_contrast_draws,
        regret_contrast_labels,
    )
    simultaneous_regret.update(
        {
            "family_definition": (
                "all CE-minus-naive and RW-minus-naive reference-regret "
                "contrasts across the requested top-k cutoffs"
            ),
            "contrast_definition": (
                "estimator reference regret minus naive reference regret; "
                "negative favors the estimator"
            ),
            "unit": "paired user",
            "pointwise_intervals_retained_elsewhere_are_nominal": True,
        }
    )
    for k in requested_ks:
        key = str(k)
        for estimator in ("ce", "rw"):
            label = f"{estimator}_minus_naive_k{k}"
            simultaneous_regret["per_contrast"][label].update(
                {
                    "k": int(k),
                    "estimator": estimator,
                    "contrast": f"{estimator} minus naive",
                    "nominal_pointwise_percentile_ci95": paired_change[key][
                        estimator
                    ]["ci95"]["reference_regret"],
                }
            )

    inclusion_stability: dict[str, object] = {}
    for k in requested_ks:
        key = str(k)
        inclusion_stability[key] = {}
        for ranker in RANKERS:
            probabilities_by_category = inclusion_counts[key][ranker] / n_boot
            point_selected = set(
                point[key][ranker]["selected_categories"]
                if ranker != "reference"
                else point[key]["reference"]["selected_categories"]
            )
            per_category = {}
            grouped = {"stable_in": [], "stable_out": [], "uncertain": []}
            for index, category in enumerate(categories):
                probability = float(probabilities_by_category[index])
                classification = inclusion_classification(probability)
                grouped[classification].append(category)
                per_category[category] = {
                    "inclusion_probability": probability,
                    "classification": classification,
                    "point_selected": category in point_selected,
                }
            inclusion_stability[key][ranker] = {
                "summary_counts": {
                    label: len(values) for label, values in grouped.items()
                },
                "category_groups": grouped,
                "per_category": per_category,
            }

    bands = {}
    for estimator in ESTIMATORS:
        bands[estimator] = {
            "raw_signed_error": max_t_simultaneous_band(
                raw_point_errors[estimator],
                raw_error_draws[estimator],
                categories,
            ),
            "random_share_centered_signed_error": max_t_simultaneous_band(
                centered_point_errors[estimator],
                centered_error_draws[estimator],
                categories,
            ),
        }

    certificates = {
        str(k): {
            estimator: ranking_stability_certificate(
                point_vectors[estimator],
                point_vectors["reference"],
                point_vectors["reference_shares"],
                k=k,
            )
            for estimator in ESTIMATORS
        }
        for k in requested_ks
    }

    return {
        "definitions": {
            "reference_regret": (
                "mean random-reference outcome over the reference top-k minus "
                "mean random-reference outcome over the candidate top-k; "
                "lower is better and zero is optimal"
            ),
            "topk_overlap": "intersection size divided by k; higher is better",
            "tie_rule": (
                "sort by descending score, then ascending category string "
                "label for exact ties"
            ),
            "tie_note": (
                "a reference boundary tie may reduce set overlap while leaving "
                "reference regret at zero"
            ),
            "shift_only": (
                "naive category means plus random grand mean minus standard "
                "grand mean; ranking-identical to naive by construction"
            ),
            "ce": "P8 observed-cell-equal user-demeaned vector",
            "rw": "P8 production-row-exposure-weighted demean sensitivity",
            "centered_error": CENTER_DEFINITION,
            "stable_in": f"bootstrap inclusion probability >= {STABLE_IN_THRESHOLD}",
            "stable_out": f"bootstrap inclusion probability <= {STABLE_OUT_THRESHOLD}",
            "uncertain": (
                f"bootstrap inclusion probability in ({STABLE_OUT_THRESHOLD}, "
                f"{STABLE_IN_THRESHOLD})"
            ),
            "ranking_margin_proposition": (
                "if max absolute centered category error is strictly below "
                "half the reference rank-k/rank-(k+1) boundary gap, the top-k "
                "set is unchanged; this is sufficient, not necessary"
            ),
        },
        "design": {
            "unit": "paired user",
            "same_user_multiplicity_in_both_logs": True,
            "same_seed_and_draw_rule_as_p8": True,
            "category_complete_draws": True,
            "categories_fixed_across_draws": True,
            "n_bootstrap": int(n_boot),
            "attempts": int(attempts),
            "seed": int(seed),
            "ks": list(requested_ks),
            "n_users": int(n_users),
            "n_categories": int(n_categories),
            "familywise_regret_estimators": ["ce", "rw"],
            "familywise_regret_contrast_count": int(2 * len(requested_ks)),
        },
        "point": point,
        "ci95": ci95,
        "ci95_metadata": {
            "method": "paired-user bootstrap percentile interval",
            "scope": "nominal pointwise",
            "multiplicity_adjusted": False,
            "note": (
                "these original decision-metric intervals are descriptive "
                "pointwise intervals; use familywise_regret_contrasts_vs_naive "
                "for multiplicity-adjusted inference on the primary regret "
                "contrast family"
            ),
        },
        "paired_change_vs_naive": paired_change,
        "familywise_regret_contrasts_vs_naive": simultaneous_regret,
        "category_inclusion_stability": inclusion_stability,
        "simultaneous_signed_error_bands": bands,
        "ranking_stability_margin": certificates,
        "point_category_vectors": {
            name: {
                category: float(value)
                for category, value in point_vectors[name].items()
            }
            for name in ("reference", "reference_shares", *ESTIMATORS)
        },
        "invariants": {
            "shift_only_ranking_identical_to_naive": True,
            "shift_only_centered_error_identical_to_naive": True,
            "reference_regret_nonnegative": True,
            "fixed_category_family_for_simultaneous_bands": True,
            "same_draws_for_familywise_regret_contrasts": True,
            "familywise_regret_contrast_count": int(2 * len(requested_ks)),
        },
    }


def make_figure(result: Mapping[str, object], output_path: str | Path) -> None:
    """Plot decision regret, overlap, and k=10 inclusion stability."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from category_labels import english_category_label as category_label

    plt.rcParams["font.family"] = "DejaVu Sans"
    labels = {
        "naive": "Naive",
        "shift_only": "Shift only",
        "ce": "CE",
        "rw": "RW",
    }
    colors = {
        "naive": "#4C78A8",
        "shift_only": "#9C9C9C",
        "ce": "#F58518",
        "rw": "#54A24B",
    }
    # Small horizontal offsets keep ranking-identical naive/shift-only points
    # visible instead of silently drawing one directly on top of the other.
    offsets = {"naive": -0.24, "shift_only": -0.08, "ce": 0.08, "rw": 0.24}
    ks = [int(value) for value in result["design"]["ks"]]
    fig = plt.figure(figsize=(13.2, 8.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]

    for estimator in ESTIMATORS:
        plot_k = np.asarray(ks, dtype=float) + offsets[estimator]
        regret = np.array(
            [result["point"][str(k)][estimator]["reference_regret"] for k in ks]
        )
        regret_ci = np.array(
            [result["ci95"][str(k)][estimator]["reference_regret"] for k in ks]
        )
        regret_err = np.vstack([regret - regret_ci[:, 0], regret_ci[:, 1] - regret])
        axes[0].errorbar(
            plot_k,
            regret,
            yerr=np.maximum(regret_err, 0.0),
            marker="o",
            capsize=3,
            linewidth=1.4,
            label=labels[estimator],
            color=colors[estimator],
        )
        overlap = np.array(
            [result["point"][str(k)][estimator]["topk_overlap"] for k in ks]
        )
        overlap_ci = np.array(
            [result["ci95"][str(k)][estimator]["topk_overlap"] for k in ks]
        )
        overlap_err = np.vstack(
            [overlap - overlap_ci[:, 0], overlap_ci[:, 1] - overlap]
        )
        axes[1].errorbar(
            plot_k,
            overlap,
            yerr=np.maximum(overlap_err, 0.0),
            marker="o",
            capsize=3,
            linewidth=1.4,
            label=labels[estimator],
            color=colors[estimator],
        )
    axes[0].set_title(
        "(a) Reference regret (nominal pointwise 95% CIs; lower is better)"
    )
    axes[0].set_xlabel("Top-k decision")
    axes[0].set_ylabel("Reference regret")
    axes[0].set_xticks(ks)
    axes[0].axhline(0, color="0.25", linewidth=0.7)
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].set_title(
        "(b) Reference top-k overlap (nominal pointwise 95% CIs)"
    )
    axes[1].set_xlabel("Top-k decision")
    axes[1].set_ylabel("Overlap")
    axes[1].set_xticks(ks)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.2)

    heat_ax = fig.add_subplot(grid[1, :])
    heat_k = 10 if 10 in ks else ks[0]
    reference_vector = result["point_category_vectors"]["reference"]
    ordered_categories = deterministic_topk(
        pd.Series(reference_vector, dtype=float), len(reference_vector)
    )
    heat_rankers = ("reference", "naive", "shift_only", "ce", "rw")
    heat = np.array(
        [
            [
                result["category_inclusion_stability"][str(heat_k)][ranker][
                    "per_category"
                ][category]["inclusion_probability"]
                for category in ordered_categories
            ]
            for ranker in heat_rankers
        ]
    )
    image = heat_ax.imshow(heat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    heat_ax.set_title(
        f"(c) Paired-bootstrap top-{heat_k} inclusion probability "
        "(categories ordered by reference mean)"
    )
    heat_ax.set_yticks(
        np.arange(len(heat_rankers)),
        ["Reference", "Naive", "Shift only", "CE", "RW"],
    )
    heat_ax.set_xticks(np.arange(len(ordered_categories)))
    heat_ax.set_xticklabels(
        [category_label(category) for category in ordered_categories],
        rotation=70,
        ha="right",
        fontsize=6.5,
    )
    colorbar = fig.colorbar(image, ax=heat_ax, fraction=0.018, pad=0.015)
    colorbar.set_label("Inclusion probability")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def run_p11(
    *,
    root: str | Path | None = None,
    category_path: str | Path | None = None,
    n_boot: int = B_USERS,
    seed: int = SEED,
) -> dict[str, object]:
    """Load canonical data, build the P8 paired scope, and run P11."""
    random, standard, base_scope = load_kuairand_buckets(
        root,
        category_path=category_path,
        matched_period=True,
        tab=1,
        common_videos=True,
    )
    paired_random, paired_standard, paired_scope = paired_user_audit_scope(
        random, standard, min_categories=MIN_CATEGORIES
    )
    audit = decision_aware_audit(
        paired_random,
        paired_standard,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "analysis": "P11 decision-aware paired dashboard audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "base_scope": base_scope,
        "paired_scope": paired_scope,
        **audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=B_USERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--category-path", type=Path, default=None)
    args = parser.parse_args()
    result = run_p11(
        root=args.data_root,
        category_path=args.category_path,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "p11_decision_audit.json"
    figure_path = RESULTS_DIR / "fig11_decision_audit.png"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    make_figure(result, figure_path)
    print(f"wrote {json_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
