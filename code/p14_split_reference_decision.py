"""P14: leakage-free split-reference decision sensitivity.

P11 uses the same randomized category means both to define its reference
top-k and to evaluate candidate top-k sets.  That is a useful descriptive
audit, but reusing a noisy reference can make its decision regret optimistic.
This module freezes a deterministic split of randomized impressions:

* the selection half alone defines the reference top-k;
* the disjoint evaluation half alone values that set and every candidate set;
* naive, CE, and RW candidate rankings are estimated from the complete
  standard-serving bucket, never from either randomized outcome half; and
* one paired-user bootstrap multiplicity vector is applied synchronously to
  the selection, evaluation, and standard sufficient statistics.

Because the selection-half reference set need not maximize the independent
evaluation-half outcome, split-reference regret is signed and may be negative.
Lower remains better: a negative value means that a candidate set has higher
held-out value than the selection-half reference set.

The frozen split hashes an outcome-free event signature.  KuaiRand-Pure does
not expose a request/event identifier, so repeated rows with an identical
observable signature are deliberately kept on the same side of the split.

Output (when :func:`main` is run):
``results/p14_split_reference_decision.json``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from audit_metrics import paired_user_audit_scope
from data_utils import RESULTS_DIR, load_kuairand_buckets
from p8_paired_audit import B_USERS, MIN_CATEGORIES, SEED
from p11_decision_audit import (
    deterministic_topk,
    max_t_simultaneous_contrast_intervals,
)


TOP_KS = (5, 10, 20)
ESTIMATORS = ("naive", "ce", "rw")
PRIMARY_CONTRAST_ESTIMATORS = ("ce", "rw")
SPLIT_SALT = "p14-split-reference-v1"
SPLIT_KEY_FIELDS = ("user_id", "video_id", "date", "duration_ms", "tab")
SPLIT_HASH_ALGORITHM = "sha256"
SPLIT_ASSIGNMENT_RULE = (
    "first SHA-256 digest byte modulo 2: 0=selection, 1=evaluation"
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _finite_outcome(frame: pd.DataFrame, outcome_col: str, label: str) -> None:
    _require_columns(frame, (outcome_col,), label)
    values = pd.to_numeric(frame[outcome_col], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}.{outcome_col} contains missing/non-finite values")


def _canonical_tokens(values: pd.Series) -> pd.Series:
    """Return stable string tokens for integer-like event-key columns."""
    if values.isna().any():
        raise ValueError("split key columns cannot contain missing values")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        array = numeric.to_numpy(float)
        if np.isfinite(array).all() and np.equal(array, np.floor(array)).all():
            return pd.Series(
                array.astype(np.int64).astype(str), index=values.index, dtype="string"
            )
    return values.astype("string")


def frozen_event_hash_assignments(
    random: pd.DataFrame,
    *,
    salt: str = SPLIT_SALT,
    key_fields: Sequence[str] = SPLIT_KEY_FIELDS,
) -> pd.DataFrame:
    """Return deterministic outcome-free event hashes and split labels.

    Assignment depends only on ``key_fields`` and ``salt``.  Identical event
    signatures therefore remain together, which avoids putting duplicate
    observable events on opposite sides when no request ID is available.
    """
    fields = tuple(str(value) for value in key_fields)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("key_fields must be a non-empty sequence of unique names")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")
    forbidden = {"watch_ratio", "play_time_ms", "is_click", "long_view"}
    if forbidden.intersection(fields):
        raise ValueError("split keys must not contain outcome fields")
    _require_columns(random, fields, "random")
    if random.empty:
        raise ValueError("random cannot be empty")

    key = pd.Series(salt, index=random.index, dtype="string")
    for field in fields:
        key = key.str.cat(
            pd.Series(f"\x1f{field}=", index=random.index, dtype="string")
        ).str.cat(_canonical_tokens(random[field]))
    digest = key.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    first_byte = digest.str.slice(0, 2).map(lambda value: int(value, 16))
    side = np.where((first_byte.to_numpy(int) % 2) == 0, "selection", "evaluation")
    return pd.DataFrame(
        {
            "event_sha256": digest.to_numpy(str),
            "split": side,
        },
        index=random.index,
    )


def _membership_sha256(event_hashes: pd.Series) -> str:
    canonical = "\n".join(sorted(event_hashes.astype(str).tolist()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _support_summary(frame: pd.DataFrame) -> dict[str, object]:
    category_rows = frame.groupby("cat1_name", sort=True).size()
    return {
        "rows": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "videos": int(frame["video_id"].nunique()),
        "categories": int(frame["cat1_name"].nunique()),
        "minimum_category_rows": int(category_rows.min()),
        "maximum_category_rows": int(category_rows.max()),
        "category_rows": {
            str(category): int(count) for category, count in category_rows.items()
        },
    }


def frozen_reference_split(
    random: pd.DataFrame,
    *,
    salt: str = SPLIT_SALT,
    key_fields: Sequence[str] = SPLIT_KEY_FIELDS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Split randomized rows and validate complete category support."""
    _require_columns(
        random,
        (*key_fields, "user_id", "video_id", "cat1_name"),
        "random",
    )
    if random[["user_id", "video_id", "cat1_name"]].isna().any().any():
        raise ValueError("random has missing scope identifiers")
    assignments = frozen_event_hash_assignments(
        random, salt=salt, key_fields=key_fields
    )
    selection_mask = assignments["split"].eq("selection")
    selection = random.loc[selection_mask].copy()
    evaluation = random.loc[~selection_mask].copy()
    if selection.empty or evaluation.empty:
        raise ValueError("frozen hash split produced an empty half")

    full_categories = {str(value) for value in random["cat1_name"].unique()}
    selection_categories = {
        str(value) for value in selection["cat1_name"].unique()
    }
    evaluation_categories = {
        str(value) for value in evaluation["cat1_name"].unique()
    }
    if selection_categories != full_categories:
        missing = sorted(full_categories - selection_categories)
        raise ValueError(f"selection half lost category support: {missing}")
    if evaluation_categories != full_categories:
        missing = sorted(full_categories - evaluation_categories)
        raise ValueError(f"evaluation half lost category support: {missing}")

    selection_hashes = assignments.loc[selection_mask, "event_sha256"]
    evaluation_hashes = assignments.loc[~selection_mask, "event_sha256"]
    # Identical observable event signatures are assigned as a group.
    signature_disjoint = set(selection_hashes).isdisjoint(set(evaluation_hashes))
    if not signature_disjoint:
        raise AssertionError("an event signature appears in both split halves")
    assignment_lines = sorted(
        f"{event_hash}:{side}"
        for event_hash, side in assignments[["event_sha256", "split"]].itertuples(
            index=False, name=None
        )
    )
    assignment_sha = hashlib.sha256(
        "\n".join(assignment_lines).encode("utf-8")
    ).hexdigest()
    metadata = {
        "name": "frozen outcome-free randomized-event hash split",
        "version": 1,
        "hash_algorithm": SPLIT_HASH_ALGORITHM,
        "salt": salt,
        "key_fields": list(key_fields),
        "outcome_fields_in_key": [],
        "assignment_rule": SPLIT_ASSIGNMENT_RULE,
        "identical_event_signatures_kept_together": True,
        "split_fixed_before_point_estimation_and_bootstrap": True,
        "selection_role": "define randomized-reference top-k only",
        "evaluation_role": "value the reference and candidate top-k sets only",
        "standard_role": "estimate all candidate dashboard rankings",
        "selection": {
            **_support_summary(selection),
            "membership_sha256": _membership_sha256(selection_hashes),
        },
        "evaluation": {
            **_support_summary(evaluation),
            "membership_sha256": _membership_sha256(evaluation_hashes),
        },
        "assignment_sha256": assignment_sha,
        "users_observed_in_both_halves": int(
            len(set(selection["user_id"]) & set(evaluation["user_id"]))
        ),
        "invariants": {
            "row_partition_complete": int(len(selection) + len(evaluation))
            == int(len(random)),
            "event_signatures_disjoint": bool(signature_disjoint),
            "selection_has_complete_category_support": True,
            "evaluation_has_complete_category_support": True,
            "outcome_free_assignment": True,
        },
    }
    return selection, evaluation, metadata


def _user_category_arrays(
    frame: pd.DataFrame,
    users: Sequence[object],
    categories: Sequence[object],
    outcome_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    _finite_outcome(frame, outcome_col, "bucket")
    user_index = {user: index for index, user in enumerate(users)}
    category_index = {
        category: index for index, category in enumerate(categories)
    }
    cells = (
        frame.groupby(["user_id", "cat1_name"], sort=False)[outcome_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    ui = cells["user_id"].map(user_index)
    ci = cells["cat1_name"].map(category_index)
    if ui.isna().any() or ci.isna().any():
        raise ValueError("frame contains users/categories outside fixed support")
    sums = np.zeros((len(users), len(categories)), dtype=float)
    counts = np.zeros_like(sums)
    np.add.at(sums, (ui.astype(int), ci.astype(int)), cells["sum"].to_numpy(float))
    np.add.at(
        counts,
        (ui.astype(int), ci.astype(int)),
        cells["count"].to_numpy(float),
    )
    return sums, counts


def split_sufficient_statistics(
    selection: pd.DataFrame,
    evaluation: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
) -> dict[str, object]:
    """Create aligned arrays for split-reference paired-user inference."""
    required = ("user_id", "cat1_name", outcome_col)
    for label, frame in (
        ("selection", selection),
        ("evaluation", evaluation),
        ("standard", standard),
    ):
        _require_columns(frame, required, label)
        if frame.empty:
            raise ValueError(f"{label} cannot be empty")
    random_users = set(selection["user_id"]) | set(evaluation["user_id"])
    standard_users = set(standard["user_id"])
    if random_users != standard_users:
        raise ValueError("random split union and standard must share the user set")
    selection_categories = set(selection["cat1_name"])
    evaluation_categories = set(evaluation["cat1_name"])
    standard_categories = set(standard["cat1_name"])
    if not (
        selection_categories == evaluation_categories == standard_categories
    ):
        raise ValueError("all three sources must have identical category support")
    users = sorted(standard_users, key=lambda value: str(value))
    categories = sorted(standard_categories, key=lambda value: str(value))
    selection_sums, selection_counts = _user_category_arrays(
        selection, users, categories, outcome_col
    )
    evaluation_sums, evaluation_counts = _user_category_arrays(
        evaluation, users, categories, outcome_col
    )
    standard_sums, standard_counts = _user_category_arrays(
        standard, users, categories, outcome_col
    )
    return {
        "users": users,
        "categories": categories,
        "selection_sums": selection_sums,
        "selection_counts": selection_counts,
        "evaluation_sums": evaluation_sums,
        "evaluation_counts": evaluation_counts,
        "standard_sums": standard_sums,
        "standard_counts": standard_counts,
        "outcome": outcome_col,
    }


def vectors_from_user_multiplicity(
    stats: Mapping[str, object], multiplicity: np.ndarray
) -> dict[str, pd.Series]:
    """Compute split references and standard-only ranking scores."""
    categories = list(stats["categories"])
    weights = np.asarray(multiplicity, dtype=float)
    selection_sums = np.asarray(stats["selection_sums"], dtype=float)
    selection_counts = np.asarray(stats["selection_counts"], dtype=float)
    evaluation_sums = np.asarray(stats["evaluation_sums"], dtype=float)
    evaluation_counts = np.asarray(stats["evaluation_counts"], dtype=float)
    standard_sums = np.asarray(stats["standard_sums"], dtype=float)
    standard_counts = np.asarray(stats["standard_counts"], dtype=float)
    if weights.ndim != 1 or len(weights) != selection_sums.shape[0]:
        raise ValueError("multiplicity must have one entry per paired user")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("multiplicity must be finite, non-negative, and nonzero")
    column_weights = weights[:, None]

    def aggregate(sums: np.ndarray, counts: np.ndarray, label: str):
        total_sum = np.sum(sums * column_weights, axis=0, dtype=float)
        total_count = np.sum(counts * column_weights, axis=0, dtype=float)
        if (total_count <= 0).any():
            raise ValueError(f"resample lost {label} category support")
        return total_sum, total_count

    sel_sum, sel_count = aggregate(
        selection_sums, selection_counts, "selection"
    )
    eval_sum, eval_count = aggregate(
        evaluation_sums, evaluation_counts, "evaluation"
    )
    std_sum, std_count = aggregate(standard_sums, standard_counts, "standard")
    selection_reference = sel_sum / sel_count
    evaluation_reference = eval_sum / eval_count
    naive = std_sum / std_count

    # These are standard-only centered ranking scores.  P8 adds a randomized
    # grand-mean scalar anchor to CE/RW for level reporting, but that scalar is
    # category-invariant and cannot affect a top-k ranking.
    user_total_count = standard_counts.sum(axis=1)
    if (user_total_count <= 0).any():
        raise ValueError("a paired user has no standard observations")
    user_mean = standard_sums.sum(axis=1) / user_total_count
    observed = standard_counts > 0
    cell_mean = np.divide(
        standard_sums,
        standard_counts,
        out=np.zeros_like(standard_sums),
        where=observed,
    )
    deviations = cell_mean - user_mean[:, None]
    weighted_observed = column_weights * observed
    cell_denominator = weighted_observed.sum(axis=0)
    if (cell_denominator <= 0).any():
        raise ValueError("resample lost standard user-category cell support")
    ce = np.sum(column_weights * deviations * observed, axis=0, dtype=float)
    ce /= cell_denominator
    residual_sums = standard_sums - user_mean[:, None] * standard_counts
    rw = np.sum(column_weights * residual_sums, axis=0, dtype=float) / std_count

    index = pd.Index([str(value) for value in categories], name="cat1_name")
    return {
        "selection_reference": pd.Series(selection_reference, index=index),
        "evaluation_reference": pd.Series(evaluation_reference, index=index),
        "naive": pd.Series(naive, index=index),
        "ce": pd.Series(ce, index=index),
        "rw": pd.Series(rw, index=index),
    }


def heldout_decision_metrics(
    estimate: pd.Series,
    selection_reference: pd.Series,
    evaluation_reference: pd.Series,
    *,
    k: int,
) -> dict[str, object]:
    """Rank on ``estimate`` and value only on the held-out reference."""
    categories = pd.Index(
        [str(value) for value in selection_reference.index], name="cat1_name"
    )
    selection_reference = pd.Series(
        selection_reference.to_numpy(float), index=categories
    )
    evaluation_reference = pd.Series(
        evaluation_reference.to_numpy(float),
        index=pd.Index([str(value) for value in evaluation_reference.index]),
    ).reindex(categories)
    estimate = pd.Series(
        estimate.to_numpy(float),
        index=pd.Index([str(value) for value in estimate.index]),
    ).reindex(categories)
    if estimate.isna().any() or evaluation_reference.isna().any():
        raise ValueError("decision vectors must share complete category support")
    reference_selected = deterministic_topk(selection_reference, k)
    candidate_selected = deterministic_topk(estimate, k)
    reference_value = float(evaluation_reference.loc[reference_selected].mean())
    candidate_value = float(evaluation_reference.loc[candidate_selected].mean())
    return {
        "k": int(k),
        "selected_categories": candidate_selected,
        "selection_reference_selected_categories": reference_selected,
        "candidate_heldout_evaluation_mean": candidate_value,
        "selection_reference_heldout_evaluation_mean": reference_value,
        "split_reference_regret": float(reference_value - candidate_value),
        "selection_topk_overlap": float(
            len(set(reference_selected) & set(candidate_selected)) / int(k)
        ),
    }


def _percentile_interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not len(values):
        raise ValueError("interval values must be non-empty and finite")
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def _supported_direction(interval: Sequence[float]) -> str:
    lower, upper = (float(value) for value in interval)
    if upper < 0.0:
        return "lower regret than naive"
    if lower > 0.0:
        return "higher regret than naive"
    return "not distinguishable from zero"


def split_reference_decision_audit(
    selection: pd.DataFrame,
    evaluation: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
    ks: Sequence[int] = TOP_KS,
    n_boot: int = B_USERS,
    seed: int = SEED,
) -> dict[str, object]:
    """Run held-out-reference decisions with synchronized user bootstrap."""
    requested_ks = tuple(int(value) for value in ks)
    if not requested_ks or len(set(requested_ks)) != len(requested_ks):
        raise ValueError("ks must be a non-empty sequence of unique integers")
    if n_boot < 2:
        raise ValueError("n_boot must be at least two")
    stats = split_sufficient_statistics(
        selection, evaluation, standard, outcome_col=outcome_col
    )
    categories = [str(value) for value in stats["categories"]]
    if any(not 1 <= k < len(categories) for k in requested_ks):
        raise ValueError("each k must satisfy 1 <= k < n_categories")
    n_users = len(stats["users"])
    point_vectors = vectors_from_user_multiplicity(
        stats, np.ones(n_users, dtype=float)
    )

    point: dict[str, object] = {}
    for k in requested_ks:
        reference_selected = deterministic_topk(
            point_vectors["selection_reference"], k
        )
        reference_selection_value = float(
            point_vectors["selection_reference"].loc[reference_selected].mean()
        )
        reference_evaluation_value = float(
            point_vectors["evaluation_reference"].loc[reference_selected].mean()
        )
        point[str(k)] = {
            "reference": {
                "selected_categories": reference_selected,
                "selection_half_topk_mean": reference_selection_value,
                "heldout_evaluation_mean": reference_evaluation_value,
                "split_reference_regret": 0.0,
                "selection_topk_overlap": 1.0,
            },
            **{
                estimator: heldout_decision_metrics(
                    point_vectors[estimator],
                    point_vectors["selection_reference"],
                    point_vectors["evaluation_reference"],
                    k=k,
                )
                for estimator in ESTIMATORS
            },
        }

    regret_draws = {
        str(k): {
            estimator: np.empty(n_boot, dtype=float)
            for estimator in ESTIMATORS
        }
        for k in requested_ks
    }
    value_draws = {
        str(k): {
            estimator: np.empty(n_boot, dtype=float)
            for estimator in ESTIMATORS
        }
        for k in requested_ks
    }
    overlap_draws = {
        str(k): {
            estimator: np.empty(n_boot, dtype=float)
            for estimator in ESTIMATORS
        }
        for k in requested_ks
    }
    reference_value_draws = {
        str(k): np.empty(n_boot, dtype=float) for k in requested_ks
    }

    rng = np.random.default_rng(seed)
    probabilities = np.repeat(1.0 / n_users, n_users)
    attempts = 0
    completed = 0
    max_attempts = max(n_boot * 20, n_boot + 100)
    while completed < n_boot and attempts < max_attempts:
        attempts += 1
        multiplicity = rng.multinomial(n_users, probabilities)
        try:
            vectors = vectors_from_user_multiplicity(stats, multiplicity)
        except ValueError as exc:
            if "resample lost" in str(exc):
                continue
            raise
        for k in requested_ks:
            key = str(k)
            reference_selected = deterministic_topk(
                vectors["selection_reference"], k
            )
            reference_value_draws[key][completed] = float(
                vectors["evaluation_reference"].loc[reference_selected].mean()
            )
            for estimator in ESTIMATORS:
                metrics = heldout_decision_metrics(
                    vectors[estimator],
                    vectors["selection_reference"],
                    vectors["evaluation_reference"],
                    k=k,
                )
                regret_draws[key][estimator][completed] = float(
                    metrics["split_reference_regret"]
                )
                value_draws[key][estimator][completed] = float(
                    metrics["candidate_heldout_evaluation_mean"]
                )
                overlap_draws[key][estimator][completed] = float(
                    metrics["selection_topk_overlap"]
                )
        completed += 1
    if completed != n_boot:
        raise RuntimeError(
            f"only {completed} complete bootstrap draws after {attempts} attempts"
        )

    nominal_ci95: dict[str, object] = {}
    paired_change: dict[str, object] = {}
    for k in requested_ks:
        key = str(k)
        nominal_ci95[key] = {
            "reference": {
                "heldout_evaluation_mean": _percentile_interval(
                    reference_value_draws[key]
                )
            },
            **{
                estimator: {
                    "split_reference_regret": _percentile_interval(
                        regret_draws[key][estimator]
                    ),
                    "candidate_heldout_evaluation_mean": _percentile_interval(
                        value_draws[key][estimator]
                    ),
                    "selection_topk_overlap": _percentile_interval(
                        overlap_draws[key][estimator]
                    ),
                }
                for estimator in ESTIMATORS
            },
        }
        paired_change[key] = {}
        for estimator in PRIMARY_CONTRAST_ESTIMATORS:
            regret_change = (
                regret_draws[key][estimator] - regret_draws[key]["naive"]
            )
            value_change = (
                value_draws[key][estimator] - value_draws[key]["naive"]
            )
            overlap_change = (
                overlap_draws[key][estimator] - overlap_draws[key]["naive"]
            )
            regret_interval = _percentile_interval(regret_change)
            paired_change[key][estimator] = {
                "definition": "estimator minus naive within the same bootstrap draw",
                "point": {
                    "split_reference_regret": float(
                        point[key][estimator]["split_reference_regret"]
                        - point[key]["naive"]["split_reference_regret"]
                    ),
                    "candidate_heldout_evaluation_mean": float(
                        point[key][estimator]["candidate_heldout_evaluation_mean"]
                        - point[key]["naive"][
                            "candidate_heldout_evaluation_mean"
                        ]
                    ),
                    "selection_topk_overlap": float(
                        point[key][estimator]["selection_topk_overlap"]
                        - point[key]["naive"]["selection_topk_overlap"]
                    ),
                },
                "nominal_percentile_ci95": {
                    "split_reference_regret": regret_interval,
                    "candidate_heldout_evaluation_mean": _percentile_interval(
                        value_change
                    ),
                    "selection_topk_overlap": _percentile_interval(
                        overlap_change
                    ),
                },
                "nominal_regret_supported_direction": _supported_direction(
                    regret_interval
                ),
                "interpretation": (
                    "negative regret contrast and positive held-out-value "
                    "contrast favor the estimator"
                ),
            }

    contrast_labels = [
        f"{estimator}_minus_naive_k{k}"
        for k in requested_ks
        for estimator in PRIMARY_CONTRAST_ESTIMATORS
    ]
    contrast_points = np.asarray(
        [
            paired_change[str(k)][estimator]["point"][
                "split_reference_regret"
            ]
            for k in requested_ks
            for estimator in PRIMARY_CONTRAST_ESTIMATORS
        ],
        dtype=float,
    )
    contrast_draws = np.column_stack(
        [
            regret_draws[str(k)][estimator] - regret_draws[str(k)]["naive"]
            for k in requested_ks
            for estimator in PRIMARY_CONTRAST_ESTIMATORS
        ]
    )
    familywise = max_t_simultaneous_contrast_intervals(
        contrast_points, contrast_draws, contrast_labels
    )
    familywise.update(
        {
            "family_definition": (
                "all CE-minus-naive and RW-minus-naive split-reference-regret "
                "contrasts across the pre-specified k=5,10,20 cutoffs"
            ),
            "contrast_definition": (
                "estimator split-reference regret minus naive split-reference "
                "regret; negative favors the estimator"
            ),
            "pointwise_intervals_elsewhere_are_nominal": True,
        }
    )
    for k in requested_ks:
        key = str(k)
        for estimator in PRIMARY_CONTRAST_ESTIMATORS:
            label = f"{estimator}_minus_naive_k{k}"
            familywise["per_contrast"][label].update(
                {
                    "k": int(k),
                    "estimator": estimator,
                    "contrast": f"{estimator} minus naive",
                    "nominal_pointwise_percentile_ci95": paired_change[key][
                        estimator
                    ]["nominal_percentile_ci95"]["split_reference_regret"],
                    "nominal_supported_direction": paired_change[key][estimator][
                        "nominal_regret_supported_direction"
                    ],
                }
            )

    return {
        "definitions": {
            "split_reference_regret": (
                "held-out evaluation-half mean of the selection-half reference "
                "top-k minus held-out evaluation-half mean of the candidate "
                "top-k; signed, lower is better, and negative is possible"
            ),
            "candidate_heldout_evaluation_mean": (
                "unweighted mean of evaluation-half randomized category means "
                "over the candidate top-k set; higher is better"
            ),
            "selection_topk_overlap": (
                "intersection with the selection-half randomized-reference "
                "top-k divided by k"
            ),
            "naive": "standard-serving category row means",
            "ce": (
                "standard-only observed-user-category-cell-equal demeaned "
                "ranking score; the omitted P8 random scalar anchor cannot "
                "change ranks"
            ),
            "rw": (
                "standard-only production-row-exposure-weighted demeaned "
                "ranking score; the omitted P8 random scalar anchor cannot "
                "change ranks"
            ),
            "tie_rule": (
                "sort by descending score, then ascending category string "
                "label for exact ties"
            ),
        },
        "design": {
            "unit": "paired user",
            "selection_and_evaluation_rows_disjoint": True,
            "selection_defines_reference_topk_only": True,
            "evaluation_values_sets_only": True,
            "candidate_rankings_use_standard_only": True,
            "same_user_multiplicity_in_selection_evaluation_and_standard": True,
            "split_membership_fixed_across_bootstrap_draws": True,
            "category_complete_draws": True,
            "categories_fixed_across_draws": True,
            "n_bootstrap": int(n_boot),
            "attempts": int(attempts),
            "seed": int(seed),
            "ks": list(requested_ks),
            "estimators": list(ESTIMATORS),
            "n_users": int(n_users),
            "n_categories": int(len(categories)),
            "familywise_regret_estimators": list(PRIMARY_CONTRAST_ESTIMATORS),
            "familywise_regret_contrast_count": int(
                len(PRIMARY_CONTRAST_ESTIMATORS) * len(requested_ks)
            ),
        },
        "point": point,
        "nominal_ci95": nominal_ci95,
        "nominal_ci95_metadata": {
            "method": "synchronized paired-user bootstrap percentile interval",
            "scope": "nominal pointwise",
            "multiplicity_adjusted": False,
            "note": (
                "use familywise_regret_contrasts_vs_naive for the pre-specified "
                "six-contrast multiplicity-adjusted decision claim family"
            ),
        },
        "paired_change_vs_naive": paired_change,
        "familywise_regret_contrasts_vs_naive": familywise,
        "point_category_vectors": {
            name: {
                str(category): float(value) for category, value in vector.items()
            }
            for name, vector in point_vectors.items()
        },
        "invariants": {
            "heldout_values_use_evaluation_reference_only": True,
            "reference_topk_uses_selection_reference_only": True,
            "candidate_topk_uses_standard_scores_only": True,
            "same_draws_for_all_familywise_contrasts": True,
            "regret_and_value_contrasts_are_exact_negatives": bool(
                all(
                    np.isclose(
                        paired_change[str(k)][estimator]["point"][
                            "split_reference_regret"
                        ],
                        -paired_change[str(k)][estimator]["point"][
                            "candidate_heldout_evaluation_mean"
                        ],
                        atol=1e-12,
                        rtol=0.0,
                    )
                    for k in requested_ks
                    for estimator in PRIMARY_CONTRAST_ESTIMATORS
                )
            ),
        },
    }


def _point_direction(value: float, *, tolerance: float = 1e-12) -> str:
    value = float(value)
    if value < -tolerance:
        return "lower regret than naive"
    if value > tolerance:
        return "higher regret than naive"
    return "tie with naive"


def compare_with_p11(
    split_audit: Mapping[str, object], p11: Mapping[str, object]
) -> dict[str, object]:
    """Compare the sign and inference conclusion with frozen P11 output."""
    requested_ks = [int(value) for value in split_audit["design"]["ks"]]
    per_contrast: dict[str, object] = {}
    point_maintained = 0
    familywise_maintained = 0
    for k in requested_ks:
        key = str(k)
        if key not in p11.get("paired_change_vs_naive", {}):
            raise ValueError(f"P11 result is missing k={k}")
        for estimator in PRIMARY_CONTRAST_ESTIMATORS:
            label = f"{estimator}_minus_naive_k{k}"
            p11_point = float(
                p11["paired_change_vs_naive"][key][estimator]["point"][
                    "reference_regret"
                ]
            )
            split_point = float(
                split_audit["paired_change_vs_naive"][key][estimator]["point"][
                    "split_reference_regret"
                ]
            )
            p11_direction = _point_direction(p11_point)
            split_direction = _point_direction(split_point)
            maintained = p11_direction == split_direction
            point_maintained += int(maintained)
            p11_fwer = p11["familywise_regret_contrasts_vs_naive"][
                "per_contrast"
            ][label]
            split_fwer = split_audit["familywise_regret_contrasts_vs_naive"][
                "per_contrast"
            ][label]
            p11_fwer_direction = str(p11_fwer["supported_direction"])
            split_fwer_direction = str(split_fwer["supported_direction"])
            fwer_maintained = p11_fwer_direction == split_fwer_direction
            familywise_maintained += int(fwer_maintained)
            per_contrast[label] = {
                "k": int(k),
                "estimator": estimator,
                "p11_point_regret_contrast": p11_point,
                "split_point_regret_contrast": split_point,
                "p11_point_direction": p11_direction,
                "split_point_direction": split_direction,
                "point_direction_maintained": maintained,
                "p11_nominal_pointwise_ci95": p11["paired_change_vs_naive"][key][
                    estimator
                ]["ci95"]["reference_regret"],
                "split_nominal_pointwise_ci95": split_audit[
                    "paired_change_vs_naive"
                ][key][estimator]["nominal_percentile_ci95"][
                    "split_reference_regret"
                ],
                "p11_familywise_ci95": p11_fwer["simultaneous_ci95"],
                "split_familywise_ci95": split_fwer["simultaneous_ci95"],
                "p11_familywise_supported_direction": p11_fwer_direction,
                "split_familywise_supported_direction": split_fwer_direction,
                "familywise_conclusion_maintained": fwer_maintained,
            }
    family_size = len(per_contrast)
    return {
        "comparison_target": "frozen P11 same-reference decision audit",
        "p11_analysis": p11.get("analysis"),
        "family_size": family_size,
        "per_contrast": per_contrast,
        "summary": {
            "point_direction_maintained_count": int(point_maintained),
            "point_direction_maintained_fraction": float(
                point_maintained / family_size
            ),
            "all_point_directions_maintained": point_maintained == family_size,
            "familywise_conclusion_maintained_count": int(familywise_maintained),
            "familywise_conclusion_maintained_fraction": float(
                familywise_maintained / family_size
            ),
            "all_familywise_conclusions_maintained": (
                familywise_maintained == family_size
            ),
        },
    }


def run_p14(
    *,
    root: str | Path | None = None,
    category_path: str | Path | None = None,
    p11_path: str | Path | None = None,
    n_boot: int = B_USERS,
    seed: int = SEED,
) -> dict[str, object]:
    """Load the P8/P11 scope, freeze the split, and run P14."""
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
    selection, evaluation, split_design = frozen_reference_split(paired_random)
    audit = split_reference_decision_audit(
        selection,
        evaluation,
        paired_standard,
        n_boot=n_boot,
        seed=seed,
    )
    resolved_p11 = Path(p11_path) if p11_path else RESULTS_DIR / "p11_decision_audit.json"
    with resolved_p11.open(encoding="utf-8") as handle:
        p11 = json.load(handle)
    p11_scope_id = p11.get("paired_scope", {}).get("primary_scope_id")
    scope_id = paired_scope.get("primary_scope_id")
    if p11_scope_id != scope_id:
        raise ValueError(
            "P11 and P14 paired-scope identities differ: "
            f"{p11_scope_id!r} != {scope_id!r}"
        )
    return {
        "analysis": "P14 leakage-free split-reference decision sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "base_scope": base_scope,
        "paired_scope": paired_scope,
        "split_design": split_design,
        **audit,
        "comparison_to_p11": {
            "paired_scope_id_matches": True,
            "p11_paired_scope_id": p11_scope_id,
            **compare_with_p11(audit, p11),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=B_USERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--category-path", type=Path, default=None)
    parser.add_argument("--p11-results", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "p14_split_reference_decision.json",
    )
    args = parser.parse_args()
    result = run_p14(
        root=args.data_root,
        category_path=args.category_path,
        p11_path=args.p11_results,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
