"""Shared estimands for the matched KuaiRand category audit.

The main comparison fixes the production cohort to users observed in at least
``min_categories`` categories.  Random-bucket category means remain the
external reference.  Centered-shape error uses the *random-bucket category
exposure shares* as one fixed set of centering weights for every estimator.
This avoids silently changing the estimand when an estimator changes the
category or row weights.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _validate_item_categories(
    frame: pd.DataFrame,
    *,
    label: str,
    item_col: str,
    category_col: str,
) -> pd.Series:
    """Return the unique item-to-category map, rejecting ambiguous labels."""
    if frame[[item_col, category_col]].isna().any().any():
        raise ValueError(f"{label} has missing item or category identifiers")
    counts = frame.groupby(item_col, dropna=False)[category_col].nunique(dropna=False)
    ambiguous = counts[counts != 1]
    if not ambiguous.empty:
        raise ValueError(
            f"{label} maps {len(ambiguous)} items to multiple categories"
        )
    return frame.drop_duplicates(item_col).set_index(item_col)[category_col]


def _support_counts(
    frame: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    category_col: str,
) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "users": int(frame[user_col].nunique()),
        "videos": int(frame[item_col].nunique()),
        "categories": int(frame[category_col].nunique()),
    }


def _identifier_sha256(values) -> str:
    """Hash a set of identifiers without depending on row order."""
    canonical = "\n".join(sorted({str(value) for value in values}))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def paired_user_audit_scope(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    min_categories: int = 5,
    *,
    user_col: str = "user_id",
    item_col: str = "video_id",
    category_col: str = "cat1_name",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Construct a fixed-point, paired-user audit scope for two traffic buckets.

    Both inputs are assumed to have already been restricted to the intended
    dates, tab, and base common-video pool.  Eligibility is first defined from
    production (``standard``) coverage.  Eligible users absent from ``random``
    are then removed from *both* buckets.  Common-video restriction and the
    production coverage check are repeated because removing a user can remove
    the only occurrence of a video from one bucket, which can in turn make a
    different production user ineligible.

    Returns ``(random_paired, standard_paired, metadata)``.  The final frames
    have identical user, video, and category sets; every retained production
    user covers at least ``min_categories`` categories.
    """
    if not isinstance(min_categories, (int, np.integer)) or min_categories < 1:
        raise ValueError("min_categories must be a positive integer")

    required = {user_col, item_col, category_col}
    _require_columns(random, required, "random")
    _require_columns(standard, required, "standard")
    if random.empty or standard.empty:
        raise ValueError("paired audit scope requires two non-empty buckets")
    if random[[user_col, item_col, category_col]].isna().any().any():
        raise ValueError("random has missing scope identifiers")
    if standard[[user_col, item_col, category_col]].isna().any().any():
        raise ValueError("standard has missing scope identifiers")

    random_item_categories = _validate_item_categories(
        random,
        label="random",
        item_col=item_col,
        category_col=category_col,
    )
    standard_item_categories = _validate_item_categories(
        standard,
        label="standard",
        item_col=item_col,
        category_col=category_col,
    )
    initially_common_items = random_item_categories.index.intersection(
        standard_item_categories.index
    )
    label_disagreement = (
        random_item_categories.reindex(initially_common_items)
        != standard_item_categories.reindex(initially_common_items)
    )
    if label_disagreement.any():
        raise ValueError(
            "random and standard assign different categories to shared videos"
        )

    random_paired = random.copy()
    standard_paired = standard.copy()
    initial = {
        "random": _support_counts(
            random_paired,
            user_col=user_col,
            item_col=item_col,
            category_col=category_col,
        ),
        "standard": _support_counts(
            standard_paired,
            user_col=user_col,
            item_col=item_col,
            category_col=category_col,
        ),
    }

    initial_coverage = standard_paired.groupby(user_col)[category_col].nunique()
    eligible_users = set(initial_coverage[initial_coverage >= min_categories].index)
    random_users = set(random_paired[user_col].unique())
    initially_paired_users = eligible_users & random_users
    if not initially_paired_users:
        raise ValueError("no eligible production users are observed in random")
    random_paired = random_paired[
        random_paired[user_col].isin(initially_paired_users)
    ].copy()
    standard_paired = standard_paired[
        standard_paired[user_col].isin(initially_paired_users)
    ].copy()

    iteration_log: list[dict[str, int]] = []
    previous_state: tuple[frozenset, frozenset, frozenset, frozenset] | None = None
    # Every non-stable iteration removes at least one user or video.  This is a
    # defensive cap, not an expected stopping rule.
    max_iterations = (
        len(initially_paired_users)
        + random_paired[item_col].nunique()
        + standard_paired[item_col].nunique()
        + 2
    )
    for iteration in range(1, max_iterations + 1):
        common_items = set(random_paired[item_col].unique()) & set(
            standard_paired[item_col].unique()
        )
        if not common_items:
            raise ValueError("paired audit scope lost all common videos")
        random_paired = random_paired[random_paired[item_col].isin(common_items)].copy()
        standard_paired = standard_paired[
            standard_paired[item_col].isin(common_items)
        ].copy()

        coverage = standard_paired.groupby(user_col)[category_col].nunique()
        covered_standard_users = set(coverage[coverage >= min_categories].index)
        common_users = (
            set(random_paired[user_col].unique())
            & set(standard_paired[user_col].unique())
            & covered_standard_users
        )
        if not common_users:
            raise ValueError(
                "no paired users retain the requested production category coverage"
            )
        random_paired = random_paired[random_paired[user_col].isin(common_users)].copy()
        standard_paired = standard_paired[
            standard_paired[user_col].isin(common_users)
        ].copy()

        random_user_set = frozenset(random_paired[user_col].unique())
        standard_user_set = frozenset(standard_paired[user_col].unique())
        random_item_set = frozenset(random_paired[item_col].unique())
        standard_item_set = frozenset(standard_paired[item_col].unique())
        state = (
            random_user_set,
            standard_user_set,
            random_item_set,
            standard_item_set,
        )
        iteration_log.append(
            {
                "iteration": iteration,
                "paired_users": len(random_user_set & standard_user_set),
                "common_videos": len(random_item_set & standard_item_set),
                "random_rows": int(len(random_paired)),
                "standard_rows": int(len(standard_paired)),
            }
        )
        if state == previous_state:
            break
        previous_state = state
    else:  # pragma: no cover - the finite monotone construction must converge.
        raise RuntimeError("paired audit scope did not converge")

    final_random_users = set(random_paired[user_col].unique())
    final_standard_users = set(standard_paired[user_col].unique())
    final_random_items = set(random_paired[item_col].unique())
    final_standard_items = set(standard_paired[item_col].unique())
    final_random_categories = set(random_paired[category_col].unique())
    final_standard_categories = set(standard_paired[category_col].unique())
    final_coverage = standard_paired.groupby(user_col)[category_col].nunique()

    # These are estimand-defining invariants.  Keep explicit checks so they are
    # enforced even when Python assertions are disabled.
    if final_random_users != final_standard_users:
        raise AssertionError("final random and standard user sets differ")
    if final_random_items != final_standard_items:
        raise AssertionError("final random and standard video sets differ")
    if final_random_categories != final_standard_categories:
        raise AssertionError("final random and standard category sets differ")
    if (final_coverage < min_categories).any():
        raise AssertionError("a final standard user is below min_categories")

    final_counts_random = _support_counts(
        random_paired,
        user_col=user_col,
        item_col=item_col,
        category_col=category_col,
    )
    final_counts_standard = _support_counts(
        standard_paired,
        user_col=user_col,
        item_col=item_col,
        category_col=category_col,
    )
    scope_identity = {
        "min_categories": int(min_categories),
        "user_id_sha256": _identifier_sha256(final_random_users),
        "video_id_sha256": _identifier_sha256(final_random_items),
        "category_name_sha256": _identifier_sha256(final_random_categories),
        "random": final_counts_random,
        "standard": final_counts_standard,
    }
    primary_scope_id = hashlib.sha256(
        json.dumps(
            scope_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    metadata = {
        "definition": (
            "standard users with at least min_categories, intersected with random "
            "user IDs; common-video and standard-coverage restrictions iterated "
            "to a fixed point"
        ),
        "min_categories": int(min_categories),
        "initial": initial,
        "standard_eligible_users_before_random_intersection": len(eligible_users),
        "eligible_users_missing_from_random": len(eligible_users - random_users),
        "users_after_initial_random_intersection": len(initially_paired_users),
        "iterations_to_fixed_point": len(iteration_log),
        "iteration_log": iteration_log,
        "primary_scope_id": primary_scope_id,
        "final": {
            "random": final_counts_random,
            "standard": final_counts_standard,
            "category_names": sorted(final_random_categories, key=str),
            "minimum_standard_user_category_coverage": int(final_coverage.min()),
            "user_id_sha256": scope_identity["user_id_sha256"],
            "video_id_sha256": scope_identity["video_id_sha256"],
            "category_name_sha256": scope_identity["category_name_sha256"],
        },
        "invariants": {
            "same_user_set": True,
            "same_video_set": True,
            "same_category_set": True,
            "standard_coverage_satisfied": True,
        },
    }
    return random_paired, standard_paired, metadata


def symmetric_category_gap_decomposition(
    standard: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
    item_col: str = "video_id",
    category_col: str = "cat1_name",
    atol: float = 1e-10,
) -> pd.DataFrame:
    """Exactly decompose each category gap using symmetric video weights.

    For category ``c``, let ``p_s(v)`` and ``p_r(v)`` be row-exposure shares
    within the standard and reference buckets, and let ``mu_s(v)`` and
    ``mu_r(v)`` be their video-level outcome means.  The symmetric identity is

    ``standard - reference = response_context + item_mix``, where

    ``response_context = sum_v .5 * (p_s + p_r) * (mu_s - mu_r)`` and
    ``item_mix = sum_v .5 * (mu_s + mu_r) * (p_s - p_r)``.

    Thus the first component holds the two buckets' video weights symmetrically
    fixed while changing within-video response/context, and the second holds
    video responses symmetrically fixed while changing the within-category item
    mix.  Both buckets must have identical category-specific video support.
    """
    required = {outcome_col, item_col, category_col}
    _require_columns(standard, required, "standard")
    _require_columns(reference, required, "reference")
    if standard.empty or reference.empty:
        raise ValueError("gap decomposition requires two non-empty buckets")
    standard_map = _validate_item_categories(
        standard,
        label="standard",
        item_col=item_col,
        category_col=category_col,
    )
    reference_map = _validate_item_categories(
        reference,
        label="reference",
        item_col=item_col,
        category_col=category_col,
    )
    if set(standard_map.index) != set(reference_map.index):
        raise ValueError("standard and reference video sets differ")
    aligned_reference_map = reference_map.reindex(standard_map.index)
    if (standard_map != aligned_reference_map).any():
        raise ValueError("standard and reference category labels differ by video")

    standard_outcome = pd.to_numeric(standard[outcome_col], errors="coerce")
    reference_outcome = pd.to_numeric(reference[outcome_col], errors="coerce")
    if not np.isfinite(standard_outcome.to_numpy(dtype=float)).all():
        raise ValueError("standard outcome contains missing or non-finite values")
    if not np.isfinite(reference_outcome.to_numpy(dtype=float)).all():
        raise ValueError("reference outcome contains missing or non-finite values")

    standard_cells = (
        standard.assign(**{outcome_col: standard_outcome})
        .groupby([category_col, item_col], sort=True)[outcome_col]
        .agg(["mean", "size"])
        .rename(columns={"mean": "mu_standard", "size": "n_standard"})
    )
    reference_cells = (
        reference.assign(**{outcome_col: reference_outcome})
        .groupby([category_col, item_col], sort=True)[outcome_col]
        .agg(["mean", "size"])
        .rename(columns={"mean": "mu_reference", "size": "n_reference"})
    )
    if not standard_cells.index.equals(reference_cells.index):
        raise ValueError("category-specific video supports differ")
    cells = standard_cells.join(reference_cells, how="inner")
    cells["p_standard"] = cells["n_standard"] / cells.groupby(
        level=category_col
    )["n_standard"].transform("sum")
    cells["p_reference"] = cells["n_reference"] / cells.groupby(
        level=category_col
    )["n_reference"].transform("sum")
    cells["standard_contribution"] = cells["p_standard"] * cells["mu_standard"]
    cells["reference_contribution"] = (
        cells["p_reference"] * cells["mu_reference"]
    )
    cells["response_context_contribution"] = 0.5 * (
        cells["p_standard"] + cells["p_reference"]
    ) * (cells["mu_standard"] - cells["mu_reference"])
    cells["item_mix_contribution"] = 0.5 * (
        cells["mu_standard"] + cells["mu_reference"]
    ) * (cells["p_standard"] - cells["p_reference"])

    grouped = cells.groupby(level=category_col, sort=True)
    result = pd.DataFrame(
        {
            "standard": grouped["standard_contribution"].sum(),
            "reference": grouped["reference_contribution"].sum(),
            "response_context": grouped["response_context_contribution"].sum(),
            "item_mix": grouped["item_mix_contribution"].sum(),
            "n_videos": grouped.size().astype(int),
        }
    )
    result["gap"] = result["standard"] - result["reference"]
    result["reconstructed_gap"] = result["response_context"] + result["item_mix"]
    result["reconstruction_error"] = result["gap"] - result["reconstructed_gap"]
    if not np.allclose(
        result["gap"].to_numpy(),
        result["reconstructed_gap"].to_numpy(),
        rtol=0.0,
        atol=float(atol),
    ):
        maximum_error = float(result["reconstruction_error"].abs().max())
        raise AssertionError(
            f"symmetric gap decomposition failed reconstruction: {maximum_error}"
        )
    return result[
        [
            "standard",
            "reference",
            "gap",
            "response_context",
            "item_mix",
            "reconstructed_gap",
            "reconstruction_error",
            "n_videos",
        ]
    ]


def eligible_production_rows(
    standard: pd.DataFrame, min_categories: int = 5
) -> tuple[pd.DataFrame, pd.Series]:
    """Return rows from production users covering at least ``min_categories``."""
    coverage = standard.groupby("user_id")["cat1_name"].nunique()
    keep = coverage[coverage >= min_categories].index
    return standard[standard["user_id"].isin(keep)].copy(), coverage


def randomized_reference(
    random: pd.DataFrame, categories: list[str] | None = None
) -> tuple[pd.Series, pd.Series, float]:
    """Return category means, normalized exposure shares, and grand mean."""
    means = random.groupby("cat1_name")["watch_ratio"].mean()
    shares = random["cat1_name"].value_counts(normalize=True)
    if categories is None:
        categories = sorted(means.index.tolist())
    means = means.reindex(categories)
    shares = shares.reindex(categories)
    if means.isna().any() or shares.isna().any():
        raise ValueError("randomized reference is missing requested categories")
    shares = shares / shares.sum()
    grand = float((means * shares).sum())
    return means, shares, grand


def user_demeaned_category_estimate(
    standard_eligible: pd.DataFrame,
    categories: list[str],
    level_anchor: float,
) -> tuple[pd.Series, pd.Series]:
    """Average user-cell deviations by category and add an external level anchor.

    The user mean is exposure weighted within each user.  Each observed
    user-category cell then receives equal weight within category.  This is a
    descriptive aggregation rule, not the same row-weighted target used by HT,
    Hajek, or AIPW.
    """
    cells = (
        standard_eligible.groupby(["user_id", "cat1_name"])["watch_ratio"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    user_sum = cells.groupby("user_id")["sum"].transform("sum")
    user_count = cells.groupby("user_id")["count"].transform("sum")
    cells["demeaned"] = cells["mean"] - user_sum / user_count
    centered = cells.groupby("cat1_name")["demeaned"].mean().reindex(categories)
    if centered.isna().any():
        raise ValueError("eligible production cohort is missing a category")
    return centered + float(level_anchor), centered


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def evaluate_category_estimate(
    estimate: pd.Series,
    reference: pd.Series,
    center_weights: pd.Series,
    *,
    topk: int = 10,
) -> dict:
    """Evaluate one category vector under a fixed category set and weighting."""
    categories = list(reference.index)
    estimate = estimate.reindex(categories)
    center_weights = center_weights.reindex(categories)
    if estimate.isna().any() or reference.isna().any() or center_weights.isna().any():
        raise ValueError("estimate, reference, and weights must share complete support")
    weights = center_weights / center_weights.sum()
    estimate_center = float((estimate * weights).sum())
    reference_center = float((reference * weights).sum())
    centered_error = (
        (estimate - estimate_center) - (reference - reference_center)
    ).abs()
    k = min(int(topk), len(categories))
    top_reference = set(reference.nlargest(k).index)
    top_estimate = set(estimate.nlargest(k).index)
    absolute_error = (estimate - reference).abs()
    return {
        "mae": float(absolute_error.mean()),
        "centered_shape_mae": float(centered_error.mean()),
        "max_abs_error": float(absolute_error.max()),
        "rank_corr": spearman(estimate, reference),
        f"top{k}_overlap": len(top_reference & top_estimate) / k,
        "estimate_center_random_share_weighted": estimate_center,
        "reference_center_random_share_weighted": reference_center,
    }


CENTER_DEFINITION = (
    "subtract random-category-exposure-share-weighted category mean from both "
    "the estimate and randomized reference; report unweighted MAE across categories"
)
