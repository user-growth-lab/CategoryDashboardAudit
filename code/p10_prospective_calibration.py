"""P10: leakage-resistant prospective calibration of category dashboards.

The primary protocol uses standard-serving logs from 2022-04-08--04-21 to
learn production-user eligibility and user baselines, random-insertion plus
standard-serving logs from 2022-04-22--04-30 to fit calibration constants, and
2022-05-01--05-08 as a frozen test.  Test-period random outcomes are read only
to construct the evaluation reference (and an explicitly non-deployable oracle
anchor); they never enter a deployable estimator.

The module also evaluates an earlier rolling origin (calibration through
04-26, test beginning 04-27).  Its test window overlaps the primary protocol,
so it is a stability sensitivity rather than an independent replication.

Outputs (when :func:`main` is run):
``results/p10_prospective_calibration.json`` and
``results/fig10_prospective_calibration.png``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from audit_metrics import evaluate_category_estimate
from data_utils import RESULTS_DIR, load_kuairand_buckets


SEED = 47
B_USERS = 1000
MIN_CATEGORIES = 5
TOPK = 10
HISTORY_DATES = (20220408, 20220421)
PRIMARY_SPLIT = ((20220422, 20220430), (20220501, 20220508))
SECONDARY_SPLIT = ((20220422, 20220426), (20220427, 20220508))

DEPLOYABLE_METHODS = (
    "naive",
    "grand_mean_shift",
    "category_vector_shift",
    "prospective_ce_cell_equal",
    "prospective_rw_row_weighted",
)
ORACLE_METHOD = "oracle_contemporaneous_ce"
METHODS = (*DEPLOYABLE_METHODS, ORACLE_METHOD)
PLANNED_CONTRASTS = {
    "category_vector_shift_minus_grand_mean_shift": (
        "category_vector_shift",
        "grand_mean_shift",
    ),
    "prospective_ce_cell_equal_minus_grand_mean_shift": (
        "prospective_ce_cell_equal",
        "grand_mean_shift",
    ),
    "prospective_rw_row_weighted_minus_grand_mean_shift": (
        "prospective_rw_row_weighted",
        "grand_mean_shift",
    ),
    "category_vector_shift_minus_oracle_contemporaneous_ce": (
        "category_vector_shift",
        "oracle_contemporaneous_ce",
    ),
    "prospective_rw_row_weighted_minus_prospective_ce_cell_equal": (
        "prospective_rw_row_weighted",
        "prospective_ce_cell_equal",
    ),
}
METRIC_NAMES = (
    "absolute_mae",
    "centered_mae",
    "max_abs_error",
    "rank_corr",
    "top10_overlap",
)


def _require(frame: pd.DataFrame, label: str) -> None:
    required = {"user_id", "video_id", "cat1_name", "watch_ratio"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label} cannot be empty")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{label} contains missing required values")
    outcome = pd.to_numeric(frame["watch_ratio"], errors="coerce").to_numpy(float)
    if not np.isfinite(outcome).all():
        raise ValueError(f"{label}.watch_ratio contains non-finite values")
    mapping = frame.groupby("video_id")["cat1_name"].nunique()
    if (mapping != 1).any():
        raise ValueError(f"{label} maps a video to multiple categories")


def _support(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "videos": int(frame["video_id"].nunique()),
        "categories": int(frame["cat1_name"].nunique()),
    }


def _hash_ids(values) -> str:
    canonical = "\n".join(sorted({str(value) for value in values}))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def strict_prospective_scope(
    history_standard: pd.DataFrame,
    calibration_random: pd.DataFrame,
    calibration_standard: pd.DataFrame,
    test_random: pd.DataFrame,
    test_standard: pd.DataFrame,
    *,
    min_categories: int = MIN_CATEGORIES,
) -> dict[str, object]:
    """Freeze the fitting cohort/support before opening test outcomes.

    History alone determines category-coverage eligibility.  A monotone fixed
    point then retains users and videos observed in all three *fitting*
    strata: history standard, calibration random, and calibration standard.
    No test row participates in this fit-time fixed point.

    At evaluation time we retain frozen users present in both test logs and
    rows on the frozen video support.  This presence check defines the paired
    evaluation sample but never refits eligibility, baselines, or calibration.
    """
    if not isinstance(min_categories, (int, np.integer)) or min_categories < 1:
        raise ValueError("min_categories must be a positive integer")
    frames = {
        "history_standard": history_standard.copy(),
        "calibration_random": calibration_random.copy(),
        "calibration_standard": calibration_standard.copy(),
        "test_random": test_random.copy(),
        "test_standard": test_standard.copy(),
    }
    for label, frame in frames.items():
        _require(frame, label)

    initial = {label: _support(frame) for label, frame in frames.items()}
    history_coverage = frames["history_standard"].groupby("user_id")[
        "cat1_name"
    ].nunique()
    eligible = set(history_coverage[history_coverage >= min_categories].index)
    history_eligible_users = len(eligible)
    fitting_names = (
        "history_standard",
        "calibration_random",
        "calibration_standard",
    )
    fitting_users = eligible.copy()
    for name in fitting_names:
        fitting_users &= set(frames[name]["user_id"].unique())
    if not fitting_users:
        raise ValueError("no history-eligible user occurs in all fitting strata")
    for name in fitting_names:
        frames[name] = frames[name][
            frames[name]["user_id"].isin(fitting_users)
        ].copy()

    iteration_log: list[dict[str, int]] = []
    previous: tuple[frozenset, frozenset] | None = None
    max_iterations = sum(frame["video_id"].nunique() for frame in frames.values()) + len(
        fitting_users
    ) + 2
    for iteration in range(1, max_iterations + 1):
        videos = set(frames[fitting_names[0]]["video_id"].unique())
        users = set(frames[fitting_names[0]]["user_id"].unique())
        for name in fitting_names[1:]:
            videos &= set(frames[name]["video_id"].unique())
            users &= set(frames[name]["user_id"].unique())
        if not videos or not users:
            raise ValueError("fit-time fixed point lost all users or videos")
        for name in fitting_names:
            frame = frames[name]
            frames[name] = frame[
                frame["video_id"].isin(videos) & frame["user_id"].isin(users)
            ].copy()

        coverage = frames["history_standard"].groupby("user_id")[
            "cat1_name"
        ].nunique()
        users &= set(coverage[coverage >= min_categories].index)
        for name in fitting_names:
            frames[name] = frames[name][frames[name]["user_id"].isin(users)].copy()

        state = (frozenset(users), frozenset(videos))
        iteration_log.append(
            {
                "iteration": iteration,
                "fitting_users": len(users),
                "frozen_videos": len(videos),
            }
        )
        if state == previous:
            break
        previous = state
    else:  # pragma: no cover
        raise RuntimeError("fit-time fixed point did not converge")

    frozen_users = set(frames["history_standard"]["user_id"].unique())
    frozen_videos = set(frames["history_standard"]["video_id"].unique())
    for name in fitting_names[1:]:
        if set(frames[name]["user_id"].unique()) != frozen_users:
            raise AssertionError("fitting user sets differ after fixed point")
        if set(frames[name]["video_id"].unique()) != frozen_videos:
            raise AssertionError("fitting video sets differ after fixed point")
    frozen_categories = set(frames["history_standard"]["cat1_name"].unique())
    for name in fitting_names[1:]:
        if set(frames[name]["cat1_name"].unique()) != frozen_categories:
            raise AssertionError("fitting category sets differ after fixed point")

    # Evaluation selection is performed only after the fit-time cohort/support
    # has been frozen.  Crucially, fitting frames are not reduced to this set.
    evaluation_users = (
        frozen_users
        & set(frames["test_random"]["user_id"].unique())
        & set(frames["test_standard"]["user_id"].unique())
    )
    if not evaluation_users:
        raise ValueError("no frozen user occurs in both test logs")
    for name in ("test_random", "test_standard"):
        frame = frames[name]
        frames[name] = frame[
            frame["user_id"].isin(evaluation_users)
            & frame["video_id"].isin(frozen_videos)
            & frame["cat1_name"].isin(frozen_categories)
        ].copy()
        if frames[name].empty:
            raise ValueError(f"{name} is empty on frozen support")
    # A user can lose their last row when the already-frozen video support is
    # applied.  Re-intersect test presence once so both evaluation strata use
    # exactly the same paired users.  Fitting frames remain untouched.
    evaluation_users = set(frames["test_random"]["user_id"].unique()) & set(
        frames["test_standard"]["user_id"].unique()
    )
    if not evaluation_users:
        raise ValueError("frozen video support removes every paired test user")
    for name in ("test_random", "test_standard"):
        frames[name] = frames[name][
            frames[name]["user_id"].isin(evaluation_users)
        ].copy()
    evaluation_categories = set(frames["test_random"]["cat1_name"].unique()) & set(
        frames["test_standard"]["cat1_name"].unique()
    )
    if not evaluation_categories:
        raise ValueError("test logs have no common category on frozen support")
    if evaluation_categories != frozen_categories:
        missing = sorted(frozen_categories - evaluation_categories, key=str)
        raise ValueError(
            "test logs lack frozen categories; prospective category-complete "
            f"evaluation is unavailable: {missing}"
        )
    for name in ("test_random", "test_standard"):
        frames[name] = frames[name][
            frames[name]["cat1_name"].isin(evaluation_categories)
        ].copy()

    frames["metadata"] = {
        "definition": (
            "history-only coverage eligibility; common-user/video fit-time "
            "fixed point across history standard and calibration random/standard; "
            "then frozen evaluation on users present in both test logs"
        ),
        "min_history_categories": int(min_categories),
        "history_eligible_users": int(history_eligible_users),
        "history_eligible_fraction": float(
            history_eligible_users / initial["history_standard"]["users"]
        ),
        "initial": initial,
        "fit_fixed_point_iterations": len(iteration_log),
        "iteration_log": iteration_log,
        "frozen": {
            "users": len(frozen_users),
            "videos": len(frozen_videos),
            "categories": len(frozen_categories),
            "over_history_eligible_users": float(
                len(frozen_users) / history_eligible_users
            ),
            "user_id_sha256": _hash_ids(frozen_users),
            "video_id_sha256": _hash_ids(frozen_videos),
        },
        "fitting_support": {
            name: _support(frames[name]) for name in fitting_names
        },
        "evaluation": {
            "paired_users": len(evaluation_users),
            "categories": len(evaluation_categories),
            "frozen_user_retention": float(len(evaluation_users) / len(frozen_users)),
            "history_eligible_user_retention": float(
                len(evaluation_users) / history_eligible_users
            ),
            "history_baseline_coverage": 1.0,
            "test_random": _support(frames["test_random"]),
            "test_standard": _support(frames["test_standard"]),
        },
        "frozen_before_test_evaluation": True,
        "test_outcomes_used_for_fitting": False,
        "test_random_role": "evaluation reference only",
    }
    return frames


def fit_frozen_calibration(
    history_standard: pd.DataFrame,
    calibration_random: pd.DataFrame,
    calibration_standard: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
) -> dict[str, object]:
    """Fit baselines and shifts without accessing either test log."""
    for label, frame in (
        ("history_standard", history_standard),
        ("calibration_random", calibration_random),
        ("calibration_standard", calibration_standard),
    ):
        _require(frame.rename(columns={outcome_col: "watch_ratio"}) if outcome_col != "watch_ratio" else frame, label)
    user_sets = [
        set(frame["user_id"].unique())
        for frame in (history_standard, calibration_random, calibration_standard)
    ]
    if user_sets[0] != user_sets[1] or user_sets[0] != user_sets[2]:
        raise ValueError("identical fitting user sets are required")
    video_sets = [
        set(frame["video_id"].unique())
        for frame in (history_standard, calibration_random, calibration_standard)
    ]
    if video_sets[0] != video_sets[1] or video_sets[0] != video_sets[2]:
        raise ValueError("identical fitting video sets are required")
    categories = sorted(
        set(history_standard["cat1_name"].unique()), key=lambda value: str(value)
    )
    if any(set(frame["cat1_name"].unique()) != set(categories) for frame in (calibration_random, calibration_standard)):
        raise ValueError("identical fitting category sets are required")

    history_baseline = history_standard.groupby("user_id")[outcome_col].mean()
    calibration_random_mean = float(calibration_random[outcome_col].mean())
    calibration_standard_mean = float(calibration_standard[outcome_col].mean())
    random_category = (
        calibration_random.groupby("cat1_name")[outcome_col].mean().reindex(categories)
    )
    standard_category = (
        calibration_standard.groupby("cat1_name")[outcome_col].mean().reindex(categories)
    )
    if random_category.isna().any() or standard_category.isna().any():
        raise ValueError("calibration category support is incomplete")
    return {
        "outcome": outcome_col,
        "users": sorted(user_sets[0], key=lambda value: str(value)),
        "videos": sorted(video_sets[0], key=lambda value: str(value)),
        "categories": categories,
        "history_user_baseline": history_baseline,
        "calibration_random_anchor": calibration_random_mean,
        "calibration_scalar_shift": calibration_random_mean
        - calibration_standard_mean,
        "calibration_category_shift": random_category - standard_category,
        "fit_uses_test_rows": False,
    }


def _category_metrics(
    estimate: pd.Series,
    reference: pd.Series,
    shares: pd.Series,
    *,
    topk: int,
) -> dict[str, float | int]:
    raw = evaluate_category_estimate(estimate, reference, shares, topk=topk)
    categories = list(reference.index)
    weights = shares.reindex(categories).astype(float)
    weights /= weights.sum()
    estimate = estimate.reindex(categories)
    reference = reference.reindex(categories)
    estimate_center = float((estimate * weights).sum())
    reference_center = float((reference * weights).sum())
    effective_k = min(topk, len(categories))
    return {
        "absolute_mae": float(raw["mae"]),
        "centered_mae": float(raw["centered_shape_mae"]),
        "random_share_weighted_absolute_mae": float(
            ((estimate - reference).abs() * weights).sum()
        ),
        "random_share_weighted_centered_mae": float(
            (
                (
                    (estimate - estimate_center)
                    - (reference - reference_center)
                ).abs()
                * weights
            ).sum()
        ),
        "max_abs_error": float(raw["max_abs_error"]),
        "rank_corr": float(raw["rank_corr"]),
        "top10_overlap": float(raw[f"top{effective_k}_overlap"]),
        "topk_effective": int(effective_k),
    }


def _residual_vectors(
    test_standard: pd.DataFrame,
    user_baseline: pd.Series,
    categories: list,
    anchor: float,
    outcome_col: str,
) -> tuple[pd.Series, pd.Series]:
    cells = (
        test_standard.groupby(["user_id", "cat1_name"])[outcome_col]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    baseline = cells["user_id"].map(user_baseline)
    if baseline.isna().any():
        raise ValueError("test standard contains a user without frozen history baseline")
    cells["cell_residual"] = cells["mean"] - baseline
    ce = cells.groupby("cat1_name")["cell_residual"].mean().reindex(categories)
    cells["row_residual_sum"] = cells["sum"] - baseline * cells["count"]
    grouped = cells.groupby("cat1_name")[["row_residual_sum", "count"]].sum()
    rw = (
        grouped["row_residual_sum"] / grouped["count"]
    ).reindex(categories)
    if ce.isna().any() or rw.isna().any():
        raise ValueError("test standard category support is incomplete")
    return ce + anchor, rw + anchor


def evaluate_frozen_calibration(
    model: Mapping[str, object],
    test_random: pd.DataFrame,
    test_standard: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
    topk: int = TOPK,
) -> dict[str, object]:
    """Evaluate frozen methods; test random is reference-only except oracle."""
    for label, frame in (("test_random", test_random), ("test_standard", test_standard)):
        _require(frame.rename(columns={outcome_col: "watch_ratio"}) if outcome_col != "watch_ratio" else frame, label)
    categories = [
        category
        for category in model["categories"]
        if category in set(test_random["cat1_name"].unique())
        and category in set(test_standard["cat1_name"].unique())
    ]
    if not categories:
        raise ValueError("test logs have no evaluable frozen category")
    random = test_random[test_random["cat1_name"].isin(categories)]
    standard = test_standard[test_standard["cat1_name"].isin(categories)]
    reference = random.groupby("cat1_name")[outcome_col].mean().reindex(categories)
    counts = random["cat1_name"].value_counts().reindex(categories)
    shares = counts / counts.sum()
    naive = standard.groupby("cat1_name")[outcome_col].mean().reindex(categories)
    grand_shift = naive + float(model["calibration_scalar_shift"])
    category_shift = naive + model["calibration_category_shift"].reindex(categories)
    prospective_ce, prospective_rw = _residual_vectors(
        standard,
        model["history_user_baseline"],
        categories,
        float(model["calibration_random_anchor"]),
        outcome_col,
    )

    current_baseline = standard.groupby("user_id")[outcome_col].mean()
    oracle_ce, _ = _residual_vectors(
        standard,
        current_baseline,
        categories,
        float(random[outcome_col].mean()),
        outcome_col,
    )
    vectors = {
        "reference": reference,
        "reference_shares": shares,
        "naive": naive,
        "grand_mean_shift": grand_shift,
        "category_vector_shift": category_shift,
        "prospective_ce_cell_equal": prospective_ce,
        "prospective_rw_row_weighted": prospective_rw,
        "oracle_contemporaneous_ce": oracle_ce,
    }
    return {
        "methods": list(METHODS),
        "method_roles": {
            "naive": "deployable; no random calibration",
            "grand_mean_shift": "deployable after calibration window",
            "category_vector_shift": "deployable after calibration window",
            "prospective_ce_cell_equal": (
                "deployable after history/calibration; equal observed user-category cells"
            ),
            "prospective_rw_row_weighted": (
                "deployable after history/calibration; standard exposure-row target"
            ),
            "oracle_contemporaneous_ce": (
                "non-deployable upper-bound diagnostic; uses test random grand mean"
            ),
        },
        "test_random_used_to_construct_deployable_vectors": False,
        "test_random_used_for_evaluation_metrics": True,
        "point": {
            name: _category_metrics(
                vectors[name], reference, shares, topk=topk
            )
            for name in METHODS
        },
        "category_vectors": {
            name: {str(key): float(value) for key, value in vector.items()}
            for name, vector in vectors.items()
        },
    }


def _user_category_arrays(
    frame: pd.DataFrame,
    users: list,
    categories: list,
    outcome_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    user_index = {user: index for index, user in enumerate(users)}
    category_index = {category: index for index, category in enumerate(categories)}
    cells = (
        frame.groupby(["user_id", "cat1_name"])[outcome_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    sums = np.zeros((len(users), len(categories)), dtype=float)
    counts = np.zeros_like(sums)
    ui = cells["user_id"].map(user_index)
    ci = cells["cat1_name"].map(category_index)
    keep = ui.notna() & ci.notna()
    np.add.at(
        sums,
        (ui[keep].astype(int), ci[keep].astype(int)),
        cells.loc[keep, "sum"].to_numpy(float),
    )
    np.add.at(
        counts,
        (ui[keep].astype(int), ci[keep].astype(int)),
        cells.loc[keep, "count"].to_numpy(float),
    )
    return sums, counts


def _bootstrap_statistics(
    scope: Mapping[str, object], outcome_col: str
) -> dict[str, object]:
    fitting_users = sorted(
        scope["history_standard"]["user_id"].unique(), key=lambda value: str(value)
    )
    categories = sorted(
        set(scope["test_random"]["cat1_name"].unique())
        & set(scope["test_standard"]["cat1_name"].unique()),
        key=lambda value: str(value),
    )
    arrays = {}
    for name in (
        "history_standard",
        "calibration_random",
        "calibration_standard",
        "test_random",
        "test_standard",
    ):
        sums, counts = _user_category_arrays(
            scope[name], fitting_users, categories, outcome_col
        )
        arrays[f"{name}.sum"] = sums
        arrays[f"{name}.count"] = counts
    hs = arrays["history_standard.sum"]
    hc = arrays["history_standard.count"]
    history_count = hc.sum(axis=1)
    if (history_count <= 0).any():
        raise ValueError("a frozen fitting user lacks history rows")
    history_baseline = hs.sum(axis=1) / history_count
    ts = arrays["test_standard.sum"]
    tc = arrays["test_standard.count"]
    test_count = tc.sum(axis=1)
    test_baseline = np.divide(
        ts.sum(axis=1), test_count, out=np.zeros_like(test_count), where=test_count > 0
    )
    test_observed = tc > 0
    test_cell_mean = np.divide(ts, tc, out=np.zeros_like(ts), where=test_observed)
    arrays["prospective.cell_residual"] = (
        test_cell_mean - history_baseline[:, None]
    ) * test_observed
    arrays["prospective.row_residual_sum"] = ts - history_baseline[:, None] * tc
    arrays["oracle.cell_residual"] = (
        test_cell_mean - test_baseline[:, None]
    ) * test_observed
    arrays["test_standard.observed"] = test_observed.astype(float)
    return {"users": fitting_users, "categories": categories, "arrays": arrays}


def _weighted_sum(weights: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # einsum avoids platform-specific warnings observed for NumPy's vector @
    # matrix path while preserving a single transparent synchronized reduction.
    return np.einsum("u,uc->c", weights, matrix, optimize=False)


def _vectors_from_multiplicity(
    stats: Mapping[str, object], multiplicity: np.ndarray
) -> dict[str, pd.Series]:
    arrays = stats["arrays"]
    categories = list(stats["categories"])

    def aggregate(name: str) -> np.ndarray:
        return _weighted_sum(multiplicity, arrays[name])

    crs = aggregate("calibration_random.sum")
    crc = aggregate("calibration_random.count")
    css = aggregate("calibration_standard.sum")
    csc = aggregate("calibration_standard.count")
    trs = aggregate("test_random.sum")
    trc = aggregate("test_random.count")
    tss = aggregate("test_standard.sum")
    tsc = aggregate("test_standard.count")
    if any((counts <= 0).any() for counts in (crc, csc, trc, tsc)):
        raise ValueError("resample lost category support")
    reference = trs / trc
    shares = trc / trc.sum()
    naive = tss / tsc
    cal_random_grand = float(crs.sum() / crc.sum())
    cal_standard_grand = float(css.sum() / csc.sum())
    grand_shift = naive + cal_random_grand - cal_standard_grand
    category_shift = naive + crs / crc - css / csc

    observed_denominator = aggregate("test_standard.observed")
    if (observed_denominator <= 0).any():
        raise ValueError("resample lost test user-category cells")
    prospective_ce = (
        aggregate("prospective.cell_residual") / observed_denominator
        + cal_random_grand
    )
    prospective_rw = (
        aggregate("prospective.row_residual_sum") / tsc + cal_random_grand
    )
    test_random_grand = float(trs.sum() / trc.sum())
    oracle = aggregate("oracle.cell_residual") / observed_denominator + test_random_grand
    index = pd.Index(categories, name="cat1_name")
    return {
        "reference": pd.Series(reference, index=index),
        "reference_shares": pd.Series(shares, index=index),
        "naive": pd.Series(naive, index=index),
        "grand_mean_shift": pd.Series(grand_shift, index=index),
        "category_vector_shift": pd.Series(category_shift, index=index),
        "prospective_ce_cell_equal": pd.Series(prospective_ce, index=index),
        "prospective_rw_row_weighted": pd.Series(prospective_rw, index=index),
        "oracle_contemporaneous_ce": pd.Series(oracle, index=index),
    }


def prospective_paired_bootstrap(
    scope: Mapping[str, object],
    *,
    outcome_col: str = "watch_ratio",
    n_boot: int = B_USERS,
    seed: int = SEED,
    topk: int = TOPK,
    return_replicates: bool = False,
) -> dict[str, object]:
    """Refit the frozen pipeline with one multiplicity vector in all strata."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    stats = _bootstrap_statistics(scope, outcome_col)
    n_users = len(stats["users"])
    point_vectors = _vectors_from_multiplicity(stats, np.ones(n_users, dtype=float))

    def metrics(vectors):
        return {
            name: _category_metrics(
                vectors[name],
                vectors["reference"],
                vectors["reference_shares"],
                topk=topk,
            )
            for name in METHODS
        }

    point = metrics(point_vectors)
    rng = np.random.default_rng(seed)
    probabilities = np.repeat(1.0 / n_users, n_users)
    rows: list[dict[str, float]] = []
    attempts = 0
    max_attempts = max(n_boot * 20, n_boot + 100)
    while len(rows) < n_boot and attempts < max_attempts:
        attempts += 1
        multiplicity = rng.multinomial(n_users, probabilities).astype(float)
        try:
            replicate = metrics(_vectors_from_multiplicity(stats, multiplicity))
        except ValueError as exc:
            if "resample lost" in str(exc):
                continue
            raise
        row = {
            f"{method}.{metric}": float(replicate[method][metric])
            for method in METHODS
            for metric in METRIC_NAMES
        }
        for method in METHODS:
            for metric in METRIC_NAMES:
                row[f"{method}_minus_naive.{metric}"] = (
                    float(replicate[method][metric])
                    - float(replicate["naive"][metric])
                )
        for contrast, (left, right) in PLANNED_CONTRASTS.items():
            for metric in METRIC_NAMES:
                row[f"{contrast}.{metric}"] = (
                    float(replicate[left][metric])
                    - float(replicate[right][metric])
                )
        rows.append(row)
    if len(rows) != n_boot:
        raise RuntimeError(
            f"only {len(rows)} complete draws after {attempts} attempts"
        )
    replicates = pd.DataFrame(rows)

    def interval(column: str) -> list[float]:
        values = replicates[column].to_numpy(float)
        values = values[np.isfinite(values)]
        if not len(values):
            return [float("nan"), float("nan")]
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    output: dict[str, object] = {
        "design": {
            "unit": "user",
            "same_user_multiplicity_all_five_strata": True,
            "refit_calibration_within_draw": True,
            "n_bootstrap": int(n_boot),
            "attempts": int(attempts),
            "seed": int(seed),
        },
        "methods": list(METHODS),
        "point": point,
        "ci95": {
            method: {
                metric: interval(f"{method}.{metric}")
                for metric in METRIC_NAMES
            }
            for method in METHODS
        },
        "method_minus_naive": {
            method: {
                "point": {
                    metric: float(point[method][metric])
                    - float(point["naive"][metric])
                    for metric in METRIC_NAMES
                },
                "ci95": {
                    metric: interval(f"{method}_minus_naive.{metric}")
                    for metric in METRIC_NAMES
                },
            }
            for method in METHODS
        },
        "planned_paired_contrasts": {
            contrast: {
                "definition": f"{left} minus {right} within each draw",
                "point": {
                    metric: float(point[left][metric])
                    - float(point[right][metric])
                    for metric in METRIC_NAMES
                },
                "ci95": {
                    metric: interval(f"{contrast}.{metric}")
                    for metric in METRIC_NAMES
                },
            }
            for contrast, (left, right) in PLANNED_CONTRASTS.items()
        },
    }
    if return_replicates:
        output["replicates"] = rows
    return output


def run_split(
    history_standard: pd.DataFrame,
    random: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    calibration_dates: tuple[int, int],
    test_dates: tuple[int, int],
    n_boot: int,
    seed: int,
    min_categories: int = MIN_CATEGORIES,
) -> dict[str, object]:
    calibration_random = random[random["date"].between(*calibration_dates)].copy()
    calibration_standard = standard[
        standard["date"].between(*calibration_dates)
    ].copy()
    test_random = random[random["date"].between(*test_dates)].copy()
    test_standard = standard[standard["date"].between(*test_dates)].copy()
    scope = strict_prospective_scope(
        history_standard,
        calibration_random,
        calibration_standard,
        test_random,
        test_standard,
        min_categories=min_categories,
    )
    model = fit_frozen_calibration(
        scope["history_standard"],
        scope["calibration_random"],
        scope["calibration_standard"],
    )
    evaluation = evaluate_frozen_calibration(
        model, scope["test_random"], scope["test_standard"]
    )
    reference_vector = pd.Series(evaluation["category_vectors"]["reference"])
    naive_vector = pd.Series(evaluation["category_vectors"]["naive"])
    realized_test_gap = reference_vector - naive_vector
    fitted_category_gap = pd.Series(
        {
            str(key): float(value)
            for key, value in model["calibration_category_shift"].items()
        }
    ).reindex(realized_test_gap.index)
    test_scalar_gap = float(scope["test_random"]["watch_ratio"].mean()) - float(
        scope["test_standard"]["watch_ratio"].mean()
    )
    transport_diagnostics = {
        "calibration_scalar_gap": float(model["calibration_scalar_shift"]),
        "realized_test_scalar_gap_evaluation_only": test_scalar_gap,
        "calibration_minus_test_scalar_gap": float(
            model["calibration_scalar_shift"] - test_scalar_gap
        ),
        "calibration_vs_test_category_gap_mae": float(
            (fitted_category_gap - realized_test_gap).abs().mean()
        ),
        "calibration_vs_test_category_gap_max_abs": float(
            (fitted_category_gap - realized_test_gap).abs().max()
        ),
        "calibration_vs_test_category_gap_rank_corr": float(
            fitted_category_gap.rank().corr(realized_test_gap.rank())
        ),
        "test_gap_used_for_fitting": False,
    }
    bootstrap = prospective_paired_bootstrap(
        scope, n_boot=n_boot, seed=seed
    )
    # These are computed separately as an executable leakage invariant.  The
    # bootstrap point and direct evaluation must agree to numerical precision.
    for method in METHODS:
        for metric in METRIC_NAMES:
            if not np.isclose(
                evaluation["point"][method][metric],
                bootstrap["point"][method][metric],
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                raise AssertionError(f"point mismatch for {method}.{metric}")
    return {
        "dates": {
            "history_standard": list(HISTORY_DATES),
            "calibration": list(calibration_dates),
            "test": list(test_dates),
        },
        "scope": scope["metadata"],
        "frozen_calibration": {
            "fit_uses_test_rows": False,
            "history_user_baselines": len(model["history_user_baseline"]),
            "calibration_random_anchor": float(model["calibration_random_anchor"]),
            "calibration_scalar_shift": float(model["calibration_scalar_shift"]),
            "calibration_category_shift": {
                str(key): float(value)
                for key, value in model["calibration_category_shift"].items()
            },
        },
        "evaluation": evaluation,
        "transport_diagnostics": transport_diagnostics,
        "paired_bootstrap": bootstrap,
    }


def run_p10(
    *,
    n_boot: int = B_USERS,
    secondary_n_boot: int | None = None,
    seed: int = SEED,
) -> dict[str, object]:
    random, standard, base_scope = load_kuairand_buckets(
        matched_period=False, common_videos=False
    )
    history = standard[standard["date"].between(*HISTORY_DATES)].copy()
    random_eval = random[random["date"].between(20220422, 20220508)].copy()
    standard_eval = standard[standard["date"].between(20220422, 20220508)].copy()
    if history.empty or random_eval.empty or standard_eval.empty:
        raise ValueError("the requested prospective windows are not available")
    secondary_draws = n_boot if secondary_n_boot is None else secondary_n_boot
    primary = run_split(
        history,
        random_eval,
        standard_eval,
        calibration_dates=PRIMARY_SPLIT[0],
        test_dates=PRIMARY_SPLIT[1],
        n_boot=n_boot,
        seed=seed,
    )
    secondary = run_split(
        history,
        random_eval,
        standard_eval,
        calibration_dates=SECONDARY_SPLIT[0],
        test_dates=SECONDARY_SPLIT[1],
        n_boot=secondary_draws,
        seed=seed + 1,
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": (
            "prospective descriptive calibration audit; not a causal effect estimate"
        ),
        "base_scope": base_scope,
        "leakage_guard": {
            "eligibility_source": "history standard only",
            "history_dates": list(HISTORY_DATES),
            "test_random_outcomes_used_to_fit_or_construct_deployable_vectors": False,
            "test_random_outcomes_used_for_evaluation_metrics": True,
            "test_random_role": "reference evaluation only",
            "oracle_is_deployable": False,
            "parameters_refit_after_test_opened": False,
        },
        "method_estimands": {
            "shared_evaluation_reference": (
                "test-period random-insertion exposure-weighted category means"
            ),
            "naive": "test standard exposure-row category mean",
            "grand_mean_shift": "naive plus frozen scalar calibration gap",
            "category_vector_shift": "naive plus frozen category calibration gap",
            "prospective_ce_cell_equal": (
                "history-baseline residual averaged equally over observed test "
                "standard user-category cells, plus frozen random calibration anchor"
            ),
            "prospective_rw_row_weighted": (
                "history-baseline residual averaged over test standard exposure rows, "
                "plus frozen random calibration anchor"
            ),
            "oracle_contemporaneous_ce": (
                "test-standard contemporaneous user residuals plus test-random grand "
                "mean; non-deployable diagnostic upper bound"
            ),
        },
        "primary": primary,
        "secondary_rolling_origin": {
            "independent_replication": False,
            "warning": (
                "test window overlaps the primary test; interpret only as cutoff "
                "sensitivity"
            ),
            **secondary,
        },
        "failure_boundaries": [
            "Prospective shifts transport only if calibration-period level/category gaps remain stable.",
            "History-baseline CE/RW require returning users with an observed history baseline.",
            "Cell-equal CE and exposure-row RW target different production aggregations.",
            "Evaluation conditions on frozen users present in both test logs and frozen video support.",
            "The second rolling origin overlaps the primary test and is not independent replication.",
            "The oracle contemporaneous CE reads the test-random grand mean and is not deployable.",
        ],
    }


def make_figure(result: Mapping[str, object], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "naive": "Naive",
        "grand_mean_shift": "Scalar shift",
        "category_vector_shift": "Category shift",
        "prospective_ce_cell_equal": "Prospective CE",
        "prospective_rw_row_weighted": "Prospective RW",
        "oracle_contemporaneous_ce": "Oracle CE*",
    }
    colors = {
        "naive": "#666666",
        "grand_mean_shift": "#4C78A8",
        "category_vector_shift": "#72B7B2",
        "prospective_ce_cell_equal": "#F58518",
        "prospective_rw_row_weighted": "#E45756",
        "oracle_contemporaneous_ce": "#B279A2",
    }
    methods = list(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    for ax, split_name, title in (
        (axes[0], "primary", "Primary frozen test: May 1--8"),
        (
            axes[1],
            "secondary_rolling_origin",
            "Earlier rolling origin (overlapping test)",
        ),
    ):
        cell = result[split_name]["paired_bootstrap"]
        x = np.arange(len(methods))
        for metric, marker, offset, alpha in (
            ("absolute_mae", "o", -0.09, 1.0),
            ("centered_mae", "s", 0.09, 0.82),
        ):
            values = np.array([cell["point"][m][metric] for m in methods])
            intervals = np.array([cell["ci95"][m][metric] for m in methods])
            # Percentile intervals need not contain the original-sample point
            # in small pilot runs.  Matplotlib requires non-negative bar
            # lengths, so truncate only the displayed whisker at the point.
            error = np.maximum(
                0.0,
                np.vstack(
                    (values - intervals[:, 0], intervals[:, 1] - values)
                ),
            )
            ax.errorbar(
                x + offset,
                values,
                yerr=error,
                fmt=marker,
                capsize=3,
                linewidth=1.2,
                markersize=5,
                color="#222222" if metric == "centered_mae" else "#4C78A8",
                alpha=alpha,
                label="Absolute MAE" if metric == "absolute_mae" else "Centered MAE",
            )
        ax.set_xticks(x, [labels[m] for m in methods], rotation=31, ha="right")
        ax.set_ylabel("Category error")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.99,
            0.04,
            "* non-deployable",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=colors[ORACLE_METHOD],
        )
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Prospective calibration: synchronized paired-user 95% intervals")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=B_USERS)
    parser.add_argument(
        "--secondary-bootstrap",
        type=int,
        default=None,
        help="defaults to --bootstrap",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = run_p10(
        n_boot=args.bootstrap,
        secondary_n_boot=args.secondary_bootstrap,
        seed=args.seed,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "p10_prospective_calibration.json"
    figure_path = RESULTS_DIR / "fig10_prospective_calibration.png"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
    make_figure(result, figure_path)
    print(f"wrote {json_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
