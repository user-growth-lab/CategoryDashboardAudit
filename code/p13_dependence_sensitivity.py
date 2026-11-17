"""P13: finite-calendar and finite-catalog dependence sensitivity.

The P8 synchronized user-cluster bootstrap is the paper's primary inference
procedure.  Its intervals are conditional on the realized 17-day calendar and
the realized shared video catalog.  This module does not replace those
intervals or claim multiway-bootstrap coverage.  It adds two deliberately
limited perturbation audits on the *frozen P8 canonical paired scope*:

* leave one calendar date out of both traffic buckets, without rebuilding the
  eligibility or common-support fixed point; and
* apply one shared, strictly positive exponential multiplier per video to all
  of that video's rows in both buckets.

The first output is a calendar-deletion range and the second is a finite-
catalog perturbation envelope.  Neither is a confidence interval.  Positive
video multipliers retain every observed video, category, user, and row-support
cell, avoiding accidental support changes in this diagnostic.

Running :func:`main` writes ``results/p13_dependence_sensitivity.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from audit_metrics import paired_user_audit_scope
from data_utils import RESULTS_DIR, load_kuairand_buckets
from p8_paired_audit import ERROR_METRICS, METRIC_NAMES, TOPK, _category_metrics


SEED = 42
N_VIDEO_DRAWS = 500
MIN_CATEGORIES = 5
P8_RESULT_PATH = RESULTS_DIR / "p8_paired_audit.json"
OUTPUT_PATH = RESULTS_DIR / "p13_dependence_sensitivity.json"


@dataclass(frozen=True)
class BucketArrays:
    """Integer-indexed rows for one traffic bucket."""

    user: np.ndarray
    date: np.ndarray
    video: np.ndarray
    category: np.ndarray
    outcome: np.ndarray


@dataclass(frozen=True)
class ClusteredStatistics:
    """Aligned canonical-scope arrays for deterministic reweighting."""

    users: tuple
    dates: tuple
    videos: tuple
    categories: tuple
    random: BucketArrays
    standard: BucketArrays
    outcome_col: str


def _sorted(values) -> tuple:
    return tuple(sorted(values, key=lambda value: str(value)))


def _index_rows(
    frame: pd.DataFrame,
    *,
    users: Sequence,
    dates: Sequence,
    videos: Sequence,
    categories: Sequence,
    outcome_col: str,
    label: str,
) -> BucketArrays:
    required = {"user_id", "date", "video_id", "cat1_name", outcome_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
    outcome = pd.to_numeric(frame[outcome_col], errors="coerce").to_numpy(float)
    if not np.isfinite(outcome).all():
        raise ValueError(f"{label}.{outcome_col} contains non-finite values")

    def codes(column: str, levels: Sequence) -> np.ndarray:
        mapping = {value: idx for idx, value in enumerate(levels)}
        encoded = frame[column].map(mapping)
        if encoded.isna().any():
            raise ValueError(f"{label}.{column} contains values outside support")
        return encoded.to_numpy(dtype=np.int64)

    return BucketArrays(
        user=codes("user_id", users),
        date=codes("date", dates),
        video=codes("video_id", videos),
        category=codes("cat1_name", categories),
        outcome=outcome,
    )


def prepare_clustered_statistics(
    random: pd.DataFrame,
    standard: pd.DataFrame,
    *,
    outcome_col: str = "watch_ratio",
) -> ClusteredStatistics:
    """Validate and index an already canonical P8 paired scope."""
    if random.empty or standard.empty:
        raise ValueError("dependence sensitivity requires two non-empty buckets")
    for column in ("user_id", "video_id", "cat1_name"):
        if set(random[column].unique()) != set(standard[column].unique()):
            raise ValueError(f"paired scope requires identical {column} sets")
    users = _sorted(random["user_id"].unique())
    videos = _sorted(random["video_id"].unique())
    categories = _sorted(random["cat1_name"].unique())
    dates = _sorted(set(random["date"].unique()) | set(standard["date"].unique()))
    if len(dates) < 2:
        raise ValueError("leave-one-date-out sensitivity requires at least two dates")
    return ClusteredStatistics(
        users=users,
        dates=dates,
        videos=videos,
        categories=categories,
        random=_index_rows(
            random,
            users=users,
            dates=dates,
            videos=videos,
            categories=categories,
            outcome_col=outcome_col,
            label="random",
        ),
        standard=_index_rows(
            standard,
            users=users,
            dates=dates,
            videos=videos,
            categories=categories,
            outcome_col=outcome_col,
            label="standard",
        ),
        outcome_col=outcome_col,
    )


def _validate_cluster_weights(
    weights: np.ndarray | None,
    size: int,
    label: str,
) -> np.ndarray:
    if weights is None:
        return np.ones(size, dtype=float)
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) != size:
        raise ValueError(f"{label} must contain exactly {size} entries")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{label} must be finite and non-negative")
    if not np.any(values > 0):
        raise ValueError(f"{label} cannot be all zero")
    return values


def _row_weights(
    bucket: BucketArrays,
    user_weights: np.ndarray,
    date_weights: np.ndarray,
    video_weights: np.ndarray,
) -> np.ndarray:
    return (
        user_weights[bucket.user]
        * date_weights[bucket.date]
        * video_weights[bucket.video]
    )


def _support_summary(
    bucket: BucketArrays,
    row_weights: np.ndarray,
) -> dict[str, int]:
    active = row_weights > 0
    return {
        "rows": int(active.sum()),
        "users": int(np.unique(bucket.user[active]).size),
        "videos": int(np.unique(bucket.video[active]).size),
        "categories": int(np.unique(bucket.category[active]).size),
    }


def _summarize_vectors(
    vectors: Mapping[str, pd.Series],
    *,
    topk: int,
) -> dict[str, object]:
    metrics = {
        estimator: _category_metrics(
            vectors[estimator],
            vectors["reference"],
            vectors["reference_shares"],
            topk=topk,
        )
        for estimator in ("naive", "demean")
    }
    direct_change = {
        metric: float(metrics["demean"][metric])
        - float(metrics["naive"][metric])
        for metric in METRIC_NAMES
    }
    relative_reduction = {
        metric: (
            1.0
            - float(metrics["demean"][metric])
            / float(metrics["naive"][metric])
            if float(metrics["naive"][metric]) != 0.0
            else None
        )
        for metric in ERROR_METRICS
    }
    return {
        "metrics": metrics,
        "direct_change_demean_minus_naive": direct_change,
        "relative_error_reduction_one_minus_demean_over_naive": (
            relative_reduction
        ),
    }


def evaluate_cluster_weights(
    stats: ClusteredStatistics,
    *,
    user_weights: np.ndarray | None = None,
    date_weights: np.ndarray | None = None,
    video_weights: np.ndarray | None = None,
    topk: int = TOPK,
    include_support: bool = False,
) -> dict[str, object]:
    """Evaluate P8 category estimators under shared non-negative weights.

    Date/video weights change row-level means.  User weights additionally
    weight the equal-user-category aggregation, exactly reproducing P8's
    synchronized user multiplicities when date and video weights are one.
    Zero weights are permitted for deletion diagnostics; positive multiplier
    diagnostics use strictly positive values and therefore retain support.
    """
    n_users = len(stats.users)
    n_dates = len(stats.dates)
    n_videos = len(stats.videos)
    n_categories = len(stats.categories)
    uw = _validate_cluster_weights(user_weights, n_users, "user_weights")
    dw = _validate_cluster_weights(date_weights, n_dates, "date_weights")
    vw = _validate_cluster_weights(video_weights, n_videos, "video_weights")
    rw = _row_weights(stats.random, uw, dw, vw)
    sw = _row_weights(stats.standard, uw, dw, vw)

    def category_totals(bucket: BucketArrays, weights: np.ndarray):
        sums = np.bincount(
            bucket.category,
            weights=weights * bucket.outcome,
            minlength=n_categories,
        )
        counts = np.bincount(
            bucket.category, weights=weights, minlength=n_categories
        )
        if (counts <= 0).any():
            raise ValueError("weight perturbation lost at least one category")
        return sums, counts

    random_sum, random_count = category_totals(stats.random, rw)
    standard_sum, standard_count = category_totals(stats.standard, sw)
    reference = random_sum / random_count
    shares = random_count / random_count.sum()
    naive = standard_sum / standard_count
    anchor = float(random_sum.sum() / random_count.sum())

    user_sum = np.bincount(
        stats.standard.user,
        weights=sw * stats.standard.outcome,
        minlength=n_users,
    )
    user_count = np.bincount(
        stats.standard.user, weights=sw, minlength=n_users
    )
    cell_id = stats.standard.user * n_categories + stats.standard.category
    cell_sum = np.bincount(
        cell_id,
        weights=sw * stats.standard.outcome,
        minlength=n_users * n_categories,
    )
    cell_count = np.bincount(
        cell_id, weights=sw, minlength=n_users * n_categories
    )
    observed_id = np.flatnonzero(cell_count > 0)
    cell_user = observed_id // n_categories
    cell_category = observed_id % n_categories
    valid = user_count[cell_user] > 0
    observed_id = observed_id[valid]
    cell_user = cell_user[valid]
    cell_category = cell_category[valid]
    cell_mean = cell_sum[observed_id] / cell_count[observed_id]
    user_mean = user_sum[cell_user] / user_count[cell_user]
    residual = cell_mean - user_mean
    aggregation_weight = uw[cell_user]
    residual_sum = np.bincount(
        cell_category,
        weights=aggregation_weight * residual,
        minlength=n_categories,
    )
    residual_count = np.bincount(
        cell_category, weights=aggregation_weight, minlength=n_categories
    )
    if (residual_count <= 0).any():
        raise ValueError("weight perturbation lost standard user-category support")
    demean = residual_sum / residual_count + anchor

    index = pd.Index(stats.categories, name="cat1_name")
    vectors = {
        "reference": pd.Series(reference, index=index),
        "reference_shares": pd.Series(shares, index=index),
        "naive": pd.Series(naive, index=index),
        "demean": pd.Series(demean, index=index),
    }
    output = _summarize_vectors(vectors, topk=topk)
    if include_support:
        output["effective_support"] = {
            "random": _support_summary(stats.random, rw),
            "standard": _support_summary(stats.standard, sw),
            "standard_user_category_cells": int(len(observed_id)),
        }
    return output


def _metric_sections(point: Mapping[str, object]) -> dict[str, Mapping[str, float]]:
    return {
        "naive": point["metrics"]["naive"],
        "demean": point["metrics"]["demean"],
        "direct_change": point["direct_change_demean_minus_naive"],
        "relative_error_reduction": point[
            "relative_error_reduction_one_minus_demean_over_naive"
        ],
    }


def _favorable(metric: str, direct_change: float) -> bool:
    return direct_change < 0.0 if metric in ERROR_METRICS else direct_change > 0.0


def leave_one_date_out_sensitivity(
    stats: ClusteredStatistics,
    *,
    topk: int = TOPK,
) -> dict[str, object]:
    """Delete each date jointly from both buckets on the frozen P8 scope."""
    canonical = evaluate_cluster_weights(stats, topk=topk)
    canonical_sections = _metric_sections(canonical)
    cells: dict[str, object] = {}
    points = []
    for date_index, date in enumerate(stats.dates):
        weights = np.ones(len(stats.dates), dtype=float)
        weights[date_index] = 0.0
        point = evaluate_cluster_weights(
            stats,
            date_weights=weights,
            topk=topk,
            include_support=True,
        )
        points.append(point)
        cells[str(date)] = point

    ranges: dict[str, object] = {}
    deviations: dict[str, object] = {}
    for section in ("naive", "demean", "direct_change", "relative_error_reduction"):
        ranges[section] = {}
        deviations[section] = {}
        keys = canonical_sections[section].keys()
        for metric in keys:
            values = np.asarray(
                [_metric_sections(point)[section][metric] for point in points],
                dtype=float,
            )
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite leave-one-date-out {section}.{metric}")
            ranges[section][metric] = {
                "min": float(values.min()),
                "max": float(values.max()),
            }
            deviations[section][metric] = float(
                np.max(np.abs(values - float(canonical_sections[section][metric])))
            )

    direct_rows = [
        point["direct_change_demean_minus_naive"] for point in points
    ]
    directional = {
        metric: {
            "favorable_count": int(
                sum(_favorable(metric, float(row[metric])) for row in direct_rows)
            ),
            "total": len(points),
            "stable_in_every_omission": bool(
                all(_favorable(metric, float(row[metric])) for row in direct_rows)
            ),
        }
        for metric in METRIC_NAMES
    }
    all_error = [
        all(float(row[metric]) < 0.0 for metric in ERROR_METRICS)
        for row in direct_rows
    ]
    return {
        "design": {
            "method": "joint leave-one-date-out deletion",
            "interpretation": "calendar-deletion sensitivity range",
            "same_date_deleted_from_both_buckets": True,
            "selection_scope_frozen": True,
            "support_repaired_after_deletion": False,
            "is_confidence_interval": False,
            "estimand_note": (
                "P8 user/video/category eligibility is never rebuilt; estimates "
                "use the remaining rows, so a user-category cell with no "
                "remaining row does not contribute to that deletion cell"
            ),
        },
        "n_omitted_dates": len(stats.dates),
        "omitted_dates": [int(value) for value in stats.dates],
        "cells": cells,
        "range_across_omitted_dates": ranges,
        "max_absolute_deviation_from_canonical": deviations,
        "directional_stability": {
            **directional,
            "all_three_error_metrics_favorable_count": int(sum(all_error)),
            "all_three_error_metrics_favorable_in_every_omission": bool(
                all(all_error)
            ),
        },
    }


def _quantiles(values: Sequence[float | None]) -> dict[str, float | int | None]:
    array = np.asarray(
        [value for value in values if value is not None], dtype=float
    )
    if not len(array):
        return {
            "q02_5": None,
            "q50": None,
            "q97_5": None,
            "min": None,
            "max": None,
            "n_finite": 0,
        }
    if not np.isfinite(array).all():
        raise ValueError("sensitivity draws contain non-finite values")
    return {
        "q02_5": float(np.percentile(array, 2.5)),
        "q50": float(np.percentile(array, 50.0)),
        "q97_5": float(np.percentile(array, 97.5)),
        "min": float(array.min()),
        "max": float(array.max()),
        "n_finite": int(len(array)),
    }


def shared_video_sensitivity(
    stats: ClusteredStatistics,
    *,
    n_draws: int = N_VIDEO_DRAWS,
    seed: int = SEED,
    topk: int = TOPK,
) -> dict[str, object]:
    """Perturb the realized catalog with shared positive video multipliers."""
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(int(n_draws)):
        weights = np.maximum(
            rng.exponential(size=len(stats.videos)), np.finfo(float).tiny
        )
        weights /= weights.mean()
        points.append(
            evaluate_cluster_weights(stats, video_weights=weights, topk=topk)
        )

    sections: dict[str, dict[str, list[float | None]]] = {
        "naive": {},
        "demean": {},
        "direct_change": {},
        "relative_error_reduction": {},
    }
    for section in sections:
        keys = _metric_sections(points[0])[section].keys()
        sections[section] = {
            metric: [
                (
                    float(_metric_sections(point)[section][metric])
                    if _metric_sections(point)[section][metric] is not None
                    else None
                )
                for point in points
            ]
            for metric in keys
        }
    quantile_summary = {
        section: {
            metric: _quantiles(values) for metric, values in metrics.items()
        }
        for section, metrics in sections.items()
    }
    direct_rows = [
        point["direct_change_demean_minus_naive"] for point in points
    ]
    directional = {
        metric: {
            "favorable_draws": int(
                sum(_favorable(metric, float(row[metric])) for row in direct_rows)
            ),
            "fraction": float(
                np.mean(
                    [_favorable(metric, float(row[metric])) for row in direct_rows]
                )
            ),
        }
        for metric in METRIC_NAMES
    }
    all_error = np.asarray(
        [
            all(float(row[metric]) < 0.0 for metric in ERROR_METRICS)
            for row in direct_rows
        ],
        dtype=float,
    )
    return {
        "design": {
            "method": "shared positive exponential multiplier perturbation",
            "cluster": "video",
            "same_video_weights_in_both_buckets": True,
            "strictly_positive_weights_preserve_observed_support": True,
            "selection_scope_frozen": True,
            "is_confidence_interval": False,
            "interpretation": "finite-catalog perturbation envelope",
            "warning": (
                "quantiles summarize deliberately strong catalog reweighting; "
                "they are not a calibrated cluster bootstrap or coverage interval"
            ),
        },
        "n_draws": int(n_draws),
        "seed": int(seed),
        "sensitivity_quantiles": quantile_summary,
        "directional_stability": {
            **directional,
            "all_three_error_metrics_favorable_draws": int(all_error.sum()),
            "all_three_error_metrics_favorable_fraction": float(all_error.mean()),
        },
    }


def _compare_with_frozen_p8(
    scope: Mapping[str, object],
    canonical: Mapping[str, object],
    p8_path: Path,
) -> dict[str, object]:
    if not p8_path.is_file():
        raise FileNotFoundError(
            f"frozen P8 result is required for scope verification: {p8_path}"
        )
    p8 = json.loads(p8_path.read_text(encoding="utf-8"))
    expected_scope = p8["paired_scope"]["primary_scope_id"]
    actual_scope = scope["primary_scope_id"]
    if actual_scope != expected_scope:
        raise ValueError(
            f"P13 scope {actual_scope} differs from frozen P8 scope {expected_scope}"
        )
    frozen_point = p8["main_paired_metrics_and_bootstrap"]["point"]
    max_difference = 0.0
    for estimator in ("naive", "demean"):
        for metric in METRIC_NAMES:
            difference = abs(
                float(canonical["metrics"][estimator][metric])
                - float(frozen_point[estimator][metric])
            )
            max_difference = max(max_difference, difference)
    if max_difference > 1e-10:
        raise ValueError(
            f"P13 canonical metrics differ from frozen P8 by {max_difference}"
        )
    return {
        "p8_result": str(p8_path.relative_to(p8_path.parents[1])),
        "scope_id_matches": True,
        "point_metrics_match": True,
        "max_absolute_metric_difference": float(max_difference),
        "p8_generated_at_utc": p8.get("generated_at_utc"),
    }


def run_p13(
    *,
    root: str | Path | None = None,
    category_path: str | Path | None = None,
    n_video_draws: int = N_VIDEO_DRAWS,
    seed: int = SEED,
) -> dict[str, object]:
    """Run P13 on the exact canonical P8 paired scope."""
    random, standard, base_scope = load_kuairand_buckets(
        root=root,
        category_path=category_path,
        matched_period=True,
        tab=1,
        common_videos=True,
    )
    paired_random, paired_standard, paired_scope = paired_user_audit_scope(
        random, standard, min_categories=MIN_CATEGORIES
    )
    stats = prepare_clustered_statistics(paired_random, paired_standard)
    canonical = evaluate_cluster_weights(stats, include_support=True)
    verification = _compare_with_frozen_p8(
        paired_scope, canonical, P8_RESULT_PATH
    )
    lodo = leave_one_date_out_sensitivity(stats)
    video = shared_video_sensitivity(
        stats, n_draws=n_video_draws, seed=seed + 1
    )
    canonical_direct = canonical["direct_change_demean_minus_naive"]
    canonical_favorable = {
        metric: _favorable(metric, float(canonical_direct[metric]))
        for metric in METRIC_NAMES
    }
    return {
        "analysis": "P13 finite-calendar and finite-catalog dependence sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "scope": {
            "definition": "exact frozen P8 canonical paired-user/video scope",
            "primary_scope_id": paired_scope["primary_scope_id"],
            "min_categories": MIN_CATEGORIES,
            "dates": [int(value) for value in stats.dates],
            "n_dates": len(stats.dates),
            "random": paired_scope["final"]["random"],
            "standard": paired_scope["final"]["standard"],
            "base_scope": base_scope,
        },
        "p8_frozen_result_verification": verification,
        "primary_inference_boundary": {
            "primary_interval": "P8 synchronized paired-user cluster bootstrap",
            "conditional_on_realized_calendar": True,
            "conditional_on_realized_video_catalog": True,
            "p13_outputs_replace_primary_interval": False,
            "p13_outputs_are_confidence_intervals": False,
        },
        "canonical_point": canonical,
        "leave_one_date_out_sensitivity": lodo,
        "shared_video_positive_multiplier_sensitivity": video,
        "conclusion_stability": {
            "canonical_direction_favorable": canonical_favorable,
            "canonical_all_three_error_metrics_favorable": bool(
                all(canonical_favorable[metric] for metric in ERROR_METRICS)
            ),
            "leave_one_date_out_all_three_favorable_in_every_omission": lodo[
                "directional_stability"
            ]["all_three_error_metrics_favorable_in_every_omission"],
            "video_perturbation_all_three_favorable_fraction": video[
                "directional_stability"
            ]["all_three_error_metrics_favorable_fraction"],
        },
        "limitations": [
            "The primary P8 user-cluster interval remains conditional on the realized catalog and calendar.",
            "Leave-one-date-out is a deletion range with only 17 observed dates, not a date-cluster confidence interval.",
            "Positive video multipliers are a support-preserving finite-catalog stress test, not a calibrated video-cluster bootstrap.",
            "Neither diagnostic identifies request-level position, the replaced item, or a new platform/window.",
        ],
    }


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--category-path", type=Path, default=None)
    parser.add_argument("--video-draws", type=int, default=N_VIDEO_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    result = run_p13(
        root=args.root,
        category_path=args.category_path,
        n_video_draws=args.video_draws,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        handle.write("\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
