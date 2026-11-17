"""P9: a complete, descriptive dashboard-error ledger.

The analysis holds the P8 canonical paired-user cohort fixed, but restores
each log's bucket-specific video support.  For every category, it follows the
auditable path

    s_full - r_full
      = (s_full - s_common)
      + within_video_residual
      + video_mix
      + (r_common - r_full).

The middle two terms are the existing symmetric decomposition on common video
support.  The first and last terms expose what the common-support restriction
conditions away.  A random-full-exposure-weighted inner-product ledger then
accounts for the level and centered-shape squared error.  Component
contributions may be negative when paths offset one another.

This is descriptive accounting, not causal attribution, and the path identity
is standard algebra rather than a new theorem.

Running this module writes ``results/p9_error_ledger.json`` and
``results/fig9_error_ledger.png``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from audit_metrics import (
    paired_user_audit_scope,
    symmetric_category_gap_decomposition,
)
from category_labels import CATEGORY_ENGLISH_NAMES
from data_utils import RESULTS_DIR, load_kuairand_buckets


SEED = 47
B_USERS = 1000
MIN_CATEGORIES = 5
CHUNK_SIZE = 25
COMPONENTS = (
    "standard_support_entry_exit",
    "within_video_residual",
    "video_mix",
    "random_support_entry_exit",
)
DISPLAY_NAMES = {
    "standard_support_entry_exit": "Standard support",
    "within_video_residual": "Within-video residual",
    "video_mix": "Video mix",
    "random_support_entry_exit": "Random support",
}
ACCOUNTING_WARNING = (
    "Descriptive accounting only, not causal attribution. The four-term path "
    "is an exact algebraic audit of observed log summaries, not a new identity."
)


def english_category_label(category: object) -> str:
    """Return the shared publication label and reject untranslated text.

    A single shared table prevents released figures from silently drifting to
    different English names. ASCII fixture labels remain valid for unit tests.
    """
    original = str(category)
    label = str(CATEGORY_ENGLISH_NAMES.get(original, original))
    if not label.isascii():
        raise ValueError(f"missing English category label for {original!r}")
    return label


def _require_columns(frame: pd.DataFrame, label: str) -> None:
    required = {"user_id", "video_id", "cat1_name", "watch_ratio"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{label} cannot be empty")
    if frame[["user_id", "video_id", "cat1_name"]].isna().any().any():
        raise ValueError(f"{label} has missing scope identifiers")
    outcome = pd.to_numeric(frame["watch_ratio"], errors="coerce").to_numpy(float)
    if not np.isfinite(outcome).all():
        raise ValueError(f"{label}.watch_ratio contains missing/non-finite values")


def _support(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "videos": int(frame["video_id"].nunique()),
        "categories": int(frame["cat1_name"].nunique()),
    }


def canonical_ledger_scope(
    random_full: pd.DataFrame,
    standard_full: pd.DataFrame,
    *,
    min_categories: int = MIN_CATEGORIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Freeze P8 paired users, then recover bucket-specific full supports.

    Eligibility and the paired fixed point are constructed on the initial
    common-video base exactly as in P8.  Those final users and categories are
    then frozen while every categorized full-support row for those users is
    restored separately in each bucket.
    """
    _require_columns(random_full, "random_full")
    _require_columns(standard_full, "standard_full")
    shared_videos = set(random_full["video_id"].unique()) & set(
        standard_full["video_id"].unique()
    )
    if not shared_videos:
        raise ValueError("full logs have no common video support")
    random_base = random_full[random_full["video_id"].isin(shared_videos)].copy()
    standard_base = standard_full[
        standard_full["video_id"].isin(shared_videos)
    ].copy()
    base_categories = set(random_base["cat1_name"].unique()) & set(
        standard_base["cat1_name"].unique()
    )
    random_base = random_base[random_base["cat1_name"].isin(base_categories)].copy()
    standard_base = standard_base[
        standard_base["cat1_name"].isin(base_categories)
    ].copy()
    random_common, standard_common, paired_scope = paired_user_audit_scope(
        random_base,
        standard_base,
        min_categories=min_categories,
    )

    canonical_users = set(random_common["user_id"].unique())
    categories = sorted(random_common["cat1_name"].unique(), key=str)
    category_set = set(categories)
    random_full_fixed = random_full[
        random_full["user_id"].isin(canonical_users)
        & random_full["cat1_name"].isin(category_set)
    ].copy()
    standard_full_fixed = standard_full[
        standard_full["user_id"].isin(canonical_users)
        & standard_full["cat1_name"].isin(category_set)
    ].copy()
    frames = {
        "random_full": random_full_fixed,
        "standard_full": standard_full_fixed,
        "random_common": random_common,
        "standard_common": standard_common,
    }
    for label, frame in frames.items():
        users = set(frame["user_id"].unique())
        if users != canonical_users:
            raise ValueError(f"{label} is missing at least one canonical paired user")
        if set(frame["cat1_name"].unique()) != category_set:
            raise ValueError(f"{label} does not cover every canonical category")
    if set(random_common["video_id"].unique()) != set(
        standard_common["video_id"].unique()
    ):
        raise AssertionError("canonical common-video supports differ")

    metadata = {
        "definition": (
            "P8 eligibility and paired U/V fixed point on the common-video "
            "base; final paired users and categories are frozen before each "
            "bucket's categorized full video support is restored"
        ),
        "min_standard_categories": int(min_categories),
        "canonical_paired_scope_id": paired_scope["primary_scope_id"],
        "canonical_user_id_sha256": paired_scope["final"]["user_id_sha256"],
        "canonical_common_video_id_sha256": paired_scope["final"][
            "video_id_sha256"
        ],
        "canonical_category_name_sha256": paired_scope["final"][
            "category_name_sha256"
        ],
        "supports": {label: _support(frame) for label, frame in frames.items()},
        "full_video_support_overlap": {
            "intersection": int(
                len(
                    set(random_full_fixed["video_id"].unique())
                    & set(standard_full_fixed["video_id"].unique())
                )
            ),
            "random_only": int(
                len(
                    set(random_full_fixed["video_id"].unique())
                    - set(standard_full_fixed["video_id"].unique())
                )
            ),
            "standard_only": int(
                len(
                    set(standard_full_fixed["video_id"].unique())
                    - set(random_full_fixed["video_id"].unique())
                )
            ),
        },
        "excluded_noncanonical_categories": {
            "random": sorted(
                set(random_full["cat1_name"].unique()) - category_set, key=str
            ),
            "standard": sorted(
                set(standard_full["cat1_name"].unique()) - category_set, key=str
            ),
        },
        "invariants": {
            "same_canonical_users_all_four_cells": True,
            "same_categories_all_four_cells": True,
            "common_cells_have_identical_video_support": True,
            "full_video_supports_allowed_to_differ": True,
        },
        "paired_scope": paired_scope,
    }
    return (
        random_full_fixed.reset_index(drop=True),
        standard_full_fixed.reset_index(drop=True),
        random_common.reset_index(drop=True),
        standard_common.reset_index(drop=True),
        metadata,
    )


def _category_mean(frame: pd.DataFrame, categories: list) -> pd.Series:
    vector = frame.groupby("cat1_name")["watch_ratio"].mean().reindex(categories)
    if vector.isna().any():
        raise ValueError("category vector has incomplete support")
    return vector.astype(float)


def category_error_components(
    random_full: pd.DataFrame,
    standard_full: pd.DataFrame,
    random_common: pd.DataFrame,
    standard_common: pd.DataFrame,
    *,
    atol: float = 1e-10,
) -> pd.DataFrame:
    """Return the four exact path components for each category."""
    categories = sorted(random_common["cat1_name"].unique(), key=str)
    expected = set(categories)
    for label, frame in {
        "random_full": random_full,
        "standard_full": standard_full,
        "random_common": random_common,
        "standard_common": standard_common,
    }.items():
        _require_columns(frame, label)
        if set(frame["cat1_name"].unique()) != expected:
            raise ValueError(f"{label} category support differs")

    s_full = _category_mean(standard_full, categories)
    r_full = _category_mean(random_full, categories)
    s_common = _category_mean(standard_common, categories)
    r_common = _category_mean(random_common, categories)
    common = symmetric_category_gap_decomposition(
        standard_common,
        random_common,
        outcome_col="watch_ratio",
        atol=atol,
    ).reindex(categories)
    if common.isna().any().any():
        raise ValueError("common-support decomposition has incomplete categories")

    table = pd.DataFrame(index=pd.Index(categories, name="cat1_name"))
    table["standard_full_mean"] = s_full
    table["standard_common_mean"] = s_common
    table["random_common_mean"] = r_common
    table["random_full_mean"] = r_full
    table["standard_support_entry_exit"] = s_full - s_common
    table["within_video_residual"] = common["response_context"]
    table["video_mix"] = common["item_mix"]
    table["random_support_entry_exit"] = r_common - r_full
    table["full_gap"] = s_full - r_full
    table["reconstructed_full_gap"] = table[list(COMPONENTS)].sum(axis=1)
    table["reconstruction_error"] = (
        table["full_gap"] - table["reconstructed_full_gap"]
    )
    counts = random_full["cat1_name"].value_counts().reindex(categories).astype(float)
    table["random_full_exposure_share"] = counts / counts.sum()
    maximum_error = float(table["reconstruction_error"].abs().max())
    if maximum_error > atol:
        raise AssertionError(f"four-term path failed reconstruction: {maximum_error}")
    return table


def weighted_squared_error_ledger(
    components: pd.DataFrame,
    random_shares: pd.Series,
    *,
    atol: float = 1e-12,
) -> dict[str, object]:
    """Account for level/shape squared error by component inner products.

    If ``D = sum_j g_j`` and ``w`` are normalized random exposure shares, then
    ``<D,D>_w = sum_j <g_j,D>_w``.  Weighted centering orthogonally splits each
    inner product into level and shape terms.  Signed contributions are
    intentional: a component can offset another component along this path.
    """
    if tuple(components.columns) != COMPONENTS:
        components = components.reindex(columns=list(COMPONENTS))
    if components.isna().any().any():
        raise ValueError("components must contain every ledger component")
    weights = random_shares.reindex(components.index).astype(float)
    if weights.isna().any() or not np.isfinite(weights.to_numpy()).all():
        raise ValueError("random shares must be finite on every category")
    if (weights < 0).any() or float(weights.sum()) <= 0:
        raise ValueError("random shares must be nonnegative with positive mass")
    weights /= float(weights.sum())
    matrix = components.to_numpy(float).T
    w = weights.to_numpy(float)
    gap = matrix.sum(axis=0)
    component_means = matrix @ w
    gap_mean = float(gap @ w)
    centered_gap = gap - gap_mean
    centered_components = matrix - component_means[:, None]
    level_contributions = component_means * gap_mean
    shape_contributions = (centered_components * centered_gap[None, :]) @ w
    total_contributions = (matrix * gap[None, :]) @ w
    level_error = float(gap_mean**2)
    shape_error = float((centered_gap**2) @ w)
    total_error = float((gap**2) @ w)

    errors = {
        "orthogonal_total_minus_level_plus_shape": total_error
        - level_error
        - shape_error,
        "level_component_sum": level_error - float(level_contributions.sum()),
        "shape_component_sum": shape_error - float(shape_contributions.sum()),
        "total_component_sum": total_error - float(total_contributions.sum()),
    }
    if max(abs(value) for value in errors.values()) > atol:
        raise AssertionError(f"weighted squared-error ledger failed: {errors}")

    def share(value: float, denominator: float) -> float | None:
        return float(value / denominator) if denominator != 0.0 else None

    component_output = {}
    for idx, name in enumerate(COMPONENTS):
        component_output[name] = {
            "display_name": DISPLAY_NAMES[name],
            "weighted_mean": float(component_means[idx]),
            "level_inner_product": float(level_contributions[idx]),
            "shape_inner_product": float(shape_contributions[idx]),
            "total_inner_product": float(total_contributions[idx]),
            "level_share_signed": share(level_contributions[idx], level_error),
            "shape_share_signed": share(shape_contributions[idx], shape_error),
            "total_share_signed": share(total_contributions[idx], total_error),
        }
    return {
        "definition": (
            "with D=sum_j g_j and random-full row-exposure weights w: level "
            "contribution_j=mean_w(g_j)*mean_w(D); shape contribution_j="
            "<g_j-mean_w(g_j), D-mean_w(D)>_w; signed contributions sum "
            "exactly to the corresponding squared errors"
        ),
        "weight_definition": "random-full category row-exposure shares",
        "center_definition": "subtract the random-full-share-weighted mean",
        "signed_contributions_can_be_negative": True,
        "full_gap": {
            "weighted_mean": gap_mean,
            "level_squared_error": level_error,
            "shape_squared_error": shape_error,
            "total_weighted_squared_error": total_error,
            "weighted_rmse": float(np.sqrt(total_error)),
        },
        "components": component_output,
        "reconstruction_error": {key: float(value) for key, value in errors.items()},
    }


def _user_category_arrays(
    frame: pd.DataFrame,
    users: list,
    categories: list,
) -> tuple[np.ndarray, np.ndarray]:
    user_index = {value: idx for idx, value in enumerate(users)}
    category_index = {value: idx for idx, value in enumerate(categories)}
    cells = (
        frame.groupby(["user_id", "cat1_name"], sort=False)["watch_ratio"]
        .agg(["sum", "count"])
        .reset_index()
    )
    ui = cells["user_id"].map(user_index)
    ci = cells["cat1_name"].map(category_index)
    if ui.isna().any() or ci.isna().any():
        raise ValueError("frame has users/categories outside aligned support")
    shape = (len(users), len(categories))
    sums = np.zeros(shape, dtype=float)
    counts = np.zeros(shape, dtype=float)
    np.add.at(sums, (ui.to_numpy(int), ci.to_numpy(int)), cells["sum"].to_numpy(float))
    np.add.at(
        counts,
        (ui.to_numpy(int), ci.to_numpy(int)),
        cells["count"].to_numpy(float),
    )
    return sums, counts


def _user_video_sparse_arrays(
    frame: pd.DataFrame,
    users: list,
    videos: list,
):
    from scipy import sparse

    user_index = {value: idx for idx, value in enumerate(users)}
    video_index = {value: idx for idx, value in enumerate(videos)}
    cells = (
        frame.groupby(["user_id", "video_id"], sort=False)["watch_ratio"]
        .agg(["sum", "count"])
        .reset_index()
    )
    ui = cells["user_id"].map(user_index)
    vi = cells["video_id"].map(video_index)
    if ui.isna().any() or vi.isna().any():
        raise ValueError("common frame has users/videos outside aligned support")
    indices = (ui.to_numpy(int), vi.to_numpy(int))
    shape = (len(users), len(videos))
    sums = sparse.csr_matrix((cells["sum"].to_numpy(float), indices), shape=shape)
    counts = sparse.csr_matrix(
        (cells["count"].to_numpy(float), indices), shape=shape
    )
    return sums, counts


def ledger_sufficient_statistics(
    random_full: pd.DataFrame,
    standard_full: pd.DataFrame,
    random_common: pd.DataFrame,
    standard_common: pd.DataFrame,
) -> dict[str, object]:
    """Build aligned dense category and sparse common-video statistics."""
    user_set = set(random_common["user_id"].unique())
    category_set = set(random_common["cat1_name"].unique())
    video_set = set(random_common["video_id"].unique())
    frames = (random_full, standard_full, random_common, standard_common)
    if any(set(frame["user_id"].unique()) != user_set for frame in frames):
        raise ValueError("ledger bootstrap requires identical user sets")
    if any(set(frame["cat1_name"].unique()) != category_set for frame in frames):
        raise ValueError("ledger bootstrap requires identical category sets")
    if set(standard_common["video_id"].unique()) != video_set:
        raise ValueError("common cells require identical video support")
    users = sorted(user_set, key=str)
    categories = sorted(category_set, key=str)
    videos = sorted(video_set, key=str)

    random_map = random_common.drop_duplicates("video_id").set_index("video_id")[
        "cat1_name"
    ]
    standard_map = standard_common.drop_duplicates("video_id").set_index("video_id")[
        "cat1_name"
    ]
    aligned_video_categories = random_map.reindex(videos)
    if aligned_video_categories.isna().any() or not aligned_video_categories.equals(
        standard_map.reindex(videos)
    ):
        raise ValueError("common cells require one shared video taxonomy")
    category_index = {value: idx for idx, value in enumerate(categories)}
    video_category_index = aligned_video_categories.map(category_index).to_numpy(int)

    stats: dict[str, object] = {
        "users": users,
        "categories": categories,
        "videos": videos,
        "video_category_index": video_category_index,
    }
    for label, frame in {
        "random_full": random_full,
        "standard_full": standard_full,
        "random_common": random_common,
        "standard_common": standard_common,
    }.items():
        sums, counts = _user_category_arrays(frame, users, categories)
        stats[f"{label}_category_sums"] = sums
        stats[f"{label}_category_counts"] = counts
    for label, frame in {
        "random_common": random_common,
        "standard_common": standard_common,
    }.items():
        sums, counts = _user_video_sparse_arrays(frame, users, videos)
        stats[f"{label}_video_sums"] = sums
        stats[f"{label}_video_counts"] = counts
    return stats


def _batch_ledger(stats: Mapping[str, object], weights: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate category paths and aggregate ledgers for a weight batch."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != len(stats["users"]):
        raise ValueError("weights must have shape (draws, paired users)")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("positive multiplier weights must be finite and positive")

    category_vectors = {}
    category_counts = {}
    for label in (
        "random_full",
        "standard_full",
        "random_common",
        "standard_common",
    ):
        sums = np.asarray(stats[f"{label}_category_sums"], dtype=float)
        counts = np.asarray(stats[f"{label}_category_counts"], dtype=float)
        # Avoid the vector/matrix Accelerate path.  Some macOS NumPy builds
        # emit spurious divide/overflow warnings for finite non-negative
        # arrays.  The explicit einsum is deterministic and warning-free.
        weighted_sums = np.einsum("bu,uc->bc", weights, sums, optimize=False)
        weighted_counts = np.einsum("bu,uc->bc", weights, counts, optimize=False)
        if (weighted_counts <= 0).any():  # pragma: no cover - positive weights
            raise AssertionError("positive multiplier lost category support")
        category_vectors[label] = weighted_sums / weighted_counts
        category_counts[label] = weighted_counts

    def weighted_video(label: str) -> tuple[np.ndarray, np.ndarray]:
        sums = stats[f"{label}_video_sums"]
        counts = stats[f"{label}_video_counts"]
        weighted_sums = np.asarray(sums.T.dot(weights.T), dtype=float).T
        weighted_counts = np.asarray(counts.T.dot(weights.T), dtype=float).T
        if (weighted_counts <= 0).any():  # pragma: no cover - positive weights
            raise AssertionError("positive multiplier lost fixed video support")
        return weighted_sums / weighted_counts, weighted_counts

    r_mean, r_count = weighted_video("random_common")
    s_mean, s_count = weighted_video("standard_common")
    video_category_index = np.asarray(stats["video_category_index"], dtype=int)
    n_categories = len(stats["categories"])

    def category_totals(counts: np.ndarray) -> np.ndarray:
        output = np.zeros((len(counts), n_categories), dtype=float)
        for draw in range(len(counts)):
            output[draw] = np.bincount(
                video_category_index,
                weights=counts[draw],
                minlength=n_categories,
            )
        return output

    r_category_count = category_totals(r_count)
    s_category_count = category_totals(s_count)
    r_share = r_count / r_category_count[:, video_category_index]
    s_share = s_count / s_category_count[:, video_category_index]
    within_video = np.zeros((len(weights), n_categories), dtype=float)
    video_mix = np.zeros_like(within_video)
    within_video_at_item = 0.5 * (s_share + r_share) * (s_mean - r_mean)
    video_mix_at_item = 0.5 * (s_mean + r_mean) * (s_share - r_share)
    for draw in range(len(weights)):
        within_video[draw] = np.bincount(
            video_category_index,
            weights=within_video_at_item[draw],
            minlength=n_categories,
        )
        video_mix[draw] = np.bincount(
            video_category_index,
            weights=video_mix_at_item[draw],
            minlength=n_categories,
        )

    components = np.stack(
        [
            category_vectors["standard_full"]
            - category_vectors["standard_common"],
            within_video,
            video_mix,
            category_vectors["random_common"] - category_vectors["random_full"],
        ],
        axis=1,
    )
    full_gap = category_vectors["standard_full"] - category_vectors["random_full"]
    reconstructed_gap = components.sum(axis=1)
    reconstruction_error = full_gap - reconstructed_gap

    random_weights = category_counts["random_full"]
    random_weights /= random_weights.sum(axis=1, keepdims=True)
    gap_mean = np.sum(random_weights * full_gap, axis=1)
    component_means = np.sum(
        components * random_weights[:, None, :], axis=2
    )
    centered_gap = full_gap - gap_mean[:, None]
    centered_components = components - component_means[:, :, None]
    level_contributions = component_means * gap_mean[:, None]
    shape_contributions = np.sum(
        centered_components
        * centered_gap[:, None, :]
        * random_weights[:, None, :],
        axis=2,
    )
    total_contributions = np.sum(
        components * full_gap[:, None, :] * random_weights[:, None, :], axis=2
    )
    level_error = gap_mean**2
    shape_error = np.sum(random_weights * centered_gap**2, axis=1)
    total_error = np.sum(random_weights * full_gap**2, axis=1)
    ledger_error = np.column_stack(
        [
            total_error - level_error - shape_error,
            level_error - level_contributions.sum(axis=1),
            shape_error - shape_contributions.sum(axis=1),
            total_error - total_contributions.sum(axis=1),
        ]
    )
    return {
        "components": components,
        "full_gap": full_gap,
        "random_shares": random_weights,
        "gap_mean": gap_mean,
        "level_error": level_error,
        "shape_error": shape_error,
        "total_error": total_error,
        "component_means": component_means,
        "level_contributions": level_contributions,
        "shape_contributions": shape_contributions,
        "total_contributions": total_contributions,
        "path_reconstruction_error": reconstruction_error,
        "ledger_reconstruction_error": ledger_error,
    }


def paired_positive_multiplier_bootstrap(
    random_full: pd.DataFrame,
    standard_full: pd.DataFrame,
    random_common: pd.DataFrame,
    standard_common: pd.DataFrame,
    *,
    n_boot: int = B_USERS,
    seed: int = SEED,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, object]:
    """Paired-user positive-multiplier uncertainty for the full ledger."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    stats = ledger_sufficient_statistics(
        random_full,
        standard_full,
        random_common,
        standard_common,
    )
    n_users = len(stats["users"])
    n_categories = len(stats["categories"])
    component_draws = np.empty((n_boot, len(COMPONENTS), n_categories), dtype=float)
    gap_draws = np.empty((n_boot, n_categories), dtype=float)
    aggregate = {
        "weighted_mean": np.empty(n_boot),
        "level_squared_error": np.empty(n_boot),
        "shape_squared_error": np.empty(n_boot),
        "total_weighted_squared_error": np.empty(n_boot),
        "component_weighted_mean": np.empty((n_boot, len(COMPONENTS))),
        "component_level_inner_product": np.empty((n_boot, len(COMPONENTS))),
        "component_shape_inner_product": np.empty((n_boot, len(COMPONENTS))),
        "component_total_inner_product": np.empty((n_boot, len(COMPONENTS))),
        "component_level_share": np.empty((n_boot, len(COMPONENTS))),
        "component_shape_share": np.empty((n_boot, len(COMPONENTS))),
        "component_total_share": np.empty((n_boot, len(COMPONENTS))),
    }
    maximum_path_error = 0.0
    maximum_ledger_error = 0.0
    rng = np.random.default_rng(seed)
    position = 0
    while position < n_boot:
        batch = min(chunk_size, n_boot - position)
        weights = np.maximum(
            rng.exponential(size=(batch, n_users)),
            np.finfo(float).tiny,
        )
        evaluated = _batch_ledger(stats, weights)
        slc = slice(position, position + batch)
        component_draws[slc] = evaluated["components"]
        gap_draws[slc] = evaluated["full_gap"]
        aggregate["weighted_mean"][slc] = evaluated["gap_mean"]
        aggregate["level_squared_error"][slc] = evaluated["level_error"]
        aggregate["shape_squared_error"][slc] = evaluated["shape_error"]
        aggregate["total_weighted_squared_error"][slc] = evaluated["total_error"]
        aggregate["component_weighted_mean"][slc] = evaluated["component_means"]
        aggregate["component_level_inner_product"][slc] = evaluated[
            "level_contributions"
        ]
        aggregate["component_shape_inner_product"][slc] = evaluated[
            "shape_contributions"
        ]
        aggregate["component_total_inner_product"][slc] = evaluated[
            "total_contributions"
        ]
        aggregate["component_level_share"][slc] = np.divide(
            evaluated["level_contributions"],
            evaluated["level_error"][:, None],
            out=np.full_like(evaluated["level_contributions"], np.nan),
            where=evaluated["level_error"][:, None] != 0,
        )
        aggregate["component_shape_share"][slc] = np.divide(
            evaluated["shape_contributions"],
            evaluated["shape_error"][:, None],
            out=np.full_like(evaluated["shape_contributions"], np.nan),
            where=evaluated["shape_error"][:, None] != 0,
        )
        aggregate["component_total_share"][slc] = np.divide(
            evaluated["total_contributions"],
            evaluated["total_error"][:, None],
            out=np.full_like(evaluated["total_contributions"], np.nan),
            where=evaluated["total_error"][:, None] != 0,
        )
        maximum_path_error = max(
            maximum_path_error,
            float(np.max(np.abs(evaluated["path_reconstruction_error"]))),
        )
        maximum_ledger_error = max(
            maximum_ledger_error,
            float(np.max(np.abs(evaluated["ledger_reconstruction_error"]))),
        )
        position += batch

    def ci(values: np.ndarray) -> list[float | None]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return [None, None]
        return [
            float(np.percentile(finite, 2.5)),
            float(np.percentile(finite, 97.5)),
        ]

    full_gap_ci = {
        "weighted_mean": ci(aggregate["weighted_mean"]),
        "level_squared_error": ci(aggregate["level_squared_error"]),
        "shape_squared_error": ci(aggregate["shape_squared_error"]),
        "total_weighted_squared_error": ci(
            aggregate["total_weighted_squared_error"]
        ),
        "weighted_rmse": ci(np.sqrt(aggregate["total_weighted_squared_error"])),
    }
    component_ci = {}
    for idx, name in enumerate(COMPONENTS):
        component_ci[name] = {
            "weighted_mean": ci(aggregate["component_weighted_mean"][:, idx]),
            "level_inner_product": ci(
                aggregate["component_level_inner_product"][:, idx]
            ),
            "shape_inner_product": ci(
                aggregate["component_shape_inner_product"][:, idx]
            ),
            "total_inner_product": ci(
                aggregate["component_total_inner_product"][:, idx]
            ),
            "level_share_signed": ci(aggregate["component_level_share"][:, idx]),
            "shape_share_signed": ci(aggregate["component_shape_share"][:, idx]),
            "total_share_signed": ci(aggregate["component_total_share"][:, idx]),
            "level_inner_product_fraction_positive": float(
                np.mean(aggregate["component_level_inner_product"][:, idx] > 0)
            ),
            "shape_inner_product_fraction_positive": float(
                np.mean(aggregate["component_shape_inner_product"][:, idx] > 0)
            ),
        }
    per_category_ci = {}
    for cat_idx, category in enumerate(stats["categories"]):
        per_category_ci[str(category)] = {
            "full_gap": ci(gap_draws[:, cat_idx]),
            **{
                name: ci(component_draws[:, idx, cat_idx])
                for idx, name in enumerate(COMPONENTS)
            },
        }
    return {
        "design": {
            "unit": "paired user cluster",
            "method": "positive exponential multiplier bootstrap",
            "same_user_weight_all_four_cells": True,
            "strictly_positive_weights_preserve_fixed_video_support": True,
            "random_full_shares_recomputed_each_draw": True,
            "interval": "2.5th and 97.5th percentile",
            "n_bootstrap": int(n_boot),
            "seed": int(seed),
            "chunk_size": int(chunk_size),
        },
        "ci95": {
            "full_gap": full_gap_ci,
            "components": component_ci,
            "per_category": per_category_ci,
        },
        "maximum_bootstrap_path_reconstruction_error": maximum_path_error,
        "maximum_bootstrap_ledger_reconstruction_error": maximum_ledger_error,
    }


def run_p9(
    *,
    n_boot: int = B_USERS,
    seed: int = SEED,
    chunk_size: int = CHUNK_SIZE,
    root: str | Path | None = None,
    category_path: str | Path | None = None,
) -> dict[str, object]:
    """Load KuaiRand, construct the canonical ledger, and bootstrap it."""
    random_full, standard_full, base_scope = load_kuairand_buckets(
        root,
        category_path=category_path,
        matched_period=True,
        tab=1,
        common_videos=False,
    )
    (
        random_full,
        standard_full,
        random_common,
        standard_common,
        scope,
    ) = canonical_ledger_scope(random_full, standard_full)
    table = category_error_components(
        random_full,
        standard_full,
        random_common,
        standard_common,
    )
    ledger = weighted_squared_error_ledger(
        table[list(COMPONENTS)],
        table["random_full_exposure_share"],
    )
    bootstrap = paired_positive_multiplier_bootstrap(
        random_full,
        standard_full,
        random_common,
        standard_common,
        n_boot=n_boot,
        seed=seed,
        chunk_size=chunk_size,
    )
    per_category = {
        str(category): {column: float(value) for column, value in row.items()}
        for category, row in table.iterrows()
    }
    maximum_path_error = float(table["reconstruction_error"].abs().max())
    return {
        "analysis": "P9 complete descriptive dashboard-error ledger",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "warning": ACCOUNTING_WARNING,
        "novelty_note": (
            "The contribution is an audit protocol and estimand organization; "
            "the additive path and inner-product reconstruction are not claimed "
            "as new mathematical identities."
        ),
        "path_definition": {
            "full_gap": "standard_full_mean - random_full_mean",
            "standard_support_entry_exit": (
                "standard_full_mean - standard_common_mean"
            ),
            "within_video_residual_plus_video_mix": (
                "standard_common_mean - random_common_mean, using the symmetric "
                "common-video decomposition"
            ),
            "random_support_entry_exit": "random_common_mean - random_full_mean",
            "support_term_note": (
                "each full-minus-common contrast includes the exclusive-video "
                "rows and the induced renormalization of retained video weights"
            ),
            "component_order": list(COMPONENTS),
        },
        "base_scope": base_scope,
        "ledger_scope": scope,
        "per_category": per_category,
        "maximum_point_path_reconstruction_error": maximum_path_error,
        "random_share_weighted_squared_error_ledger": ledger,
        "paired_positive_multiplier_bootstrap": bootstrap,
    }


def make_figure(result: Mapping[str, object], output_path: str | Path) -> None:
    """Plot the largest category paths and aggregate level/shape ledger."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial Unicode MS",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    table = pd.DataFrame.from_dict(result["per_category"], orient="index")
    n_largest = min(9, len(table))
    selected = list(table["full_gap"].abs().nlargest(n_largest).index)
    negative_path_mass = -table[list(COMPONENTS)].clip(upper=0).sum(axis=1)
    strongest_offset = negative_path_mass.idxmax()
    if strongest_offset not in selected:
        selected.append(strongest_offset)
    elif len(table) > len(selected):
        selected.append(table["full_gap"].abs().nlargest(len(selected) + 1).index[-1])
    table = table.loc[selected]
    table = table.sort_values("full_gap")
    bootstrap_categories = result["paired_positive_multiplier_bootstrap"]["ci95"][
        "per_category"
    ]
    colors = {
        "standard_support_entry_exit": "#4C78A8",
        "within_video_residual": "#F58518",
        "video_mix": "#54A24B",
        "random_support_entry_exit": "#B279A2",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.6, 5.5),
        gridspec_kw={"width_ratios": [1.55, 1]},
    )
    ax = axes[0]
    y = np.arange(len(table))
    positive_left = np.zeros(len(table))
    negative_left = np.zeros(len(table))
    for name in COMPONENTS:
        values = table[name].to_numpy(float)
        left = np.where(values >= 0, positive_left, negative_left)
        ax.barh(
            y,
            values,
            left=left,
            color=colors[name],
            label=DISPLAY_NAMES[name],
            height=0.72,
        )
        positive_left += np.where(values >= 0, values, 0.0)
        negative_left += np.where(values < 0, values, 0.0)
    gap = table["full_gap"].to_numpy(float)
    intervals = np.asarray(
        [bootstrap_categories[str(category)]["full_gap"] for category in table.index]
    )
    ax.errorbar(
        gap,
        y,
        xerr=np.vstack([gap - intervals[:, 0], intervals[:, 1] - gap]),
        fmt="D",
        markersize=3.5,
        color="black",
        ecolor="0.25",
        elinewidth=0.8,
        capsize=2,
        label="Full gap (95% CI)",
        zorder=5,
    )
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_yticks(y, [english_category_label(category) for category in table.index])
    ax.set_xlabel("Watch-ratio gap / path component")
    ax.set_title("(a) Largest full-support gaps and strongest offset")
    ax.grid(axis="x", alpha=0.2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))

    ax = axes[1]
    ledger = result["random_share_weighted_squared_error_ledger"]
    raw_totals = np.asarray(
        [
            ledger["full_gap"]["level_squared_error"],
            ledger["full_gap"]["shape_squared_error"],
        ]
    )
    totals = np.ones(2)
    x = np.arange(2)
    positive_bottom = np.zeros(2)
    negative_bottom = np.zeros(2)
    for name in COMPONENTS:
        component = ledger["components"][name]
        values = np.asarray(
            [component["level_inner_product"], component["shape_inner_product"]]
        )
        values = values / raw_totals
        bottom = np.where(values >= 0, positive_bottom, negative_bottom)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.62,
            color=colors[name],
            label=DISPLAY_NAMES[name],
        )
        positive_bottom += np.where(values >= 0, values, 0.0)
        negative_bottom += np.where(values < 0, values, 0.0)
    ax.scatter(x, totals, marker="D", s=28, color="black", zorder=5, label="Total")
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set_xticks(x, ["Level", "Centered shape"])
    ax.set_ylabel("Signed share of squared-error total")
    ax.set_title("(b) Signed inner-product ledger")
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(min(-0.05, float(negative_bottom.min()) * 1.15), 1.14)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    for position, raw_total in enumerate(raw_totals):
        ax.text(
            position,
            1.035,
            f"total={raw_total:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="0.25",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.025,
        "Canonical paired users are fixed; signed terms may offset. Accounting is descriptive, not causal attribution.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.82, wspace=0.34)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--bootstrap", type=int, default=B_USERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--kuairand-root", type=Path, default=None)
    parser.add_argument("--category-path", type=Path, default=None)
    args = parser.parse_args()
    result = run_p9(
        n_boot=args.bootstrap,
        seed=args.seed,
        chunk_size=args.chunk_size,
        root=args.kuairand_root,
        category_path=args.category_path,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "p9_error_ledger.json"
    figure_path = RESULTS_DIR / "fig9_error_ledger.png"
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
