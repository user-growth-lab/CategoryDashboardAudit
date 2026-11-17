"""Known-truth stress test for category-dashboard audits.

The real KuaiRand audit has a randomized reference but does not reveal the
counterfactual dashboard under arbitrary serving policies.  This module builds
an empirically calibrated simulation whose target category shares equal the
paired KuaiRand random-insertion shares.  Potential outcome probabilities are
known for every synthetic user--video pair, while serving can select users,
user--category affinities, and within-category videos with controlled overlap.

The benchmark is deliberately a failure-boundary exercise, not evidence about
the size of any real platform effect.  It compares descriptive diagnostics
against the known target and includes a self-normalized oracle-propensity
baseline.  The latter is unavailable in the released KuaiRand logs and is used
only to distinguish identification from finite-overlap variance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
P8_PATH = RESULTS / "p8_paired_audit.json"
OUTPUT = RESULTS / "p12_known_truth_benchmark.json"
FIGURE = RESULTS / "fig12_known_truth_benchmark.png"

SEED = 20260812
N_REPLICATIONS = 100
N_USERS = 1_000
N_EXPOSURES_PER_USER = 60
N_ITEMS = 760
MIN_ITEMS_PER_CATEGORY = 4


@dataclass(frozen=True)
class Scenario:
    """One serving/outcome stress condition."""

    label: str
    description: str
    user_level_selection: float
    user_category_selection: float
    item_selection: float
    outcome_interaction_scale: float
    uniform_overlap_floor: float


SCENARIOS = (
    Scenario(
        "target_like",
        "Serving equals the target user and within-category video mixture.",
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ),
    Scenario(
        "user_composition",
        "Category exposure selects on a user-level outcome term that is "
        "additive on the logit scale.",
        1.25,
        0.0,
        0.0,
        0.0,
        0.10,
    ),
    Scenario(
        "user_category_interaction",
        "Serving selects user-category affinities that also enter outcomes.",
        0.60,
        1.10,
        0.0,
        0.045,
        0.10,
    ),
    Scenario(
        "within_category_item_mix",
        "Serving favors high-outcome videos within each category.",
        0.0,
        0.0,
        1.50,
        0.0,
        0.10,
    ),
    Scenario(
        "combined_weak_overlap",
        "User, affinity, and video selection act together with weak overlap.",
        1.00,
        1.00,
        1.35,
        0.045,
        0.02,
    ),
)

ESTIMATOR_ORDER = (
    "naive",
    "scalar_level_shift",
    "cell_equal_demean",
    "row_weighted_demean",
    "oracle_hajek",
    "oracle_dm_hajek",
)

ORACLE_ANCHORED_ESTIMATORS = (
    "scalar_level_shift",
    "cell_equal_demean",
    "row_weighted_demean",
)

METRIC_ORDER = (
    "absolute_mae",
    "centered_mae",
    "max_abs_error",
    "rank_corr",
    "top10_overlap",
    "top10_regret",
)

SUMMARY_METRICS = (*METRIC_ORDER, "topk_effective")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def load_empirical_category_shares(
    path: Path = P8_PATH,
) -> tuple[list[str], np.ndarray, str]:
    """Read the fixed paired-random category shares used for calibration."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cells = payload["item_gap_accounting"]["per_category"]
    names = sorted(cells)
    shares = np.asarray(
        [cells[name]["random_exposure_share"] for name in names], dtype=float
    )
    if len(names) < 2 or not np.isfinite(shares).all() or (shares <= 0).any():
        raise ValueError("P8 category shares must be finite and strictly positive")
    shares /= shares.sum()
    return names, shares, _sha256(path)


def allocate_items(
    shares: np.ndarray,
    n_items: int = N_ITEMS,
    min_items: int = MIN_ITEMS_PER_CATEGORY,
) -> np.ndarray:
    """Largest-remainder allocation with a positive per-category floor."""
    shares = np.asarray(shares, dtype=float)
    if shares.ndim != 1 or len(shares) < 2:
        raise ValueError("shares must be a one-dimensional vector")
    if not np.isfinite(shares).all() or (shares <= 0).any():
        raise ValueError("shares must be finite and strictly positive")
    if n_items < min_items * len(shares):
        raise ValueError("n_items is too small for the requested category floor")
    normalized = shares / shares.sum()
    remaining = n_items - min_items * len(shares)
    raw = normalized * remaining
    counts = np.floor(raw).astype(int) + min_items
    for index in np.argsort(-(raw - np.floor(raw)))[: n_items - counts.sum()]:
        counts[index] += 1
    if counts.sum() != n_items or (counts < min_items).any():
        raise AssertionError("item allocation invariant failed")
    return counts


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation without a SciPy dependency in this module."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank vectors must have the same length >= 2")
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def dashboard_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    target_shares: np.ndarray,
    *,
    top_k: int = 10,
) -> dict[str, float]:
    """Evaluate level, shape, rank, and plug-in decision discrepancy."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    weights = np.asarray(target_shares, dtype=float)
    if estimate.shape != truth.shape or estimate.shape != weights.shape:
        raise ValueError("estimate, truth, and target_shares must align")
    if not np.isfinite(estimate).all() or not np.isfinite(truth).all():
        raise ValueError("dashboard vectors must be finite")
    weights = weights / weights.sum()
    error = estimate - truth
    centered_error = error - np.dot(weights, error)
    top_k = min(int(top_k), len(truth))
    truth_top = np.argsort(-truth, kind="mergesort")[:top_k]
    estimate_top = np.argsort(-estimate, kind="mergesort")[:top_k]
    overlap = len(set(truth_top).intersection(estimate_top)) / top_k
    regret = truth[truth_top].mean() - truth[estimate_top].mean()
    return {
        "absolute_mae": float(np.mean(np.abs(error))),
        "centered_mae": float(np.mean(np.abs(centered_error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "rank_corr": _rank_corr(estimate, truth),
        "top10_overlap": float(overlap),
        "top10_regret": float(max(regret, 0.0)),
        "topk_effective": int(top_k),
    }


def _draw_exposures(
    rng: np.random.Generator,
    category_probabilities: np.ndarray,
    item_probabilities: list[np.ndarray],
    n_exposures: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a fixed exposure count for every user."""
    n_users, n_categories = category_probabilities.shape
    user_ids = np.repeat(np.arange(n_users, dtype=int), n_exposures)
    categories = np.empty(n_users * n_exposures, dtype=int)
    cursor = 0
    for user in range(n_users):
        counts = rng.multinomial(n_exposures, category_probabilities[user])
        categories[cursor : cursor + n_exposures] = np.repeat(
            np.arange(n_categories, dtype=int), counts
        )
        cursor += n_exposures
    local_items = np.empty_like(categories)
    for category in range(n_categories):
        mask = categories == category
        local_items[mask] = rng.choice(
            len(item_probabilities[category]),
            size=int(mask.sum()),
            replace=True,
            p=item_probabilities[category],
        )
    return user_ids, categories, local_items


def estimate_dashboards(
    user_ids: np.ndarray,
    categories: np.ndarray,
    local_items: np.ndarray,
    outcomes: np.ndarray,
    category_probabilities: np.ndarray,
    item_probabilities: list[np.ndarray],
    target_shares: np.ndarray,
    truth: np.ndarray,
    outcome_probability: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compute audit diagnostics and an oracle self-normalized IPW baseline."""
    n_users, n_categories = category_probabilities.shape
    n_rows = len(outcomes)
    if not (
        len(user_ids) == len(categories) == len(local_items) == n_rows
    ):
        raise ValueError("exposure arrays must align")
    category_counts = np.bincount(categories, minlength=n_categories).astype(float)
    category_sums = np.bincount(
        categories, weights=outcomes, minlength=n_categories
    )
    if (category_counts == 0).any():
        raise ValueError("every category must be observed")
    naive = category_sums / category_counts

    user_counts = np.bincount(user_ids, minlength=n_users).astype(float)
    user_sums = np.bincount(user_ids, weights=outcomes, minlength=n_users)
    if (user_counts == 0).any():
        raise ValueError("every synthetic user must be observed")
    user_means = user_sums / user_counts

    flat = user_ids * n_categories + categories
    cell_counts = np.bincount(
        flat, minlength=n_users * n_categories
    ).reshape(n_users, n_categories)
    cell_sums = np.bincount(
        flat, weights=outcomes, minlength=n_users * n_categories
    ).reshape(n_users, n_categories)
    observed = cell_counts > 0
    cell_means = np.divide(
        cell_sums,
        cell_counts,
        out=np.zeros_like(cell_sums, dtype=float),
        where=observed,
    )
    cell_residual = cell_means - user_means[:, None]
    ce_residual = np.divide(
        (cell_residual * observed).sum(axis=0),
        observed.sum(axis=0),
    )
    row_residual = outcomes - user_means[user_ids]
    rw_residual = np.bincount(
        categories, weights=row_residual, minlength=n_categories
    ) / category_counts

    target_level = float(np.dot(target_shares, truth))
    scalar_level_shift = naive - np.dot(target_shares, naive) + target_level
    cell_equal = ce_residual + target_level
    row_weighted = rw_residual + target_level

    behavior_probability = category_probabilities[user_ids, categories].copy()
    within_target_probability = np.empty(n_rows, dtype=float)
    for category, probabilities in enumerate(item_probabilities):
        mask = categories == category
        behavior_probability[mask] *= probabilities[local_items[mask]]
        within_target_probability[mask] = 1.0 / len(probabilities)
    target_probability = target_shares[categories] * within_target_probability
    importance_weight = target_probability / behavior_probability
    weighted_sum = np.bincount(
        categories,
        weights=importance_weight * outcomes,
        minlength=n_categories,
    )
    weight_sum = np.bincount(
        categories, weights=importance_weight, minlength=n_categories
    )
    oracle_hajek = weighted_sum / weight_sum

    oracle_dm_hajek = None
    if outcome_probability is not None:
        outcome_probability = np.asarray(outcome_probability, dtype=float)
        if outcome_probability.shape != outcomes.shape:
            raise ValueError("outcome_probability must align with outcomes")
        if not np.isfinite(outcome_probability).all():
            raise ValueError("outcome_probability must be finite")
        residual_sum = np.bincount(
            categories,
            weights=importance_weight * (outcomes - outcome_probability),
            minlength=n_categories,
        )
        # The simulation exposes the complete conditional outcome surface.
        # Adding its exact target mean leaves only weighted Bernoulli noise.
        oracle_dm_hajek = truth + residual_sum / weight_sum

    estimators = {
        "naive": naive,
        "scalar_level_shift": scalar_level_shift,
        "cell_equal_demean": cell_equal,
        "row_weighted_demean": row_weighted,
        "oracle_hajek": oracle_hajek,
    }
    if oracle_dm_hajek is not None:
        estimators["oracle_dm_hajek"] = oracle_dm_hajek
    diagnostics = {
        "observed_user_category_cell_fraction": float(observed.mean()),
        "importance_weight_ess_fraction": float(
            importance_weight.sum() ** 2
            / (np.square(importance_weight).sum() * n_rows)
        ),
        "importance_weight_p99": float(np.quantile(importance_weight, 0.99)),
        "importance_weight_max": float(np.max(importance_weight)),
    }
    per_category_ess = []
    for category in range(n_categories):
        category_weights = importance_weight[categories == category]
        per_category_ess.append(
            category_weights.sum() ** 2 / np.square(category_weights).sum()
        )
    diagnostics.update(
        {
            "importance_weight_category_ess_min": float(
                np.min(per_category_ess)
            ),
            "importance_weight_category_ess_median": float(
                np.median(per_category_ess)
            ),
        }
    )
    return estimators, diagnostics


def simulate_replication(
    rng: np.random.Generator,
    target_shares: np.ndarray,
    item_counts: np.ndarray,
    scenario: Scenario,
    *,
    n_users: int = N_USERS,
    n_exposures_per_user: int = N_EXPOSURES_PER_USER,
    mean_scale_additive: bool = False,
) -> dict[str, Any]:
    """Simulate one complete known-truth dashboard experiment."""
    target_shares = np.asarray(target_shares, dtype=float)
    target_shares /= target_shares.sum()
    n_categories = len(target_shares)
    if len(item_counts) != n_categories:
        raise ValueError("item_counts must align with target_shares")

    user_level = np.clip(rng.normal(size=n_users), -2.5, 2.5)
    user_level -= user_level.mean()
    category_effect_z = rng.normal(size=n_categories)
    category_effect_z -= np.dot(target_shares, category_effect_z)
    category_effect = 0.055 * category_effect_z
    category_audience_loading = rng.normal(size=n_categories)
    category_audience_loading -= np.dot(
        target_shares, category_audience_loading
    )
    category_audience_loading /= max(
        np.std(category_audience_loading), 1e-8
    )

    affinity_z = rng.normal(size=(n_users, n_categories))
    affinity_z -= affinity_z.mean(axis=0, keepdims=True)
    affinity_effect = scenario.outcome_interaction_scale * affinity_z

    item_effects: list[np.ndarray] = []
    item_probabilities: list[np.ndarray] = []
    for count in item_counts:
        quality_z = rng.normal(size=int(count))
        quality_z -= quality_z.mean()
        quality_effect = 0.045 * quality_z
        item_effects.append(quality_effect)
        selected = _softmax(scenario.item_selection * quality_z)
        uniform = np.full(int(count), 1.0 / count)
        item_probabilities.append(
            (1.0 - scenario.uniform_overlap_floor) * selected
            + scenario.uniform_overlap_floor * uniform
        )

    logits = (
        np.log(target_shares)[None, :]
        + scenario.user_level_selection
        * user_level[:, None]
        * category_audience_loading[None, :]
        + scenario.user_category_selection * affinity_z
    )
    selected_categories = _softmax(logits, axis=1)
    category_probabilities = (
        (1.0 - scenario.uniform_overlap_floor) * selected_categories
        + scenario.uniform_overlap_floor * target_shares[None, :]
    )

    user_ids, categories, local_items = _draw_exposures(
        rng,
        category_probabilities,
        item_probabilities,
        n_exposures_per_user,
    )
    if mean_scale_additive:
        if (
            scenario.outcome_interaction_scale != 0
            or scenario.item_selection != 0
        ):
            raise ValueError(
                "mean_scale_additive is restricted to the additive positive control"
            )
        probability = (
            0.45
            + 0.06 * user_level[user_ids]
            + category_effect[categories]
        )
        if probability.min() <= 0.0 or probability.max() >= 1.0:
            raise AssertionError("mean-scale positive control left (0,1)")
    else:
        # Work on the logit scale so the benchmark remains valid for every seed
        # and for every potential pair, including unobserved pairs.  Item effects
        # below are also logits; target truth is enumerated over the complete
        # finite user--video population rather than inferred from sampled rows.
        linear_predictor = (
            -0.20
            + 0.24 * user_level[user_ids]
            + 4.0 * category_effect[categories]
            + 4.0 * affinity_effect[user_ids, categories]
        )
        for category, effects in enumerate(item_effects):
            mask = categories == category
            linear_predictor[mask] += 4.0 * effects[local_items[mask]]
        probability = 1.0 / (1.0 + np.exp(-linear_predictor))
    outcomes = rng.binomial(1, probability).astype(float)

    truth = np.empty(n_categories, dtype=float)
    for category, effects in enumerate(item_effects):
        if mean_scale_additive:
            truth[category] = 0.45 + category_effect[category]
        else:
            complete_logits = (
                -0.20
                + 0.24 * user_level[:, None]
                + 4.0 * category_effect[category]
                + 4.0 * affinity_effect[:, category, None]
                + 4.0 * effects[None, :]
            )
            truth[category] = np.mean(
                1.0 / (1.0 + np.exp(-complete_logits))
            )

    estimators, diagnostics = estimate_dashboards(
        user_ids,
        categories,
        local_items,
        outcomes,
        category_probabilities,
        item_probabilities,
        target_shares,
        truth,
        probability,
    )
    return {
        "truth": truth,
        "estimates": estimators,
        "metrics": {
            name: dashboard_metrics(value, truth, target_shares)
            for name, value in estimators.items()
        },
        "diagnostics": diagnostics,
    }


def exposure_scale_sensitivity(
    target_shares: np.ndarray,
    item_counts: np.ndarray,
    *,
    exposures_per_user: tuple[int, ...] = (30, 60, 150, 300),
    n_replications: int = 20,
    n_users: int = N_USERS,
    seed: int = SEED + 1,
) -> dict[str, Any]:
    """Separate weak-overlap identification from finite-sample precision.

    The finite population is still redrawn between replications, as in the
    main stress test.  Only the number of realized rows changes across cells.
    """
    scenario = next(
        scenario for scenario in SCENARIOS if scenario.label == "combined_weak_overlap"
    )
    children = np.random.SeedSequence(seed).spawn(
        len(exposures_per_user) * n_replications
    )
    cursor = 0
    cells: dict[str, Any] = {}
    for exposure_count in exposures_per_user:
        records = []
        for _ in range(n_replications):
            records.append(
                simulate_replication(
                    np.random.default_rng(children[cursor]),
                    target_shares,
                    item_counts,
                    scenario,
                    n_users=n_users,
                    n_exposures_per_user=exposure_count,
                )
            )
            cursor += 1
        cells[str(exposure_count)] = {
            "n_rows": int(n_users * exposure_count),
            "importance_weight_ess_fraction": _summarize(
                [
                    record["diagnostics"]["importance_weight_ess_fraction"]
                    for record in records
                ]
            ),
            "centered_mae": {
                estimator: _summarize(
                    [
                        record["metrics"][estimator]["centered_mae"]
                        for record in records
                    ]
                )
                for estimator in (
                    "naive",
                    "cell_equal_demean",
                    "row_weighted_demean",
                    "oracle_hajek",
                    "oracle_dm_hajek",
                )
            },
        }
    return {
        "scenario": scenario.label,
        "purpose": (
            "precision diagnostic: true propensity weighting is identified in "
            "the simulation but unstable when effective overlap is small"
        ),
        "n_replications_per_cell": int(n_replications),
        "n_users": int(n_users),
        "exposures_per_user": [int(value) for value in exposures_per_user],
        "cells": cells,
    }


def _summarize(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "interval_95_across_replications": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "monte_carlo_standard_error_of_mean": float(
            array.std(ddof=1) / np.sqrt(len(array))
        )
        if len(array) > 1
        else 0.0,
    }


def additive_positive_control(
    target_shares: np.ndarray,
    item_counts: np.ndarray,
    *,
    n_replications: int,
    n_users: int,
    n_exposures_per_user: int,
    seed: int,
) -> dict[str, Any]:
    """Verify demeaning under its exact outcome-mean additivity condition."""
    scenario = next(
        scenario for scenario in SCENARIOS if scenario.label == "user_composition"
    )
    records = [
        simulate_replication(
            np.random.default_rng(child),
            target_shares,
            item_counts,
            scenario,
            n_users=n_users,
            n_exposures_per_user=n_exposures_per_user,
            mean_scale_additive=True,
        )
        for child in np.random.SeedSequence(seed).spawn(n_replications)
    ]
    return {
        "definition": (
            "Bernoulli mean is exactly user-level plus category-level additive; "
            "serving selects the user level by category. This is a positive "
            "control for the demeaning assumption, not a realistic universal DGP."
        ),
        "n_replications": int(n_replications),
        "estimators": {
            estimator: {
                metric: _summarize(
                    [record["metrics"][estimator][metric] for record in records]
                )
                for metric in METRIC_ORDER
            }
            for estimator in ESTIMATOR_ORDER
        },
    }


def run_benchmark(
    *,
    n_replications: int = N_REPLICATIONS,
    n_users: int = N_USERS,
    n_exposures_per_user: int = N_EXPOSURES_PER_USER,
    n_items: int = N_ITEMS,
    seed: int = SEED,
    p8_path: Path = P8_PATH,
    include_exposure_scale: bool = True,
    exposure_scale_replications: int = 20,
) -> dict[str, Any]:
    """Run all scenarios with independent, reproducible Monte Carlo seeds."""
    if min(n_replications, n_users, n_exposures_per_user) < 1:
        raise ValueError("replications, users, and exposures must be positive")
    category_names, target_shares, p8_sha = load_empirical_category_shares(p8_path)
    item_counts = allocate_items(target_shares, n_items=n_items)
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(len(SCENARIOS) * n_replications)
    cursor = 0
    scenario_results: dict[str, Any] = {}
    for scenario in SCENARIOS:
        records = []
        for _ in range(n_replications):
            records.append(
                simulate_replication(
                    np.random.default_rng(children[cursor]),
                    target_shares,
                    item_counts,
                    scenario,
                    n_users=n_users,
                    n_exposures_per_user=n_exposures_per_user,
                )
            )
            cursor += 1

        estimator_summary: dict[str, Any] = {}
        for estimator in ESTIMATOR_ORDER:
            estimator_summary[estimator] = {
                metric: _summarize(
                    [record["metrics"][estimator][metric] for record in records]
                )
                for metric in SUMMARY_METRICS
            }
        diagnostics = {
            key: _summarize([record["diagnostics"][key] for record in records])
            for key in records[0]["diagnostics"]
        }
        scenario_results[scenario.label] = {
            "definition": asdict(scenario),
            "estimators": estimator_summary,
            "diagnostics": diagnostics,
        }

    payload = {
        "analysis": "P12 empirically calibrated known-truth stress test",
        "interpretation_boundary": (
            "Synthetic failure-boundary evidence calibrated only to the paired "
            "KuaiRand random category shares; it does not estimate a real "
            "platform effect or validate an estimator universally."
        ),
        "replication_interval_definition": (
            "Each replication redraws the finite outcome/serving world and "
            "the realized logs. Reported 2.5--97.5 percentiles are across-world "
            "stress-test prediction intervals, not sampling confidence "
            "intervals for one fixed platform. MCSE describes the reported mean."
        ),
        "target": (
            "Equal synthetic users and uniform videos within category; category "
            "shares equal paired-random KuaiRand exposure shares."
        ),
        "outcome": "Bernoulli engagement with known user-video probabilities",
        "oracle_warning": (
            "oracle_hajek uses the true simulated serving probabilities, which "
            "are unavailable in the released KuaiRand logs."
        ),
        "oracle_anchor_warning": (
            "scalar_level_shift, cell_equal_demean, and row_weighted_demean "
            "borrow the known target grand mean. They isolate shape behavior "
            "and are not deployable without exploration data for that anchor."
        ),
        "seed": int(seed),
        "n_replications": int(n_replications),
        "n_users_per_replication": int(n_users),
        "n_exposures_per_user": int(n_exposures_per_user),
        "n_categories": len(category_names),
        "n_items": int(item_counts.sum()),
        "minimum_items_per_category": int(item_counts.min()),
        "category_names": category_names,
        "target_category_shares": {
            name: float(share) for name, share in zip(category_names, target_shares)
        },
        "items_per_category": {
            name: int(count) for name, count in zip(category_names, item_counts)
        },
        "calibration_source": {
            "file": p8_path.name,
            "sha256": p8_sha,
            "field": "item_gap_accounting.per_category.*.random_exposure_share",
        },
        "estimator_definitions": {
            "naive": "standard row-weighted category mean",
            "scalar_level_shift": (
                "naive centered shape plus the known target grand mean"
            ),
            "cell_equal_demean": (
                "observed user-category cell-equal user demeaning plus the "
                "known target grand mean"
            ),
            "row_weighted_demean": (
                "row-weighted user residual plus the known target grand mean"
            ),
            "oracle_hajek": (
                "self-normalized target/behavior importance weighting with "
                "known simulated propensities"
            ),
            "oracle_dm_hajek": (
                "oracle residualized H\u00e1jek control variate around the exact "
                "simulated conditional-outcome surface; not a fitted DR learner"
            ),
        },
        "scenarios": scenario_results,
        "mean_scale_additive_positive_control": additive_positive_control(
            target_shares,
            item_counts,
            n_replications=n_replications,
            n_users=n_users,
            n_exposures_per_user=n_exposures_per_user,
            seed=seed + 2,
        ),
    }
    if include_exposure_scale:
        payload["weak_overlap_exposure_scale_sensitivity"] = (
            exposure_scale_sensitivity(
                target_shares,
                item_counts,
                n_replications=exposure_scale_replications,
                n_users=n_users,
                seed=seed + 1,
            )
        )
    return payload


def make_figure(payload: dict[str, Any], path: Path = FIGURE) -> None:
    """Plot known-truth centered error and oracle-overlap cost."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_names = list(payload["scenarios"])
    display_scenarios = [name.replace("_", "\n") for name in scenario_names]
    colors = {
        "naive": "#9b2c2c",
        "scalar_level_shift": "#dd6b20",
        "cell_equal_demean": "#2b6cb0",
        "row_weighted_demean": "#319795",
        "oracle_hajek": "#4a5568",
        "oracle_dm_hajek": "#805ad5",
    }
    display_estimators = {
        "naive": "Naive",
        "scalar_level_shift": "Level shift",
        "cell_equal_demean": "CE demean",
        "row_weighted_demean": "RW demean",
        "oracle_hajek": "Oracle H\u00e1jek",
        "oracle_dm_hajek": "Oracle DM+H\u00e1jek",
    }
    x = np.arange(len(scenario_names))
    width = 0.13
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
    for offset, estimator in enumerate(ESTIMATOR_ORDER):
        means = [
            payload["scenarios"][scenario]["estimators"][estimator][
                "centered_mae"
            ]["mean"]
            for scenario in scenario_names
        ]
        intervals = [
            payload["scenarios"][scenario]["estimators"][estimator][
                "centered_mae"
            ]["interval_95_across_replications"]
            for scenario in scenario_names
        ]
        lower = np.asarray(means) - np.asarray([interval[0] for interval in intervals])
        upper = np.asarray([interval[1] for interval in intervals]) - np.asarray(means)
        position = x + (offset - 2.5) * width
        axes[0].bar(
            position,
            means,
            width,
            color=colors[estimator],
            label=display_estimators[estimator],
            alpha=0.9,
        )
        axes[0].errorbar(
            position,
            means,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#1a202c",
            elinewidth=0.7,
            capsize=1.5,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_scenarios, fontsize=7)
    axes[0].set_ylabel("Centered MAE vs known truth")
    axes[0].set_title("(a) Failure boundaries across serving mechanisms")
    axes[0].legend(ncol=3, fontsize=7, frameon=False)

    ess = [
        payload["scenarios"][scenario]["diagnostics"][
            "importance_weight_ess_fraction"
        ]["mean"]
        for scenario in scenario_names
    ]
    oracle_error = [
        payload["scenarios"][scenario]["estimators"]["oracle_hajek"][
            "centered_mae"
        ]["mean"]
        for scenario in scenario_names
    ]
    axes[1].scatter(ess, oracle_error, color="#4a5568", s=42)
    for index, label in enumerate(scenario_names):
        axes[1].annotate(
            label.replace("_", " "),
            (ess[index], oracle_error[index]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set_xlabel("Oracle importance-weight ESS / rows")
    axes[1].set_ylabel("Oracle H\u00e1jek centered MAE")
    axes[1].set_title("(b) Identification is not finite-overlap precision")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=N_REPLICATIONS)
    parser.add_argument("--users", type=int, default=N_USERS)
    parser.add_argument("--exposures-per-user", type=int, default=N_EXPOSURES_PER_USER)
    parser.add_argument("--items", type=int, default=N_ITEMS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(
        n_replications=args.replications,
        n_users=args.users,
        n_exposures_per_user=args.exposures_per_user,
        n_items=args.items,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_figure(payload, args.figure)
    print(f"saved {args.output}")
    print(f"saved {args.figure}")


if __name__ == "__main__":
    main()
