from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from p12_known_truth_benchmark import (  # noqa: E402
    SCENARIOS,
    allocate_items,
    dashboard_metrics,
    run_benchmark,
    simulate_replication,
)


def test_allocate_items_preserves_total_and_floor():
    shares = np.array([0.6, 0.3, 0.1])
    counts = allocate_items(shares, n_items=20, min_items=2)
    assert counts.sum() == 20
    assert np.all(counts >= 2)
    assert counts[0] > counts[1] > counts[2]


def test_dashboard_centered_metrics_are_scalar_shift_invariant():
    truth = np.array([0.1, 0.3, 0.2, 0.5])
    estimate = np.array([0.2, 0.25, 0.4, 0.45])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    original = dashboard_metrics(estimate, truth, weights, top_k=2)
    shifted = dashboard_metrics(estimate + 7.0, truth, weights, top_k=2)
    assert np.isclose(original["centered_mae"], shifted["centered_mae"])
    assert np.isclose(original["rank_corr"], shifted["rank_corr"])
    assert np.isclose(original["top10_overlap"], shifted["top10_overlap"])
    assert np.isclose(original["top10_regret"], shifted["top10_regret"])
    assert original["topk_effective"] == 2
    assert shifted["absolute_mae"] > original["absolute_mae"]


def test_one_replication_is_finite_and_oracle_weights_are_defined():
    shares = np.array([0.4, 0.3, 0.2, 0.1])
    counts = allocate_items(shares, n_items=24, min_items=3)
    result = simulate_replication(
        np.random.default_rng(7),
        shares,
        counts,
        SCENARIOS[-1],
        n_users=80,
        n_exposures_per_user=30,
    )
    assert set(result["estimates"]) == {
        "naive",
        "scalar_level_shift",
        "cell_equal_demean",
        "row_weighted_demean",
        "oracle_hajek",
        "oracle_dm_hajek",
    }
    for metrics in result["metrics"].values():
        assert all(np.isfinite(list(metrics.values())))
    assert 0 < result["diagnostics"]["importance_weight_ess_fraction"] <= 1


def test_tiny_benchmark_schema_uses_all_scenarios(tmp_path):
    # A minimal P8-shaped calibration file makes this an isolated unit test.
    calibration = tmp_path / "p8.json"
    calibration.write_text(
        '{"item_gap_accounting":{"per_category":{'
        '"a":{"random_exposure_share":0.4},'
        '"b":{"random_exposure_share":0.3},'
        '"c":{"random_exposure_share":0.2},'
        '"d":{"random_exposure_share":0.1}'
        '}}}',
        encoding="utf-8",
    )
    payload = run_benchmark(
        n_replications=2,
        n_users=50,
        n_exposures_per_user=25,
        n_items=24,
        seed=19,
        p8_path=calibration,
        include_exposure_scale=False,
    )
    assert payload["n_replications"] == 2
    assert payload["n_categories"] == 4
    assert set(payload["scenarios"]) == {scenario.label for scenario in SCENARIOS}
    cell = payload["scenarios"]["target_like"]["estimators"]["naive"]
    assert set(cell) == {
        "absolute_mae",
        "centered_mae",
        "max_abs_error",
        "rank_corr",
        "top10_overlap",
        "top10_regret",
        "topk_effective",
    }
