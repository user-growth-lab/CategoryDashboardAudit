from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from p11_decision_audit import (  # noqa: E402
    decision_aware_audit,
    decision_metrics,
    deterministic_topk,
    inclusion_classification,
    max_t_simultaneous_band,
    max_t_simultaneous_contrast_intervals,
    ranking_stability_certificate,
)


def test_deterministic_topk_uses_category_label_to_break_exact_ties():
    values = pd.Series({"b": 1.0, "a": 1.0, "c": 0.5})
    assert deterministic_topk(values, 1) == ["a"]
    assert deterministic_topk(values, 2) == ["a", "b"]


def test_reference_regret_has_declared_direction_and_tie_rule():
    reference = pd.Series({"A": 1.0, "B": 0.8, "C": 0.1})
    estimate = pd.Series({"A": 0.9, "B": 1.0, "C": 0.2})
    result = decision_metrics(estimate, reference, k=1)

    assert result["selected_categories"] == ["B"]
    assert result["reference_selected_categories"] == ["A"]
    assert result["reference_regret"] == pytest.approx(0.2)
    assert result["topk_overlap"] == pytest.approx(0.0)


def test_inclusion_probability_classification_uses_inclusive_boundaries():
    assert inclusion_classification(0.95) == "stable_in"
    assert inclusion_classification(0.05) == "stable_out"
    assert inclusion_classification(0.50) == "uncertain"


def test_max_t_band_is_simultaneous_over_fixed_category_family():
    point = np.array([0.1, -0.2, 0.0])
    draws = np.array(
        [
            [0.11, -0.18, 0.0],
            [0.08, -0.22, 0.0],
            [0.12, -0.19, 0.0],
            [0.09, -0.23, 0.0],
        ]
    )
    result = max_t_simultaneous_band(point, draws, ["A", "B", "C"])

    assert result["categories_fixed_across_draws"] is True
    assert result["family_size"] == 3
    assert result["critical_value"] >= 0
    for category in ("A", "B", "C"):
        cell = result["per_category"][category]
        assert cell["simultaneous_lower"] <= cell["point_signed_error"]
        assert cell["simultaneous_upper"] >= cell["point_signed_error"]
    assert result["per_category"]["C"]["simultaneous_lower"] == pytest.approx(0)
    assert result["per_category"]["C"]["simultaneous_upper"] == pytest.approx(0)


def test_max_t_contrast_intervals_use_one_common_familywise_critical_value():
    point = np.array([-1.0, 0.10, -0.50])
    deviations = np.array(
        [
            [0.04, -0.03, 0.05],
            [-0.05, 0.02, -0.04],
            [0.03, 0.04, -0.02],
            [-0.02, -0.05, 0.03],
            [0.01, 0.01, -0.01],
        ]
    )
    draws = point[None, :] + deviations
    result = max_t_simultaneous_contrast_intervals(
        point,
        draws,
        ["ce_minus_naive_k5", "rw_minus_naive_k5", "ce_minus_naive_k10"],
    )

    assert result["family_size"] == 3
    assert result["n_shared_bootstrap_draws"] == 5
    assert result["same_bootstrap_draws_for_all_contrasts"] is True
    assert result["critical_value"] >= 0
    strongly_negative = result["per_contrast"]["ce_minus_naive_k5"]
    assert strongly_negative["excludes_zero"] is True
    assert strongly_negative["supported_direction"] == "lower regret than naive"
    for cell in result["per_contrast"].values():
        lower, upper = cell["simultaneous_ci95"]
        assert lower <= cell["point_regret_contrast"] <= upper


def test_ranking_margin_condition_is_sufficient_for_topk_stability():
    reference = pd.Series({"A": 1.0, "B": 0.8, "C": 0.3, "D": 0.1})
    shares = pd.Series(0.25, index=reference.index)
    close = pd.Series({"A": 1.01, "B": 0.79, "C": 0.31, "D": 0.09})
    certificate = ranking_stability_certificate(close, reference, shares, k=2)

    assert certificate["reference_boundary_gap"] == pytest.approx(0.5)
    assert certificate["condition_satisfied"] is True
    assert certificate["observed_topk_set_exact"] is True
    assert certificate["condition_is_sufficient_not_necessary"] is True


def _paired_fixture():
    random_rows = []
    standard_rows = []
    categories = ["A", "B", "C", "D"]
    reference_base = np.array([0.90, 0.68, 0.42, 0.18])
    standard_base = np.array([0.70, 0.52, 0.61, 0.22])
    for user in range(8):
        user_shift = (user - 3.5) * 0.012
        for index, category in enumerate(categories):
            random_rows.append(
                (user, f"v{index}", category, reference_base[index] + user_shift)
            )
            # Unequal row exposure makes RW a genuine sensitivity estimator.
            repeats = 1 + ((user + index) % 3 == 0)
            for repeat in range(repeats):
                standard_rows.append(
                    (
                        user,
                        f"v{index}",
                        category,
                        standard_base[index]
                        + user_shift
                        + repeat * 0.004,
                    )
                )
    columns = ["user_id", "video_id", "cat1_name", "watch_ratio"]
    return pd.DataFrame(random_rows, columns=columns), pd.DataFrame(
        standard_rows, columns=columns
    )


def test_paired_decision_audit_preserves_shift_ranking_and_reports_inference():
    random, standard = _paired_fixture()
    result = decision_aware_audit(
        random,
        standard,
        ks=(1, 2),
        n_boot=30,
        seed=42,
    )

    assert result["design"]["same_user_multiplicity_in_both_logs"] is True
    assert result["design"]["same_seed_and_draw_rule_as_p8"] is True
    assert result["design"]["n_bootstrap"] == 30
    assert result["ci95_metadata"]["scope"] == "nominal pointwise"
    assert result["ci95_metadata"]["multiplicity_adjusted"] is False
    assert result["invariants"]["shift_only_ranking_identical_to_naive"] is True
    for k in ("1", "2"):
        assert result["point"][k]["shift_only"]["selected_categories"] == (
            result["point"][k]["naive"]["selected_categories"]
        )
        change = result["paired_change_vs_naive"][k]["shift_only"]
        assert change["point"]["reference_regret"] == pytest.approx(0)
        assert change["point"]["topk_overlap"] == pytest.approx(0)
        assert change["ci95"]["reference_regret"] == pytest.approx([0, 0])
        assert change["ci95"]["topk_overlap"] == pytest.approx([0, 0])
        assert change["ci95_scope"] == "nominal pointwise"
        assert change["ci95_multiplicity_adjusted"] is False
        for estimator in ("naive", "shift_only", "ce", "rw"):
            assert len(result["ci95"][k][estimator]["reference_regret"]) == 2
            assert len(result["ci95"][k][estimator]["topk_overlap"]) == 2
        for ranker in ("reference", "naive", "shift_only", "ce", "rw"):
            stability = result["category_inclusion_stability"][k][ranker]
            assert sum(stability["summary_counts"].values()) == 4
            assert set(stability["per_category"]) == {"A", "B", "C", "D"}
    family = result["familywise_regret_contrasts_vs_naive"]
    assert family["family_size"] == 4
    assert family["n_shared_bootstrap_draws"] == 30
    assert family["same_bootstrap_draws_for_all_contrasts"] is True
    assert family["pointwise_intervals_retained_elsewhere_are_nominal"] is True
    assert set(family["per_contrast"]) == {
        "ce_minus_naive_k1",
        "rw_minus_naive_k1",
        "ce_minus_naive_k2",
        "rw_minus_naive_k2",
    }
    for label, cell in family["per_contrast"].items():
        lower, upper = cell["simultaneous_ci95"]
        assert lower <= cell["point_regret_contrast"] <= upper
        assert len(cell["nominal_pointwise_percentile_ci95"]) == 2
        assert cell["contrast"] in {"ce minus naive", "rw minus naive"}
    naive_band = result["simultaneous_signed_error_bands"]["naive"][
        "random_share_centered_signed_error"
    ]
    shift_band = result["simultaneous_signed_error_bands"]["shift_only"][
        "random_share_centered_signed_error"
    ]
    for category in ("A", "B", "C", "D"):
        assert naive_band["per_category"][category][
            "point_signed_error"
        ] == pytest.approx(
            shift_band["per_category"][category]["point_signed_error"]
        )
