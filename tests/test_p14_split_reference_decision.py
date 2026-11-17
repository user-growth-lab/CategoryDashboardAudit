from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from p14_split_reference_decision import (  # noqa: E402
    compare_with_p11,
    frozen_event_hash_assignments,
    frozen_reference_split,
    heldout_decision_metrics,
    split_reference_decision_audit,
)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    random_rows = []
    standard_rows = []
    categories = ["A", "B", "C", "D"]
    random_base = {"A": 0.90, "B": 0.70, "C": 0.40, "D": 0.15}
    standard_base = {"A": 0.68, "B": 0.82, "C": 0.35, "D": 0.22}
    for user in range(14):
        user_shift = (user - 6.5) * 0.004
        for category_index, category in enumerate(categories):
            for repeat in range(6):
                video_id = category_index * 100 + repeat
                random_rows.append(
                    {
                        "user_id": user,
                        "video_id": video_id,
                        "date": 20220422 + ((user + repeat) % 16),
                        "duration_ms": 10000 + 100 * video_id,
                        "tab": 1,
                        "cat1_name": category,
                        "watch_ratio": random_base[category]
                        + user_shift
                        + (repeat - 2.5) * 0.002,
                    }
                )
            # Unequal exposure and user/category coverage make CE and RW real
            # alternatives while every user still has all four categories.
            for repeat in range(1 + ((user + category_index) % 3)):
                standard_rows.append(
                    {
                        "user_id": user,
                        "video_id": category_index * 100 + repeat,
                        "cat1_name": category,
                        "watch_ratio": standard_base[category]
                        + user_shift
                        + repeat * 0.003,
                    }
                )
    return pd.DataFrame(random_rows), pd.DataFrame(standard_rows)


def test_frozen_hash_split_is_outcome_free_and_order_invariant():
    random, _ = _fixture()
    original = frozen_event_hash_assignments(random)

    changed = random.copy()
    changed["watch_ratio"] = np.linspace(0.0, 1.0, len(changed))
    changed["play_time_ms"] = np.arange(len(changed))
    changed_assignments = frozen_event_hash_assignments(changed)
    assert original.equals(changed_assignments)

    shuffled = random.sample(frac=1.0, random_state=7)
    shuffled_assignments = frozen_event_hash_assignments(shuffled)
    assert original.sort_values("event_sha256").reset_index(drop=True).equals(
        shuffled_assignments.sort_values("event_sha256").reset_index(drop=True)
    )


def test_split_is_complete_disjoint_and_has_category_support():
    random, _ = _fixture()
    selection, evaluation, metadata = frozen_reference_split(random)

    assert len(selection) + len(evaluation) == len(random)
    assert set(selection.index).isdisjoint(set(evaluation.index))
    assert set(selection["cat1_name"]) == {"A", "B", "C", "D"}
    assert set(evaluation["cat1_name"]) == {"A", "B", "C", "D"}
    assert metadata["outcome_fields_in_key"] == []
    assert metadata["invariants"] == {
        "row_partition_complete": True,
        "event_signatures_disjoint": True,
        "selection_has_complete_category_support": True,
        "evaluation_has_complete_category_support": True,
        "outcome_free_assignment": True,
    }


def test_heldout_regret_can_be_negative_without_using_evaluation_for_selection():
    selection = pd.Series({"A": 0.9, "B": 0.8, "C": 0.1})
    evaluation = pd.Series({"A": 0.2, "B": 0.9, "C": 0.1})
    candidate = pd.Series({"A": 0.1, "B": 1.0, "C": 0.0})

    result = heldout_decision_metrics(
        candidate, selection, evaluation, k=1
    )

    assert result["selection_reference_selected_categories"] == ["A"]
    assert result["selected_categories"] == ["B"]
    assert result["selection_reference_heldout_evaluation_mean"] == pytest.approx(
        0.2
    )
    assert result["candidate_heldout_evaluation_mean"] == pytest.approx(0.9)
    assert result["split_reference_regret"] == pytest.approx(-0.7)


def test_split_audit_reports_nominal_and_familywise_inference():
    random, standard = _fixture()
    selection, evaluation, _ = frozen_reference_split(random)
    result = split_reference_decision_audit(
        selection, evaluation, standard, ks=(1, 2), n_boot=40, seed=11
    )

    assert result["design"]["n_bootstrap"] == 40
    assert result["design"][
        "same_user_multiplicity_in_selection_evaluation_and_standard"
    ] is True
    assert result["design"]["split_membership_fixed_across_bootstrap_draws"] is True
    assert result["nominal_ci95_metadata"]["scope"] == "nominal pointwise"
    assert result["nominal_ci95_metadata"]["multiplicity_adjusted"] is False
    assert result["invariants"][
        "regret_and_value_contrasts_are_exact_negatives"
    ] is True
    for k in ("1", "2"):
        for estimator in ("naive", "ce", "rw"):
            cell = result["point"][k][estimator]
            assert len(cell["selected_categories"]) == int(k)
            assert len(
                result["nominal_ci95"][k][estimator]["split_reference_regret"]
            ) == 2
        for estimator in ("ce", "rw"):
            change = result["paired_change_vs_naive"][k][estimator]
            assert change["point"]["split_reference_regret"] == pytest.approx(
                -change["point"]["candidate_heldout_evaluation_mean"]
            )

    family = result["familywise_regret_contrasts_vs_naive"]
    assert family["family_size"] == 4
    assert family["n_shared_bootstrap_draws"] == 40
    assert family["same_bootstrap_draws_for_all_contrasts"] is True
    assert set(family["per_contrast"]) == {
        "ce_minus_naive_k1",
        "rw_minus_naive_k1",
        "ce_minus_naive_k2",
        "rw_minus_naive_k2",
    }
    for cell in family["per_contrast"].values():
        lower, upper = cell["simultaneous_ci95"]
        assert lower <= cell["point_regret_contrast"] <= upper
        assert len(cell["nominal_pointwise_percentile_ci95"]) == 2


def test_candidate_rankings_do_not_change_when_random_outcomes_change():
    random, standard = _fixture()
    selection, evaluation, _ = frozen_reference_split(random)
    first = split_reference_decision_audit(
        selection, evaluation, standard, ks=(1, 2), n_boot=10, seed=5
    )

    changed_selection = selection.copy()
    changed_evaluation = evaluation.copy()
    changed_selection["watch_ratio"] = 1.0 - changed_selection["watch_ratio"]
    changed_evaluation["watch_ratio"] = 1.0 - changed_evaluation["watch_ratio"]
    second = split_reference_decision_audit(
        changed_selection,
        changed_evaluation,
        standard,
        ks=(1, 2),
        n_boot=10,
        seed=5,
    )

    for estimator in ("naive", "ce", "rw"):
        assert first["point_category_vectors"][estimator] == pytest.approx(
            second["point_category_vectors"][estimator]
        )
        for k in ("1", "2"):
            assert first["point"][k][estimator]["selected_categories"] == (
                second["point"][k][estimator]["selected_categories"]
            )


def test_p11_comparison_distinguishes_point_and_familywise_directions():
    random, standard = _fixture()
    selection, evaluation, _ = frozen_reference_split(random)
    split = split_reference_decision_audit(
        selection, evaluation, standard, ks=(1, 2), n_boot=10, seed=3
    )
    p11 = {
        "analysis": "fixture P11",
        "paired_change_vs_naive": {},
        "familywise_regret_contrasts_vs_naive": {"per_contrast": {}},
    }
    for k in ("1", "2"):
        p11["paired_change_vs_naive"][k] = {}
        for estimator in ("ce", "rw"):
            point = split["paired_change_vs_naive"][k][estimator]["point"][
                "split_reference_regret"
            ]
            label = f"{estimator}_minus_naive_k{k}"
            p11["paired_change_vs_naive"][k][estimator] = {
                "point": {"reference_regret": point},
                "ci95": {"reference_regret": [-0.1, 0.1]},
            }
            p11["familywise_regret_contrasts_vs_naive"]["per_contrast"][label] = {
                "simultaneous_ci95": [-0.2, 0.2],
                "supported_direction": "not family-wise distinguishable from zero",
            }
    comparison = compare_with_p11(split, p11)

    assert comparison["summary"]["all_point_directions_maintained"] is True
    assert comparison["family_size"] == 4
    assert all(
        "split_familywise_supported_direction" in cell
        for cell in comparison["per_contrast"].values()
    )
