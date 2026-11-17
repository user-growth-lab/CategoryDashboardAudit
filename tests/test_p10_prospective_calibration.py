from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from p10_prospective_calibration import (  # noqa: E402
    B_USERS,
    PRIMARY_SPLIT,
    SECONDARY_SPLIT,
    evaluate_frozen_calibration,
    fit_frozen_calibration,
    prospective_paired_bootstrap,
    strict_prospective_scope,
)


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["user_id", "video_id", "cat1_name", "watch_ratio"],
    )


def _five_strata_fixture():
    # User 3 deliberately fails the history-only coverage rule.  Every retained
    # user/video/category is observed in the three fitting strata and both test
    # logs, while outcomes move enough to make the frozen shifts nontrivial.
    history = _frame(
        [
            (1, "a", "A", 0.2),
            (1, "b", "B", 0.4),
            (2, "a", "A", 0.5),
            (2, "b", "B", 0.7),
            (3, "a", "A", 0.9),
        ]
    )
    calibration_random = _frame(
        [
            (1, "a", "A", 0.4),
            (1, "b", "B", 0.6),
            (2, "a", "A", 0.6),
            (2, "b", "B", 0.8),
            (3, "a", "A", 0.2),
        ]
    )
    calibration_standard = _frame(
        [
            (1, "a", "A", 0.2),
            (1, "b", "B", 0.3),
            (2, "a", "A", 0.4),
            (2, "b", "B", 0.5),
            (3, "a", "A", 0.1),
        ]
    )
    test_random = _frame(
        [
            (1, "a", "A", 0.5),
            (1, "b", "B", 0.7),
            (2, "a", "A", 0.7),
            (2, "b", "B", 0.9),
            (3, "a", "A", 0.3),
        ]
    )
    test_standard = _frame(
        [
            (1, "a", "A", 0.3),
            (1, "b", "B", 0.4),
            (2, "a", "A", 0.5),
            (2, "b", "B", 0.6),
            (3, "a", "A", 0.2),
        ]
    )
    return (
        history,
        calibration_random,
        calibration_standard,
        test_random,
        test_standard,
    )


def test_publication_defaults_include_primary_and_second_rolling_origin():
    assert B_USERS >= 1000
    assert PRIMARY_SPLIT == ((20220422, 20220430), (20220501, 20220508))
    assert SECONDARY_SPLIT[0][1] < SECONDARY_SPLIT[1][0]


def test_scope_eligibility_is_learned_from_history_before_test_selection():
    scope = strict_prospective_scope(*_five_strata_fixture(), min_categories=2)

    assert scope["metadata"]["history_eligible_users"] == 2
    assert set(scope["history_standard"]["user_id"]) == {1, 2}
    assert set(scope["calibration_random"]["user_id"]) == {1, 2}
    assert set(scope["test_random"]["user_id"]) == {1, 2}
    assert set(scope["test_random"]["user_id"]) == set(
        scope["test_standard"]["user_id"]
    )
    assert scope["metadata"]["frozen_before_test_evaluation"] is True
    assert scope["metadata"]["test_outcomes_used_for_fitting"] is False


def test_test_random_outcomes_cannot_change_any_deployable_vector():
    scope = strict_prospective_scope(*_five_strata_fixture(), min_categories=2)
    model = fit_frozen_calibration(
        scope["history_standard"],
        scope["calibration_random"],
        scope["calibration_standard"],
    )
    original = evaluate_frozen_calibration(
        model, scope["test_random"], scope["test_standard"], topk=1
    )
    changed_random = scope["test_random"].copy()
    changed_random["watch_ratio"] += 10.0
    changed = evaluate_frozen_calibration(
        model, changed_random, scope["test_standard"], topk=1
    )

    deployable = (
        "naive",
        "grand_mean_shift",
        "category_vector_shift",
        "prospective_ce_cell_equal",
        "prospective_rw_row_weighted",
    )
    for name in deployable:
        assert changed["category_vectors"][name] == pytest.approx(
            original["category_vectors"][name]
        )
    assert changed["category_vectors"]["reference"] != pytest.approx(
        original["category_vectors"]["reference"]
    )
    assert changed["category_vectors"]["oracle_contemporaneous_ce"] != pytest.approx(
        original["category_vectors"]["oracle_contemporaneous_ce"]
    )


def test_frozen_scalar_and_category_shifts_are_calibration_only():
    scope = strict_prospective_scope(*_five_strata_fixture(), min_categories=2)
    model = fit_frozen_calibration(
        scope["history_standard"],
        scope["calibration_random"],
        scope["calibration_standard"],
    )
    evaluated = evaluate_frozen_calibration(
        model, scope["test_random"], scope["test_standard"], topk=1
    )

    naive = pd.Series(evaluated["category_vectors"]["naive"])
    scalar = pd.Series(evaluated["category_vectors"]["grand_mean_shift"])
    vector = pd.Series(evaluated["category_vectors"]["category_vector_shift"])
    expected_shift = model["calibration_scalar_shift"]
    expected_vector = pd.Series(model["calibration_category_shift"])
    assert (scalar - naive).to_numpy() == pytest.approx(expected_shift)
    assert (vector - naive).sort_index().to_numpy() == pytest.approx(
        expected_vector.sort_index().to_numpy()
    )


def test_paired_bootstrap_reuses_one_user_multiplicity_across_all_strata():
    history, _, _, _, _ = _five_strata_fixture()
    # Identical calibration/test strata imply zero naive error on every draw.
    scope = strict_prospective_scope(
        history, history, history, history, history, min_categories=2
    )
    result = prospective_paired_bootstrap(
        scope,
        n_boot=30,
        seed=7,
        topk=1,
        return_replicates=True,
    )

    assert result["design"]["same_user_multiplicity_all_five_strata"] is True
    assert result["design"]["refit_calibration_within_draw"] is True
    assert "category_vector_shift_minus_grand_mean_shift" in result[
        "planned_paired_contrasts"
    ]
    assert all(
        row["naive.absolute_mae"] == pytest.approx(0.0, abs=1e-14)
        for row in result["replicates"]
    )
    for method in result["methods"]:
        assert set(result["ci95"][method]) == {
            "absolute_mae",
            "centered_mae",
            "max_abs_error",
            "rank_corr",
            "top10_overlap",
        }


def test_fit_rejects_missing_history_baseline_for_a_calibration_user():
    history, cr, cs, _, _ = _five_strata_fixture()
    with pytest.raises(ValueError, match="identical fitting user sets"):
        fit_frozen_calibration(history[history["user_id"] == 1], cr, cs)
