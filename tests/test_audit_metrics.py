from pathlib import Path
import sys

import pandas as pd
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from audit_metrics import (  # noqa: E402
    eligible_production_rows,
    evaluate_category_estimate,
    paired_user_audit_scope,
    symmetric_category_gap_decomposition,
    user_demeaned_category_estimate,
)


def test_main_cohort_and_cell_equal_demeaning_are_explicit():
    standard = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 2, 3],
            "cat1_name": ["A", "A", "B", "A", "B", "B", "B", "A"],
            "watch_ratio": [0.8, 0.8, 0.2, 0.6, 0.4, 0.4, 0.4, 0.9],
        }
    )

    eligible, coverage = eligible_production_rows(standard, min_categories=2)
    assert set(eligible["user_id"]) == {1, 2}
    assert coverage.to_dict() == {1: 2, 2: 2, 3: 1}

    absolute, centered = user_demeaned_category_estimate(
        eligible, ["A", "B"], level_anchor=0.6
    )
    # User means are exposure weighted, then observed user-category cells are
    # weighted equally within category: A=(.2+.15)/2; B=(-.4-.05)/2.
    assert centered["A"] == pytest.approx(0.175)
    assert centered["B"] == pytest.approx(-0.225)
    assert absolute["A"] == pytest.approx(0.775)
    assert absolute["B"] == pytest.approx(0.375)


def test_centered_shape_uses_one_fixed_reference_weight_vector():
    reference = pd.Series({"A": 0.6, "B": 0.4})
    estimate = pd.Series({"A": 0.8, "B": 0.1})
    random_exposure_weights = pd.Series({"A": 0.75, "B": 0.25})

    metrics = evaluate_category_estimate(
        estimate, reference, random_exposure_weights, topk=1
    )

    assert metrics["estimate_center_random_share_weighted"] == pytest.approx(0.625)
    assert metrics["reference_center_random_share_weighted"] == pytest.approx(0.55)
    assert metrics["mae"] == pytest.approx(0.25)
    assert metrics["centered_shape_mae"] == pytest.approx(0.25)
    assert metrics["max_abs_error"] == pytest.approx(0.3)
    assert metrics["rank_corr"] == pytest.approx(1.0)
    assert metrics["top1_overlap"] == pytest.approx(1.0)


def test_paired_scope_drops_five_eligible_users_missing_from_random():
    standard_rows = []
    random_rows = []
    for user_id in range(10):
        standard_rows.extend(
            [
                (user_id, "a", "A", 0.8),
                (user_id, "b", "B", 0.4),
            ]
        )
        if user_id < 5:
            random_rows.extend(
                [
                    (user_id, "a", "A", 0.7),
                    (user_id, "b", "B", 0.5),
                ]
            )
    columns = ["user_id", "video_id", "cat1_name", "watch_ratio"]
    standard = pd.DataFrame(standard_rows, columns=columns)
    random = pd.DataFrame(random_rows, columns=columns)

    random_paired, standard_paired, scope = paired_user_audit_scope(
        random, standard, min_categories=2
    )

    assert set(random_paired["user_id"]) == set(range(5))
    assert set(standard_paired["user_id"]) == set(range(5))
    assert scope["standard_eligible_users_before_random_intersection"] == 10
    assert scope["eligible_users_missing_from_random"] == 5
    assert scope["users_after_initial_random_intersection"] == 5
    assert scope["final"]["random"] == scope["final"]["standard"]
    assert len(scope["primary_scope_id"]) == 64
    assert len(scope["final"]["user_id_sha256"]) == 64
    assert all(scope["invariants"].values())


def test_paired_scope_iterates_video_and_coverage_restrictions_to_fixed_point():
    # User 1 supplies x only in random but becomes ineligible after the first
    # common-video restriction.  Removing user 1 then removes x from random,
    # which makes user 2 ineligible on the next pass.  User 3 is stable.
    columns = ["user_id", "video_id", "cat1_name", "watch_ratio"]
    standard = pd.DataFrame(
        [
            (1, "a1", "A", 0.9),
            (1, "b_missing", "B", 0.2),
            (2, "x", "A", 0.8),
            (2, "b2", "B", 0.3),
            (3, "a3", "A", 0.7),
            (3, "b3", "B", 0.4),
        ],
        columns=columns,
    )
    random = pd.DataFrame(
        [
            (1, "x", "A", 0.6),
            (2, "b2", "B", 0.5),
            (3, "a3", "A", 0.65),
            (3, "b3", "B", 0.45),
        ],
        columns=columns,
    )

    random_paired, standard_paired, scope = paired_user_audit_scope(
        random, standard, min_categories=2
    )

    assert set(random_paired["user_id"]) == {3}
    assert set(standard_paired["user_id"]) == {3}
    assert set(random_paired["video_id"]) == {"a3", "b3"}
    assert set(standard_paired["video_id"]) == {"a3", "b3"}
    assert set(random_paired["cat1_name"]) == {"A", "B"}
    assert set(standard_paired["cat1_name"]) == {"A", "B"}
    assert scope["iterations_to_fixed_point"] >= 2
    assert scope["final"]["minimum_standard_user_category_coverage"] == 2


def test_symmetric_category_gap_decomposition_is_exact():
    columns = ["video_id", "cat1_name", "watch_ratio"]
    standard = pd.DataFrame(
        [
            ("a1", "A", 0.9),
            ("a1", "A", 0.9),
            ("a1", "A", 0.9),
            ("a2", "A", 0.3),
            ("b1", "B", 0.7),
        ],
        columns=columns,
    )
    reference = pd.DataFrame(
        [
            ("a1", "A", 0.5),
            ("a2", "A", 0.4),
            ("a2", "A", 0.4),
            ("a2", "A", 0.4),
            ("b1", "B", 0.6),
        ],
        columns=columns,
    )

    decomposition = symmetric_category_gap_decomposition(standard, reference)

    assert decomposition.loc["A", "standard"] == pytest.approx(0.75)
    assert decomposition.loc["A", "reference"] == pytest.approx(0.425)
    assert decomposition.loc["A", "gap"] == pytest.approx(0.325)
    assert decomposition.loc["A", "response_context"] == pytest.approx(0.15)
    assert decomposition.loc["A", "item_mix"] == pytest.approx(0.175)
    assert decomposition.loc["B", "gap"] == pytest.approx(0.1)
    assert decomposition.loc["B", "response_context"] == pytest.approx(0.1)
    assert decomposition.loc["B", "item_mix"] == pytest.approx(0.0)
    assert decomposition["reconstruction_error"].abs().max() < 1e-12
    assert decomposition["gap"].to_numpy() == pytest.approx(
        (
            decomposition["response_context"] + decomposition["item_mix"]
        ).to_numpy()
    )
