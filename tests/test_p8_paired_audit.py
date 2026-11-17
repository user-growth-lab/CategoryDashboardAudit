from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from p8_paired_audit import (  # noqa: E402
    B_TEMPORAL,
    B_USERS,
    aggregation_target_sensitivity,
    estimates_from_user_multiplicity,
    full_vs_common_video_support_sensitivity,
    four_way_temporal_scope,
    item_gap_accounting,
    min_category_sensitivity,
    paired_sufficient_statistics,
    paired_user_bootstrap,
    shapley_absolute_accounting,
    temporal_paired_analysis,
)


def test_publication_default_uses_at_least_one_thousand_bootstrap_draws():
    assert B_USERS >= 1000
    assert B_TEMPORAL >= 1000


def _paired_bootstrap_fixture():
    rows = []
    # Each user sees both categories and both videos in both buckets.  Outcomes
    # differ across users, so independently resampling the buckets would create
    # nonzero naive gaps even though the two input frames are identical.
    for user, (a, b) in enumerate([(0.1, 0.4), (0.3, 0.7), (0.8, 0.2), (0.6, 0.9)]):
        rows.extend(
            [
                (user, "a", "A", a),
                (user, "b", "B", b),
            ]
        )
    frame = pd.DataFrame(
        rows, columns=["user_id", "video_id", "cat1_name", "watch_ratio"]
    )
    return frame.copy(), frame.copy()


def test_paired_bootstrap_uses_one_synchronized_multiplicity_vector():
    random, standard = _paired_bootstrap_fixture()
    result = paired_user_bootstrap(
        random,
        standard,
        n_boot=40,
        seed=42,
        topk=1,
        return_replicates=True,
    )

    assert result["design"]["same_user_multiplicity_in_both_buckets"] is True
    # Identical bucket rows stay identical in every draw only when the same
    # user multiplicities are applied synchronously.
    assert all(
        row["naive.absolute_mae"] == pytest.approx(0.0, abs=1e-14)
        for row in result["replicates"]
    )
    assert result["ci95"]["naive"]["absolute_mae"] == pytest.approx([0.0, 0.0])
    assert result["point"]["naive"]["random_share_weighted_absolute_mae"] == pytest.approx(
        0.0
    )
    change = result["naive_to_demean_metric_change"]
    assert set(change["point"]) == {
        "absolute_mae",
        "centered_mae",
        "max_abs_error",
        "rank_corr",
        "top10_overlap",
    }
    assert all(len(bounds) == 2 for bounds in change["ci95"].values())
    assert set(result["relative_error_reduction"]["point"]) == {
        "absolute_mae",
        "centered_mae",
        "max_abs_error",
    }
    for estimator in ("naive", "demean"):
        assert set(result["ci95"][estimator]) == {
            "absolute_mae",
            "centered_mae",
            "max_abs_error",
            "rank_corr",
            "top10_overlap",
        }


def test_explicit_multiplicity_is_applied_to_both_bucket_statistics():
    random, standard = _paired_bootstrap_fixture()
    stats = paired_sufficient_statistics(random, standard)
    vectors = estimates_from_user_multiplicity(stats, np.array([2, 0, 1, 1]))

    assert vectors["naive"].to_numpy() == pytest.approx(
        vectors["reference"].to_numpy()
    )
    assert vectors["reference_shares"].sum() == pytest.approx(1.0)


def test_finite_multiplicity_reduction_emits_no_runtime_warnings():
    random, standard = _paired_bootstrap_fixture()
    stats = paired_sufficient_statistics(random, standard)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vectors = estimates_from_user_multiplicity(
            stats, np.array([1.0, 3.0, 0.0, 2.0])
        )

    for vector in vectors.values():
        assert np.isfinite(vector.to_numpy(dtype=float)).all()


def _unequal_exposure_fixture():
    random_rows = []
    standard_rows = []
    for user in (1, 2):
        random_rows.extend(
            [(user, "a", "A", 0.6), (user, "b", "B", 0.4)]
        )
    standard_rows.extend(
        [
            (1, "a", "A", 0.9),
            (1, "a", "A", 0.9),
            (1, "a", "A", 0.9),
            (1, "b", "B", 0.1),
            (2, "a", "A", 0.2),
            (2, "b", "B", 0.8),
            (2, "b", "B", 0.8),
            (2, "b", "B", 0.8),
        ]
    )
    columns = ["user_id", "video_id", "cat1_name", "watch_ratio"]
    return pd.DataFrame(random_rows, columns=columns), pd.DataFrame(
        standard_rows, columns=columns
    )


def test_aggregation_sensitivity_changes_only_production_residual_weights():
    random, standard = _unequal_exposure_fixture()
    result = aggregation_target_sensitivity(random, standard, topk=1)

    assert result["random_reference_target_held_fixed"] is True
    comparison = result["vector_comparison"]
    assert comparison["row_weighted_minus_cell_equal_mae"] > 0
    assert comparison["row_weighted_minus_cell_equal_max_abs"] > 0
    for estimator in ("cell_equal_user_category", "row_exposure_weighted"):
        assert "random_share_weighted_absolute_mae" in result["point"][estimator]


def test_min_category_sensitivity_rebuilds_scope_for_every_threshold():
    random, standard = _unequal_exposure_fixture()
    # User 2 covers one category only, without removing global video support.
    standard = standard[
        ~((standard["user_id"] == 2) & (standard["cat1_name"] == "B"))
    ].copy()
    result = min_category_sensitivity(
        random, standard, thresholds=(1, 2), topk=1
    )

    assert result["thresholds"] == [1, 2]
    assert result["cells"]["1"]["metrics"]["n_users"] == 2
    assert result["cells"]["2"]["metrics"]["n_users"] == 1


def test_full_support_sensitivity_reports_retention_and_metric_change():
    random_common, standard_common = _unequal_exposure_fixture()
    random_full = pd.concat(
        [
            random_common,
            pd.DataFrame(
                [(1, "random_only", "A", 0.0)],
                columns=random_common.columns,
            ),
        ],
        ignore_index=True,
    )
    standard_full = pd.concat(
        [
            standard_common,
            pd.DataFrame(
                [(2, "standard_only", "B", 1.0)],
                columns=standard_common.columns,
            ),
        ],
        ignore_index=True,
    )
    result = full_vs_common_video_support_sensitivity(
        random_full,
        standard_full,
        random_common,
        standard_common,
        min_categories=2,
        topk=1,
    )

    assert result["full_bucket_specific_support"]["random"]["videos"] == 3
    assert result["common_video_support"]["random"]["videos"] == 2
    assert result["common_over_full_retention"]["random"]["video_fraction"] == pytest.approx(
        2 / 3
    )
    assert set(result["common_minus_full_metric_change"]) == {"naive", "demean"}


def test_item_gap_accounting_bootstrap_covers_absolute_and_centered_summaries():
    random, standard = _unequal_exposure_fixture()
    result = item_gap_accounting(
        standard, random, n_boot=20, seed=7
    )
    bootstrap = result["paired_user_multiplier_bootstrap"]

    assert bootstrap["design"]["same_user_weight_in_both_buckets"] is True
    assert bootstrap["design"]["n_bootstrap"] == 20
    assert bootstrap["component_display_names"]["response_context"] == (
        "within-video residual"
    )
    for name in (
        "absolute_response_context_shapley_mean",
        "absolute_item_mix_shapley_mean",
        "centered_response_context_shapley_mean",
        "centered_item_mix_shapley_mean",
    ):
        assert len(bootstrap["ci95"][name]) == 2
    summary = result["summary_unweighted_across_categories"]
    assert summary["absolute_response_context_share"] + summary[
        "absolute_item_mix_share"
    ] == pytest.approx(1.0)
    assert summary["centered_response_context_share"] + summary[
        "centered_item_mix_share"
    ] == pytest.approx(1.0)


def _temporal_frame(bucket_offset=0.0):
    rows = []
    dates = (20220425, 20220503)
    for date in dates:
        # User 1 is stable on common videos a/b.  User 2 initially has two
        # categories, but its B video is period/bucket-specific and disappears
        # under the four-way V intersection, making user 2 ineligible.
        rows.extend(
            [
                (1, "a", "A", date, 0.7 + bucket_offset),
                (1, "b", "B", date, 0.4 + bucket_offset),
                (2, "a", "A", date, 0.6 + bucket_offset),
            ]
        )
        is_early = date < 20220501
        # Both full buckets contain b_x and b_y, but in opposite halves.  Thus
        # they satisfy the full-period paired-V input invariant while neither
        # video is on four-way temporal support.
        if bucket_offset == 0.0:
            extra_video = "b_x" if is_early else "b_y"
        else:
            extra_video = "b_y" if is_early else "b_x"
        rows.append((2, extra_video, "B", date, 0.3 + bucket_offset))
    return pd.DataFrame(
        rows,
        columns=["user_id", "video_id", "cat1_name", "date", "watch_ratio"],
    )


def test_temporal_scope_reaches_four_way_fixed_point_on_common_users_videos():
    random = _temporal_frame(0.0)
    standard = _temporal_frame(0.05)
    periods, metadata = four_way_temporal_scope(
        random, standard, full_period_min_categories=2
    )

    for frame in periods.values():
        # User 2 has only category A after the four-way video intersection,
        # but remains because full-period eligibility was frozen before split.
        assert set(frame["user_id"]) == {1, 2}
        assert set(frame["video_id"]) == {"a", "b"}
        assert set(frame["cat1_name"]) == {"A", "B"}
    trajectory = [
        (entry["common_users"], entry["common_videos"])
        for entry in metadata["iteration_log"]
    ]
    assert trajectory == [(2, 4), (2, 2)]
    assert metadata["changed_updates_to_fixed_point"] == 1
    assert metadata["stability_check_iteration"] == 2
    assert metadata["full_period_eligibility_frozen"] is True
    assert metadata["half_period_eligibility_filter"] is False
    assert all(metadata["invariants"].values())


def test_temporal_joint_bootstrap_reports_synchronized_difference_intervals():
    random = _temporal_frame(0.0)
    standard = _temporal_frame(0.05)
    result = temporal_paired_analysis(
        random,
        standard,
        n_boot=20,
        seed=42,
        full_period_min_categories=2,
        topk=1,
    )

    assert result["bootstrap_design"][
        "same_user_multiplicity_in_all_four_strata"
    ] is True
    for estimator in ("naive", "demean"):
        ci = result["late_minus_early_metric_change"]["ci95"][estimator]
        assert set(ci) >= {"absolute_mae", "centered_mae", "rank_corr"}
        assert all(len(bounds) == 2 for bounds in ci.values())
        gap_ci = result["gap_vector_early_late_rank_corr"]["ci95"][estimator]
        assert len(gap_ci) == 2


def test_absolute_and_centered_shapley_accounting_is_additive():
    response = pd.Series({"A": 0.30, "B": -0.20, "C": 0.05})
    item_mix = pd.Series({"A": -0.10, "B": -0.15, "C": 0.20})

    absolute = shapley_absolute_accounting(response, item_mix)
    assert absolute["absolute_total"].to_numpy() == pytest.approx(
        (response + item_mix).abs().to_numpy()
    )
    assert (
        absolute["first_shapley"] + absolute["second_shapley"]
    ).to_numpy() == pytest.approx(absolute["absolute_total"].to_numpy())
    assert absolute["reconstruction_error"].abs().max() < 1e-12

    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    centered_response = response - float((response * weights).sum())
    centered_item = item_mix - float((item_mix * weights).sum())
    centered = shapley_absolute_accounting(centered_response, centered_item)
    assert (
        centered["first_shapley"] + centered["second_shapley"]
    ).to_numpy() == pytest.approx(centered["absolute_total"].to_numpy())
    assert centered["reconstruction_error"].abs().max() < 1e-12
