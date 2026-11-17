from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from p13_dependence_sensitivity import (  # noqa: E402
    N_VIDEO_DRAWS,
    evaluate_cluster_weights,
    leave_one_date_out_sensitivity,
    prepare_clustered_statistics,
    shared_video_sensitivity,
)
from p8_paired_audit import point_paired_metrics  # noqa: E402


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    random_rows = []
    standard_rows = []
    values = {
        1: {"A": 0.2, "B": 0.7},
        2: {"A": 0.6, "B": 0.3},
        3: {"A": 0.4, "B": 0.8},
    }
    for user, category_values in values.items():
        for date, date_shift in ((20220422, 0.0), (20220423, 0.05), (20220424, -0.03)):
            for video, category in (("a1", "A"), ("a2", "A"), ("b1", "B")):
                base = category_values[category] + date_shift
                random_rows.append((user, video, category, date, base))
                # A user/date-dependent production shift creates a nonzero gap
                # while leaving every support cell populated.
                shift = 0.08 + 0.02 * (user == 1) - 0.01 * (date == 20220424)
                standard_rows.append((user, video, category, date, base + shift))
    columns = ["user_id", "video_id", "cat1_name", "date", "watch_ratio"]
    return pd.DataFrame(random_rows, columns=columns), pd.DataFrame(
        standard_rows, columns=columns
    )


def test_publication_defaults_use_substantial_sensitivity_draws():
    assert N_VIDEO_DRAWS >= 500


def test_unit_weights_reproduce_p8_canonical_point_estimates():
    random, standard = _fixture()
    stats = prepare_clustered_statistics(random, standard)
    weighted = evaluate_cluster_weights(stats)
    p8 = point_paired_metrics(random, standard, topk=1)

    for estimator in ("naive", "demean"):
        for metric in (
            "absolute_mae",
            "centered_mae",
            "max_abs_error",
            "rank_corr",
            "top10_overlap",
        ):
            assert weighted["metrics"][estimator][metric] == pytest.approx(
                p8[estimator][metric], abs=1e-12
            )


def test_leave_one_date_out_is_a_frozen_scope_range_not_an_interval():
    random, standard = _fixture()
    stats = prepare_clustered_statistics(random, standard)
    result = leave_one_date_out_sensitivity(stats, topk=1)

    assert result["design"]["selection_scope_frozen"] is True
    assert result["design"]["support_repaired_after_deletion"] is False
    assert result["design"]["is_confidence_interval"] is False
    assert result["design"]["interpretation"] == "calendar-deletion sensitivity range"
    assert result["n_omitted_dates"] == 3
    assert set(result["cells"]) == {"20220422", "20220423", "20220424"}
    assert result["directional_stability"]["absolute_mae"]["total"] == 3
    assert set(result["range_across_omitted_dates"]["direct_change"]) >= {
        "absolute_mae",
        "centered_mae",
    }


def test_video_perturbation_uses_one_positive_weight_per_shared_video():
    random, standard = _fixture()
    stats = prepare_clustered_statistics(random, standard)
    result = shared_video_sensitivity(stats, n_draws=20, seed=17, topk=1)

    design = result["design"]
    assert design["cluster"] == "video"
    assert design["same_video_weights_in_both_buckets"] is True
    assert design["strictly_positive_weights_preserve_observed_support"] is True
    assert design["is_confidence_interval"] is False
    assert result["n_draws"] == 20
    assert 0.0 <= result["directional_stability"][
        "all_three_error_metrics_favorable_fraction"
    ] <= 1.0


def test_negative_or_all_zero_cluster_weights_are_rejected():
    random, standard = _fixture()
    stats = prepare_clustered_statistics(random, standard)

    with pytest.raises(ValueError, match="non-negative"):
        evaluate_cluster_weights(stats, date_weights=np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="all zero"):
        evaluate_cluster_weights(stats, date_weights=np.zeros(3))
