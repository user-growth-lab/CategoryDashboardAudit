from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from p9_error_ledger import (  # noqa: E402
    B_USERS,
    COMPONENTS,
    canonical_ledger_scope,
    category_error_components,
    english_category_label,
    paired_positive_multiplier_bootstrap,
    weighted_squared_error_ledger,
)


def _ledger_fixture():
    columns = ["user_id", "video_id", "cat1_name", "watch_ratio"]
    random_common = pd.DataFrame(
        [
            (1, "a", "A", 0.2),
            (1, "b", "B", 0.4),
            (2, "a", "A", 0.6),
            (2, "b", "B", 0.8),
        ],
        columns=columns,
    )
    standard_common = pd.DataFrame(
        [
            (1, "a", "A", 0.3),
            (1, "a", "A", 0.5),
            (1, "b", "B", 0.7),
            (2, "a", "A", 0.7),
            (2, "b", "B", 0.9),
            (2, "b", "B", 0.5),
        ],
        columns=columns,
    )
    random_full = pd.concat(
        [
            random_common,
            pd.DataFrame(
                [(1, "random_only", "A", 0.05), (2, "random_only", "A", 0.15)],
                columns=columns,
            ),
        ],
        ignore_index=True,
    )
    standard_full = pd.concat(
        [
            standard_common,
            pd.DataFrame(
                [
                    (1, "standard_only", "B", 0.95),
                    (2, "standard_only", "B", 0.85),
                ],
                columns=columns,
            ),
        ],
        ignore_index=True,
    )
    return random_full, standard_full, random_common, standard_common


def test_publication_default_uses_at_least_one_thousand_draws():
    assert B_USERS >= 1000


def test_publication_category_labels_reuse_shared_english_mapping():
    assert english_category_label("颜值") == "Beauty"
    assert english_category_label("科学与法律") == "Science & law"
    assert english_category_label("ASCII fixture") == "ASCII fixture"
    with pytest.raises(ValueError, match="missing English category label"):
        english_category_label("未翻译类目")


def test_canonical_scope_freezes_users_then_restores_bucket_specific_support():
    random_full, standard_full, _, _ = _ledger_fixture()
    r_full, s_full, r_common, s_common, scope = canonical_ledger_scope(
        random_full,
        standard_full,
        min_categories=2,
    )

    assert set(r_full["user_id"]) == set(s_full["user_id"]) == {1, 2}
    assert set(r_common["video_id"]) == set(s_common["video_id"]) == {"a", "b"}
    assert set(r_full["video_id"]) == {"a", "b", "random_only"}
    assert set(s_full["video_id"]) == {"a", "b", "standard_only"}
    assert scope["full_video_support_overlap"] == {
        "intersection": 2,
        "random_only": 1,
        "standard_only": 1,
    }


def test_four_term_category_path_reconstructs_the_full_gap_exactly():
    frames = _ledger_fixture()
    table = category_error_components(*frames)

    assert table["reconstructed_full_gap"].to_numpy() == pytest.approx(
        table["full_gap"].to_numpy(), abs=1e-12
    )
    assert table["reconstruction_error"].abs().max() < 1e-12
    assert table["standard_support_entry_exit"].abs().sum() > 0
    assert table["random_support_entry_exit"].abs().sum() > 0


def test_level_shape_inner_products_reconstruct_weighted_squared_error():
    frames = _ledger_fixture()
    table = category_error_components(*frames)
    ledger = weighted_squared_error_ledger(
        table[list(COMPONENTS)], table["random_full_exposure_share"]
    )

    full = ledger["full_gap"]
    assert full["total_weighted_squared_error"] == pytest.approx(
        full["level_squared_error"] + full["shape_squared_error"], abs=1e-14
    )
    for metric, component_key in (
        ("level_squared_error", "level_inner_product"),
        ("shape_squared_error", "shape_inner_product"),
        ("total_weighted_squared_error", "total_inner_product"),
    ):
        assert sum(
            ledger["components"][name][component_key] for name in COMPONENTS
        ) == pytest.approx(full[metric], abs=1e-14)
    assert max(abs(value) for value in ledger["reconstruction_error"].values()) < 1e-14


def test_positive_multiplier_bootstrap_pairs_users_and_preserves_video_support():
    frames = _ledger_fixture()
    result = paired_positive_multiplier_bootstrap(
        *frames,
        n_boot=20,
        seed=7,
        chunk_size=4,
    )

    design = result["design"]
    assert design["same_user_weight_all_four_cells"] is True
    assert design["strictly_positive_weights_preserve_fixed_video_support"] is True
    assert design["n_bootstrap"] == 20
    assert result["maximum_bootstrap_path_reconstruction_error"] < 1e-12
    assert result["maximum_bootstrap_ledger_reconstruction_error"] < 1e-12
    for name in COMPONENTS:
        component = result["ci95"]["components"][name]
        assert len(component["level_inner_product"]) == 2
        assert len(component["shape_inner_product"]) == 2
        assert len(component["level_share_signed"]) == 2
        assert len(component["shape_share_signed"]) == 2
    assert set(result["ci95"]["per_category"]) == {"A", "B"}


def test_identical_logs_have_zero_paths_in_every_bootstrap_draw():
    _, _, random_common, _ = _ledger_fixture()
    identical = random_common.copy()
    result = paired_positive_multiplier_bootstrap(
        identical,
        identical,
        identical,
        identical,
        n_boot=12,
        seed=11,
        chunk_size=3,
    )

    for bounds in result["ci95"]["full_gap"].values():
        assert bounds == pytest.approx([0.0, 0.0], abs=1e-14)
    for name in COMPONENTS:
        component = result["ci95"]["components"][name]
        for metric in (
            "weighted_mean",
            "level_inner_product",
            "shape_inner_product",
            "total_inner_product",
        ):
            assert component[metric] == pytest.approx([0.0, 0.0], abs=1e-14)


def test_invalid_random_shares_are_rejected():
    components = pd.DataFrame(
        np.zeros((2, len(COMPONENTS))),
        index=["A", "B"],
        columns=COMPONENTS,
    )
    with pytest.raises(ValueError, match="positive mass"):
        weighted_squared_error_ledger(
            components,
            pd.Series([0.0, 0.0], index=components.index),
        )
