from pathlib import Path

import pandas as pd


def _write_log(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_loader_enforces_period_tab_and_common_video(tmp_path, monkeypatch):
    import sys

    code = Path(__file__).resolve().parents[1] / "code"
    sys.path.insert(0, str(code))
    import data_utils

    local_data = tmp_path / "project-data"
    local_data.mkdir()
    monkeypatch.setattr(data_utils, "DATA_DIR", local_data)
    monkeypatch.delenv(data_utils.KUAIRAND_CATEGORY_ENV, raising=False)
    # Deliberately create colliding numeric KuaiRec IDs with wrong labels.  The
    # KuaiRand loader must not inspect or join this file.
    pd.DataFrame(
        {
            "video_id": [10, 11, 12],
            "cat1_name": ["WRONG_KUAIREC_A", "WRONG_KUAIREC_B", "WRONG_KUAIREC_C"],
        }
    ).to_csv(local_data / "kuairec_caption_clean.csv", index=False)
    pd.DataFrame(
        {
            "final_video_id": [10, 11, 12],
            "first_level_category_name": [
                "KUAIRAND_A",
                "KUAIRAND_B",
                "KUAIRAND_C",
            ],
        }
    ).to_csv(
        local_data / data_utils.KUAIRAND_CATEGORY_FALLBACK,
        index=False,
    )
    # Missing legacy ``tag`` must not exclude a video with a valid official
    # first-level category.
    pd.DataFrame({"video_id": [10, 11, 12], "tag": [None, "y", "z"]}).to_csv(
        tmp_path / data_utils.VIDEO_FEATURES, index=False
    )

    base = {
        "user_id": 1,
        "date": 20220422,
        "tab": 1,
        "play_time_ms": 50,
        "duration_ms": 100,
    }
    _write_log(
        tmp_path / data_utils.RANDOM_LOG,
        [
            {**base, "video_id": 10},
            {**base, "video_id": 11, "tab": 2},
        ],
    )
    _write_log(
        tmp_path / data_utils.STANDARD_LOGS[0],
        [{**base, "video_id": 12, "date": 20220409}],
    )
    _write_log(
        tmp_path / data_utils.STANDARD_LOGS[1],
        [{**base, "video_id": 10}, {**base, "video_id": 12}],
    )

    random, standard, scope = data_utils.load_kuairand_buckets(tmp_path)
    assert random["video_id"].tolist() == [10]
    assert standard["video_id"].tolist() == [10]
    assert random["cat1_name"].tolist() == ["KUAIRAND_A"]
    assert standard["cat1_name"].tolist() == ["KUAIRAND_A"]
    assert not random["cat1_name"].str.startswith("WRONG_KUAIREC").any()
    assert scope["n_common_videos"] == 1
    assert scope["date_min"] == 20220422
    assert scope["tab"] == 1
    assert scope["kuairand_category_source_name"] == (
        data_utils.KUAIRAND_CATEGORY_FALLBACK
    )
    assert scope["kuairand_category_join"] == "final_video_id -> video_id"


def test_category_path_environment_override_wins_over_project_fallback(
    tmp_path, monkeypatch
):
    import sys

    code = Path(__file__).resolve().parents[1] / "code"
    sys.path.insert(0, str(code))
    import data_utils

    local_data = tmp_path / "project-data"
    local_data.mkdir()
    monkeypatch.setattr(data_utils, "DATA_DIR", local_data)
    pd.DataFrame(
        {
            "final_video_id": [7],
            "first_level_category_name": ["FALLBACK_CATEGORY"],
        }
    ).to_csv(local_data / data_utils.KUAIRAND_CATEGORY_FALLBACK, index=False)

    configured = tmp_path / "official-kuairand-categories.csv"
    pd.DataFrame(
        {
            "final_video_id": [7],
            "first_level_category_name": ["CONFIGURED_CATEGORY"],
        }
    ).to_csv(configured, index=False)
    monkeypatch.setenv(data_utils.KUAIRAND_CATEGORY_ENV, str(configured))

    resolved = data_utils.resolve_kuairand_category_path()
    mapping = data_utils.load_kuairand_category_mapping()
    assert resolved == configured.resolve()
    assert mapping.to_dict("records") == [
        {"video_id": 7, "cat1_name": "CONFIGURED_CATEGORY"}
    ]


def test_missing_kuairand_mapping_never_falls_back_to_kuairec(
    tmp_path, monkeypatch
):
    import sys

    import pytest

    code = Path(__file__).resolve().parents[1] / "code"
    sys.path.insert(0, str(code))
    import data_utils

    local_data = tmp_path / "project-data"
    local_data.mkdir()
    monkeypatch.setattr(data_utils, "DATA_DIR", local_data)
    monkeypatch.delenv(data_utils.KUAIRAND_CATEGORY_ENV, raising=False)
    pd.DataFrame(
        {"video_id": [1], "cat1_name": ["KUAIREC_ONLY"]}
    ).to_csv(local_data / "kuairec_caption_clean.csv", index=False)

    with pytest.raises(FileNotFoundError, match="KuaiRand category mapping"):
        data_utils.resolve_kuairand_category_path()
