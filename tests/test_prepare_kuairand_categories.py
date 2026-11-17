import csv
import sys
from pathlib import Path


def test_extract_pure_mapping_stops_after_complete_id_set(tmp_path):
    code = Path(__file__).resolve().parents[1] / "code"
    sys.path.insert(0, str(code))
    from prepare_kuairand_categories import extract_pure_mapping

    source = tmp_path / "official.csv"
    fieldnames = ["final_video_id", "first_level_category_name"]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"final_video_id": 0, "first_level_category_name": "A"},
                {"final_video_id": 1, "first_level_category_name": "B"},
                # A malformed trailing row must never affect the completed subset.
                {"final_video_id": "bad", "first_level_category_name": "C"},
            ]
        )
    output = tmp_path / "pure.csv"
    assert extract_pure_mapping(source, output, {0, 1}) == 2
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["final_video_id"] for row in rows] == ["0", "1"]


def test_extract_pure_mapping_fails_closed_when_id_missing(tmp_path):
    import pytest

    code = Path(__file__).resolve().parents[1] / "code"
    sys.path.insert(0, str(code))
    from prepare_kuairand_categories import extract_pure_mapping

    source = tmp_path / "official.csv"
    source.write_text(
        "final_video_id,first_level_category_name\n0,A\n",
        encoding="utf-8",
    )
    output = tmp_path / "pure.csv"
    with pytest.raises(ValueError, match="missing 1"):
        extract_pure_mapping(source, output, {0, 1})
    assert not output.exists()
