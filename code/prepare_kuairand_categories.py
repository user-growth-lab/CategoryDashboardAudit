"""Extract KuaiRand-Pure rows from the official category supplement.

The official supplement uses ``final_video_id``.  KuaiRand-Pure retains 7,583
candidate-pool IDs, which are listed in ``video_features_basic_pure.csv``.
This script streams the (potentially much larger) official CSV and writes only
those rows to the project fallback consumed by ``data_utils.py``.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import pandas as pd

from data_utils import (
    DATA_DIR,
    KUAIRAND_CATEGORY_FALLBACK,
    KUAIRAND_CATEGORY_ID_COLUMN,
    VIDEO_FEATURES,
    resolve_kuairand_root,
)


def extract_pure_mapping(source: Path, output: Path, pure_ids: set[int]) -> int:
    """Stream ``source`` and atomically write the rows in ``pure_ids``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        if reader.fieldnames is None or KUAIRAND_CATEGORY_ID_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"source is missing {KUAIRAND_CATEGORY_ID_COLUMN}"
            )
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        found: set[int] = set()
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as out_handle:
                writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    try:
                        video_id = int(float(row[KUAIRAND_CATEGORY_ID_COLUMN]))
                    except (TypeError, ValueError):
                        continue
                    if video_id in pure_ids:
                        if video_id in found:
                            raise ValueError(f"duplicate final_video_id: {video_id}")
                        writer.writerow(row)
                        found.add(video_id)
                        if found == pure_ids:
                            break
            missing = pure_ids.difference(found)
            if missing:
                raise ValueError(
                    f"official mapping is missing {len(missing)} KuaiRand-Pure IDs"
                )
            os.replace(temporary_name, output)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
    return len(found)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source", type=Path,
        help="downloaded official kuairand_video_categories.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DATA_DIR / KUAIRAND_CATEGORY_FALLBACK,
    )
    args = parser.parse_args()
    root = resolve_kuairand_root()
    pure_ids = set(
        pd.read_csv(root / VIDEO_FEATURES, usecols=["video_id"])["video_id"]
        .astype(int)
        .tolist()
    )
    count = extract_pure_mapping(args.source.resolve(), args.output.resolve(), pure_ids)
    print(f"wrote {count:,} KuaiRand-Pure category rows -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
