"""Shared data loading and sample-scope utilities.

The KuaiRand logs are not redistributed with this project.  Set
``KUAIRAND_ROOT`` to the directory containing the three log CSV files and the
video feature file.  First-level categories come from KuaiRand's own
official supplementary mapping, resolved from ``KUAIRAND_CATEGORY_PATH`` or
``data/kuairand_pure_video_categories.csv``.  All KuaiRand experiments use the
same auditable scope by default:

* dates observed in the randomized bucket (2022-04-22--2022-05-08),
* recommendation surface ``tab == 1``, and
* videos observed in both the randomized and production buckets.

The common-video restriction does not make category an item-level randomized
treatment.  It only makes the two category-mix summaries comparable on a
shared support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_LOG = "log_random_4_22_to_5_08_pure.csv"
STANDARD_LOGS = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
VIDEO_FEATURES = "video_features_basic_pure.csv"
KUAIRAND_CATEGORY_ENV = "KUAIRAND_CATEGORY_PATH"
KUAIRAND_CATEGORY_FALLBACK = "kuairand_pure_video_categories.csv"
KUAIRAND_CATEGORY_ID_COLUMN = "final_video_id"
KUAIRAND_CATEGORY_NAME_COLUMN = "first_level_category_name"


def resolve_kuairand_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve and validate the external KuaiRand-Pure data directory."""
    raw = explicit or os.environ.get("KUAIRAND_ROOT")
    if not raw:
        bundled = DATA_DIR / "KuaiRand-Pure" / "data"
        if bundled.is_dir():
            raw = bundled
    if not raw:
        raise FileNotFoundError(
            "KuaiRand-Pure data not found. Set KUAIRAND_ROOT to the directory "
            "containing log_random_4_22_to_5_08_pure.csv."
        )
    root = Path(raw).expanduser().resolve()
    required = (RANDOM_LOG, *STANDARD_LOGS, VIDEO_FEATURES)
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"KUAIRAND_ROOT={root} is missing: {', '.join(missing)}"
        )
    return root


def resolve_kuairand_category_path(
    explicit: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve KuaiRand's official supplementary video-category mapping.

    KuaiRec and KuaiRand independently re-index their video IDs.  Consequently,
    a KuaiRec category table must never be used as a fallback for KuaiRand.
    """
    raw = explicit or os.environ.get(KUAIRAND_CATEGORY_ENV)
    path = (
        Path(raw).expanduser()
        if raw
        else DATA_DIR / KUAIRAND_CATEGORY_FALLBACK
    )
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "KuaiRand category mapping not found. Set "
            f"{KUAIRAND_CATEGORY_ENV} to the official supplementary "
            "kuairand_video_categories.csv file, or place it at "
            f"{DATA_DIR / KUAIRAND_CATEGORY_FALLBACK}."
        )
    return path


def load_kuairand_category_mapping(
    path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Load and validate the official KuaiRand first-level category mapping."""
    source = resolve_kuairand_category_path(path)
    categories = pd.read_csv(
        source,
        usecols=[KUAIRAND_CATEGORY_ID_COLUMN, KUAIRAND_CATEGORY_NAME_COLUMN],
    ).rename(
        columns={
            KUAIRAND_CATEGORY_ID_COLUMN: "video_id",
            KUAIRAND_CATEGORY_NAME_COLUMN: "cat1_name",
        }
    )

    video_ids = pd.to_numeric(categories["video_id"], errors="coerce")
    invalid_ids = video_ids.isna() | (video_ids % 1 != 0)
    if invalid_ids.any():
        raise ValueError(
            f"{source.name} contains {int(invalid_ids.sum())} invalid "
            f"{KUAIRAND_CATEGORY_ID_COLUMN} values"
        )
    categories["video_id"] = video_ids.astype("int64")
    categories["cat1_name"] = categories["cat1_name"].astype("string").str.strip()
    categories = categories[
        categories["cat1_name"].notna()
        & ~categories["cat1_name"].isin(["", "UNKNOWN", "MISSING", "-124"])
    ].copy()
    if categories.empty:
        raise ValueError(f"{source.name} contains no valid first-level categories")

    duplicate_ids = categories["video_id"].duplicated(keep=False)
    if duplicate_ids.any():
        examples = sorted(categories.loc[duplicate_ids, "video_id"].unique())[:5]
        raise ValueError(
            f"{source.name} has duplicate {KUAIRAND_CATEGORY_ID_COLUMN} values; "
            f"examples: {examples}"
        )
    return categories


def _read_log(path: Path, extra_cols: Iterable[str] = ()) -> pd.DataFrame:
    base = [
        "user_id",
        "video_id",
        "date",
        "tab",
        "play_time_ms",
        "duration_ms",
    ]
    usecols = list(dict.fromkeys([*base, *extra_cols]))
    frame = pd.read_csv(path, usecols=usecols)
    frame = frame[frame["duration_ms"] > 0].copy()
    frame["watch_ratio"] = (
        frame["play_time_ms"] / frame["duration_ms"]
    ).clip(lower=0.0, upper=1.0)
    return frame


def _attach_categories(
    frame: pd.DataFrame, categories: pd.DataFrame
) -> pd.DataFrame:
    """Attach KuaiRand's valid first-level category.

    The legacy ``tag`` field is not part of the estimand and is deliberately
    not used as an eligibility filter.  Official category availability alone
    determines content-side eligibility.
    """
    out = frame.merge(
        categories, on="video_id", how="left", validate="many_to_one"
    )
    out = out.dropna(subset=["cat1_name"])
    return out.copy()


def load_kuairand_buckets(
    root: str | os.PathLike[str] | None = None,
    *,
    category_path: str | os.PathLike[str] | None = None,
    matched_period: bool = True,
    tab: int | None = 1,
    common_videos: bool = True,
    extra_cols: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load randomized and production buckets under one explicit scope.

    Returns ``(random, standard, scope_metadata)``.  Filtering occurs before
    category attachment; the final common-video restriction occurs after it.
    """
    kr_root = resolve_kuairand_root(root)
    resolved_category_path = resolve_kuairand_category_path(category_path)
    categories = load_kuairand_category_mapping(resolved_category_path)
    random = _read_log(kr_root / RANDOM_LOG, extra_cols)
    standard = pd.concat(
        [_read_log(kr_root / name, extra_cols) for name in STANDARD_LOGS],
        ignore_index=True,
    )

    random_dates = sorted(int(x) for x in random["date"].dropna().unique())
    if matched_period:
        date_min, date_max = min(random_dates), max(random_dates)
        random = random[random["date"].between(date_min, date_max)]
        standard = standard[standard["date"].between(date_min, date_max)]
    else:
        date_min = date_max = None

    if tab is not None:
        random = random[random["tab"] == tab]
        standard = standard[standard["tab"] == tab]

    random = _attach_categories(random, categories)
    standard = _attach_categories(standard, categories)

    if common_videos:
        shared = set(random["video_id"].unique()).intersection(
            standard["video_id"].unique()
        )
        random = random[random["video_id"].isin(shared)]
        standard = standard[standard["video_id"].isin(shared)]
    else:
        shared = set()

    common_categories = sorted(
        set(random["cat1_name"].unique()).intersection(
            standard["cat1_name"].unique()
        )
    )
    random = random[random["cat1_name"].isin(common_categories)].copy()
    standard = standard[standard["cat1_name"].isin(common_categories)].copy()

    metadata = {
        "kuairand_root_env": "KUAIRAND_ROOT",
        "kuairand_category_path_env": KUAIRAND_CATEGORY_ENV,
        "kuairand_category_source_name": resolved_category_path.name,
        "kuairand_category_join": (
            f"{KUAIRAND_CATEGORY_ID_COLUMN} -> video_id"
        ),
        "kuairand_category_field": (
            f"{KUAIRAND_CATEGORY_NAME_COLUMN} -> cat1_name"
        ),
        "kuairand_category_mapping_rows": int(len(categories)),
        "matched_period": bool(matched_period),
        "date_min": date_min,
        "date_max": date_max,
        "tab": tab,
        "common_videos_only": bool(common_videos),
        "n_common_videos": int(len(shared)) if common_videos else None,
        "n_common_categories": int(len(common_categories)),
        "random_rows": int(len(random)),
        "standard_rows": int(len(standard)),
        "random_users": int(random["user_id"].nunique()),
        "standard_users": int(standard["user_id"].nunique()),
        "random_videos": int(random["video_id"].nunique()),
        "standard_videos": int(standard["video_id"].nunique()),
    }
    return random.reset_index(drop=True), standard.reset_index(drop=True), metadata
