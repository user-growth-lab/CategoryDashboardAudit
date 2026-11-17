"""Reference propensity estimators used by the KuaiRand audit.

These estimators target an exposure-weighted production-user population and
adjust for user identity only.  They are diagnostics, not a claim that
conditional exchangeability holds: item composition, time, and unobserved
user--category match can still invalidate the estimand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _within_user_folds(users: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    """Deterministically distribute each user's rows across folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    occurrence = users.groupby(users, sort=False).cumcount().to_numpy()
    user_hash = pd.util.hash_pandas_object(users, index=False).to_numpy(np.uint64)
    offset = (user_hash + np.uint64(seed)) % np.uint64(n_folds)
    return ((occurrence + offset.astype(np.int64)) % n_folds).astype(np.int8)


def _fit_additive_outcome(
    train: pd.DataFrame, user_col: str, treatment_col: str, outcome_col: str
) -> tuple[float, pd.Series, pd.Series]:
    """Fit a transparent additive user + category outcome regression."""
    grand = float(train[outcome_col].mean())
    user_effect = train.groupby(user_col)[outcome_col].mean() - grand
    residual = train[outcome_col] - train[user_col].map(user_effect) - grand
    category_effect = residual.groupby(train[treatment_col]).mean()
    return grand, user_effect, category_effect


def cross_fitted_multitreatment_estimators(
    frame: pd.DataFrame,
    *,
    user_col: str = "user_id",
    treatment_col: str = "cat1_name",
    outcome_col: str = "watch_ratio",
    categories: list[str] | None = None,
    n_folds: int = 2,
    propensity_prior_strength: float = 10.0,
    propensity_floor: float = 0.01,
    seed: int = 42,
) -> dict:
    """Compute cross-fitted HT, Hájek, and AIPW category means.

    Propensities are smoothed empirical ``P(category | user)`` estimates from
    the training fold.  The AIPW outcome model is an additive user/category
    regression trained on the opposite fold.  All averages target the row-
    weighted population represented by ``frame``.
    """
    required = {user_col, treatment_col, outcome_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")
    if not 0 < propensity_floor < 1:
        raise ValueError("propensity_floor must lie in (0, 1)")
    if propensity_prior_strength < 0:
        raise ValueError("propensity_prior_strength must be non-negative")

    data = frame[[user_col, treatment_col, outcome_col]].dropna().reset_index(drop=True)
    cats = categories or sorted(data[treatment_col].unique().tolist())
    cat_to_index = {cat: idx for idx, cat in enumerate(cats)}
    unknown = set(data[treatment_col].unique()).difference(cat_to_index)
    if unknown:
        raise ValueError(f"categories omit observed treatments: {sorted(unknown)}")

    folds = _within_user_folds(data[user_col], n_folds, seed)
    k = len(cats)
    ht_sum = np.zeros(k)
    hajek_y_sum = np.zeros(k)
    hajek_w_sum = np.zeros(k)
    aipw_sum = np.zeros(k)
    weight_sq_sum = np.zeros(k)
    clipped = np.zeros(k, dtype=np.int64)
    observed = np.zeros(k, dtype=np.int64)

    for fold in range(n_folds):
        train = data.loc[folds != fold]
        test = data.loc[folds == fold]
        if train.empty or test.empty:
            raise ValueError("each cross-fitting fold must have train and test rows")

        global_prop = train[treatment_col].value_counts(normalize=True).reindex(cats)
        global_prop = global_prop.fillna(1.0 / k)
        user_total = train.groupby(user_col).size()
        user_cat = train.groupby([user_col, treatment_col]).size()

        grand, user_effect, category_effect = _fit_additive_outcome(
            train, user_col, treatment_col, outcome_col
        )
        category_effect = category_effect.reindex(cats).fillna(0.0)
        test_user_effect = test[user_col].map(user_effect).fillna(0.0).to_numpy()
        baseline_common = float((grand + test_user_effect).sum())
        aipw_sum += baseline_common + len(test) * category_effect.to_numpy()

        test_users = test[user_col].to_numpy()
        test_cats = test[treatment_col].to_numpy()
        test_y = test[outcome_col].to_numpy(float)
        train_user_n = test[user_col].map(user_total).fillna(0.0).to_numpy(float)

        raw_cell_n = np.array(
            [float(user_cat.get((u, c), 0.0)) for u, c in zip(test_users, test_cats)]
        )
        prior = np.array([global_prop[c] for c in test_cats], dtype=float)
        propensity = (
            raw_cell_n + propensity_prior_strength * prior
        ) / (train_user_n + propensity_prior_strength)
        was_clipped = propensity < propensity_floor
        propensity = np.maximum(propensity, propensity_floor)
        weights = 1.0 / propensity

        cat_index = np.array([cat_to_index[c] for c in test_cats], dtype=int)
        m_observed = (
            grand
            + test_user_effect
            + category_effect.reindex(test_cats).to_numpy()
        )
        correction = weights * (test_y - m_observed)

        np.add.at(ht_sum, cat_index, weights * test_y)
        np.add.at(hajek_y_sum, cat_index, weights * test_y)
        np.add.at(hajek_w_sum, cat_index, weights)
        np.add.at(aipw_sum, cat_index, correction)
        np.add.at(weight_sq_sum, cat_index, weights**2)
        np.add.at(clipped, cat_index, was_clipped.astype(np.int64))
        np.add.at(observed, cat_index, 1)

    n = len(data)
    ht = pd.Series(ht_sum / n, index=cats, name="ht")
    hajek = pd.Series(hajek_y_sum / hajek_w_sum, index=cats, name="hajek")
    aipw = pd.Series(aipw_sum / n, index=cats, name="aipw")
    ess = pd.Series(
        np.divide(
            hajek_w_sum**2,
            weight_sq_sum,
            out=np.zeros_like(hajek_w_sum),
            where=weight_sq_sum > 0,
        ),
        index=cats,
    )
    return {
        "ht": ht,
        "hajek": hajek,
        "aipw": aipw,
        "diagnostics": {
            "n_rows": int(n),
            "n_categories": int(k),
            "n_folds": int(n_folds),
            "propensity_prior_strength": float(propensity_prior_strength),
            "propensity_floor": float(propensity_floor),
            "clipped_fraction": pd.Series(clipped / observed, index=cats),
            "effective_sample_size": ess,
        },
    }

