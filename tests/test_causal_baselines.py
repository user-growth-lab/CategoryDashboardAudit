from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from causal_baselines import cross_fitted_multitreatment_estimators


def test_cross_fitted_estimators_are_finite_and_distinguish_ht_hajek():
    rng = np.random.default_rng(7)
    rows = []
    for user in range(40):
        p_a = 0.8 if user < 20 else 0.2
        base = (user % 5) * 0.02
        for _ in range(30):
            treatment = "A" if rng.random() < p_a else "B"
            effect = 0.15 if treatment == "A" else -0.05
            rows.append((user, treatment, base + effect + rng.normal(0, 0.01)))
    frame = pd.DataFrame(rows, columns=["user_id", "cat1_name", "watch_ratio"])
    result = cross_fitted_multitreatment_estimators(frame, propensity_floor=0.02)

    for name in ["ht", "hajek", "aipw"]:
        assert list(result[name].index) == ["A", "B"]
        assert np.isfinite(result[name].to_numpy()).all()
    assert not np.allclose(result["ht"], result["hajek"])
    assert result["diagnostics"]["n_folds"] == 2


def test_rejects_invalid_propensity_floor():
    frame = pd.DataFrame(
        {"user_id": [1, 1], "cat1_name": ["A", "B"], "watch_ratio": [0.1, 0.2]}
    )
    try:
        cross_fitted_multitreatment_estimators(frame, propensity_floor=0)
    except ValueError as exc:
        assert "propensity_floor" in str(exc)
    else:
        raise AssertionError("invalid propensity floor was accepted")

