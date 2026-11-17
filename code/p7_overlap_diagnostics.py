"""P7: overlap, residual, and observational-baseline diagnostics.

The stage reports randomized-insertion exposure support, associations between
category residuals and allocation/sparsity summaries, and user-identity-only
HT, Hajek, and AIPW specifications beside naive and user-demeaned estimates.
The diagnostics do not attribute unexplained residuals to a unique mechanism.
"""
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from causal_baselines import cross_fitted_multitreatment_estimators
from audit_metrics import (
    CENTER_DEFINITION,
    evaluate_category_estimate,
    paired_user_audit_scope,
    randomized_reference,
    user_demeaned_category_estimate,
)
from data_utils import RESULTS_DIR, load_kuairand_buckets

THR = 5
PROPENSITY_FLOORS = (0.01, 0.02, 0.05)


def spearman(a, b):
    ra = pd.Series(np.asarray(a)).rank().to_numpy()
    rb = pd.Series(np.asarray(b)).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    out = {}
    print("[1/5] loading buckets ...")
    rnd_base, std_base, base_scope = load_kuairand_buckets(
        matched_period=True, tab=1, common_videos=True
    )
    rnd, std, primary_scope = paired_user_audit_scope(rnd_base, std_base, THR)
    out["base_sample_scope"] = base_scope
    out["primary_paired_scope"] = primary_scope

    common = sorted(set(rnd["cat1_name"]).intersection(std["cat1_name"]))
    gt, share_r, shift_align = randomized_reference(rnd, common)
    std_u = std
    share_s = std_u["cat1_name"].value_counts(normalize=True).reindex(common)
    shift = share_s - share_r

    # Randomized-insertion exposure support.
    print("[2/5] exposure diagnostics on random bucket ...")
    per_cat_exp = rnd["cat1_name"].value_counts()[common]
    per_vid_exp = rnd.groupby("video_id").size()
    out["random_bucket_exposure"] = {
        "per_category": {"min": int(per_cat_exp.min()),
                         "median": int(per_cat_exp.median()),
                         "max": int(per_cat_exp.max()),
                         "n_categories": int(len(per_cat_exp)),
                         "min_category": str(per_cat_exp.idxmin())},
        "per_video": {"median": int(per_vid_exp.median()),
                      "p10": int(per_vid_exp.quantile(0.10)),
                      "n_videos": int(len(per_vid_exp))},
    }

    # User-category cell summaries on the selected paired support.
    print("[3/5] building user-category cells ...")
    grp = (std_u.groupby(["user_id", "cat1_name"])["watch_ratio"]
           .agg(["mean", "count", "sum"]).reset_index())
    user_sum = grp.groupby("user_id")["sum"].transform("sum")
    user_cnt = grp.groupby("user_id")["count"].transform("sum")
    grp["demeaned"] = grp["mean"] - user_sum / user_cnt
    deb, _ = user_demeaned_category_estimate(std_u, common, shift_align)
    naive = std_u.groupby("cat1_name")["watch_ratio"].mean().reindex(common)
    gt_c = gt
    resid = (deb - gt_c)

    # Residual association diagnostics.
    print("[4/5] residual association diagnostics ...")
    n_users_cell = grp.groupby("cat1_name")["user_id"].nunique()[common]
    mean_cell = grp.groupby("cat1_name")["count"].mean()[common]
    cell_dm_std = grp.groupby("cat1_name")["demeaned"].std()[common]
    corr_shift = spearman(resid, shift[common])
    corr_freq = spearman(resid, per_cat_exp)
    corr_users = spearman(resid, n_users_cell)
    # Descriptive linear fit: r_c ~ shift + log(freq).
    X = np.column_stack([shift[common].to_numpy(),
                         np.log(per_cat_exp.to_numpy() + 1),
                         np.ones(len(common))])
    rvec = resid.to_numpy()
    beta = np.linalg.lstsq(X, rvec, rcond=None)[0]
    pred = X @ beta
    r2 = 1 - ((rvec - pred) ** 2).sum() / ((rvec - rvec.mean()) ** 2).sum()
    out["residual_diagnostics"] = {
        "n_categories": int(len(common)),
        "resid_mae": round(float(resid.abs().mean()), 4),
        "spearman_resid_vs_exposure_shift": round(corr_shift, 3),
        "spearman_resid_vs_cat_frequency": round(corr_freq, 3),
        "spearman_resid_vs_n_users": round(corr_users, 3),
        "r2_linear_on_shift_logfreq": round(float(r2), 3),
        "top_abs_resid": resid.abs().sort_values(ascending=False).head(3)
                          .round(4).to_dict(),
        "cell_dm_std_mean": round(float(cell_dm_std.mean()), 4),
    }

    # Cross-fitted IPW and AIPW specifications.
    print("[5/5] cross-fitted IPW / AIPW baselines ...")

    def eval_est(series):
        raw = evaluate_category_estimate(
            series.reindex(common), gt_c, share_r, topk=10
        )
        return {
            "mae": round(raw["mae"], 4),
            "centered_shape_mae": round(raw["centered_shape_mae"], 4),
            "max_abs_error": round(raw["max_abs_error"], 4),
            "rank_corr": round(raw["rank_corr"], 4),
            "top10_overlap": round(raw["top10_overlap"], 2),
        }

    baseline_runs = {}
    primary = None
    for floor in PROPENSITY_FLOORS:
        estimates = cross_fitted_multitreatment_estimators(
            std_u,
            categories=list(common),
            n_folds=2,
            propensity_prior_strength=10.0,
            propensity_floor=floor,
            seed=42,
        )
        diagnostics = estimates["diagnostics"]
        baseline_runs[f"floor_{floor:.2f}"] = {
            "ht": eval_est(estimates["ht"]),
            "hajek": eval_est(estimates["hajek"]),
            "aipw": eval_est(estimates["aipw"]),
            "clipped_fraction_mean": round(
                float(diagnostics["clipped_fraction"].mean()), 4
            ),
            "effective_sample_size_min": round(
                float(diagnostics["effective_sample_size"].min()), 1
            ),
        }
        if floor == 0.01:
            primary = estimates

    out["baselines"] = {
        "input_cohort": (
            "paired random/production users selected by at least 5 production "
            "categories on exact paired video support"
        ),
        "primary_scope_id": primary_scope["primary_scope_id"],
        "aggregation_warning": (
            "Naive/HT/Hajek/AIPW are row-weighted; user demeaning averages "
            "observed user-category cells equally within category. The table "
            "shares an input cohort and reference, not one aggregation target."
        ),
        "centered_shape_definition": CENTER_DEFINITION,
        "n_eligible_users": int(std_u["user_id"].nunique()),
        "n_standard_eligible_before_pairing": int(
            primary_scope["standard_eligible_users_before_random_intersection"]
        ),
        "eligible_users_missing_from_random": int(
            primary_scope["eligible_users_missing_from_random"]
        ),
        "adjustment_set": "user identity only",
        "identification_warning": (
            "Diagnostic only: user-conditional exchangeability and positivity may fail; "
            "item composition and user-category match remain unobserved."
        ),
        "naive_paired_users": eval_est(naive),
        "ht_ipw_cross_fitted": eval_est(primary["ht"]),
        "hajek_ipw_cross_fitted": eval_est(primary["hajek"]),
        "aipw_cross_fitted": eval_est(primary["aipw"]),
        "user_demeaned_random_mean_calibrated": eval_est(deb),
        "propensity_floor_sensitivity": baseline_runs,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "p7_overlap_diagnostics.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("=== P7 OVERLAP DIAGNOSTICS DONE ===")


if __name__ == "__main__":
    main()
