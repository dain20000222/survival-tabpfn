import os
import numpy as np
import pandas as pd

from sksurv.util import Surv
from sksurv.metrics import (
    brier_score,
    cumulative_dynamic_auc,
    concordance_index_censored,  # Harrell's C
    concordance_index_ipcw,  # IPCW C-index
)

# -----------------------------------------------------------
# 1. Helper: read survival dataset and collapse to (T_i, δ_i)
# -----------------------------------------------------------

def load_survival_dataset(path):
    df = pd.read_csv(path)
    pat_df = (
        df.groupby("pid", as_index=False)
          .agg(time=("time2", "max"),
               event=("event", "max"))
    )
    return pat_df


# -----------------------------------------------------------
# 2. Dynamic Brier score: BS(t | tau_landmark)
# -----------------------------------------------------------

def dynamic_brier(
    survival_train,
    survival_test,
    pids_test,
    pred_df,
    tau_landmark,
    eval_times
):
    eval_times = np.asarray(eval_times, dtype=float)
    eval_times = np.sort(eval_times)

    # test subjects alive at τ
    at_risk_test = survival_test["time"] > tau_landmark
    y_test_k = survival_test[at_risk_test]
    pids_at_risk = pids_test[at_risk_test]

    # training subjects alive at τ (for conditional censoring)
    at_risk_train = survival_train["time"] > tau_landmark
    y_train_k = survival_train[at_risk_train]

    # select predictions for those subjects and future times
    mask_pred = (
        pred_df["pid"].isin(pids_at_risk)
        & pred_df["eval_time"].isin(eval_times)
        & (pred_df["eval_time"] > tau_landmark)
    )
    df_k = pred_df.loc[mask_pred, ["pid", "eval_time", "surv_prob"]]

    pivot = (
        df_k.pivot(index="pid", columns="eval_time", values="surv_prob")
            .reindex(index=pids_at_risk, columns=eval_times)
    )
    # drop columns with all-NaN (no predictions at that time)
    pivot = pivot.loc[:, pivot.notna().any(axis=0)]
    times_used = pivot.columns.to_numpy(dtype=float)
    estimate = pivot.to_numpy()

    if estimate.size == 0:
        return np.array([]), np.array([])

    times_out, bs_values = brier_score(
        survival_train=y_train_k,
        survival_test=y_test_k,
        estimate=estimate,
        times=times_used,
    )
    return times_out, bs_values


# -----------------------------------------------------------
# 3. Dynamic AUC: AUC(t | tau_landmark) with NaN on censoring failure
# -----------------------------------------------------------

def dynamic_auc(
    survival_train,
    survival_test,
    pids_test,
    pred_df,
    tau_landmark,
    eval_times
):
    eval_times = np.asarray(eval_times, dtype=float)
    eval_times = np.sort(eval_times)

    # test subjects alive at τ
    at_risk_test = survival_test["time"] > tau_landmark
    y_test_k = survival_test[at_risk_test]
    pids_at_risk = pids_test[at_risk_test]

    # training subjects alive at τ (for conditional censoring)
    at_risk_train = survival_train["time"] > tau_landmark
    y_train_k = survival_train[at_risk_train]

    # select predictions for those subjects and future times
    mask_pred = (
        pred_df["pid"].isin(pids_at_risk)
        & pred_df["eval_time"].isin(eval_times)
        & (pred_df["eval_time"] > tau_landmark)
    )
    df_k = pred_df.loc[mask_pred, ["pid", "eval_time", "risk"]]

    pivot = (
        df_k.pivot(index="pid", columns="eval_time", values="risk")
            .reindex(index=pids_at_risk, columns=eval_times)
    )
    pivot = pivot.loc[:, pivot.notna().any(axis=0)]
    times_used = pivot.columns.to_numpy(dtype=float)
    estimate = pivot.to_numpy()

    if estimate.size == 0:
        # return NaNs for requested times (keeps shape for caller)
        return times_used, np.full(len(times_used), np.nan)

    try:
        auc_values, mean_auc = cumulative_dynamic_auc(
            survival_train=y_train_k,
            survival_test=y_test_k,
            estimate=estimate,
            times=times_used,
        )
        return times_used, auc_values
    except ValueError as e:
        if "censoring survival function is zero" in str(e) or \
           "time must be smaller than largest observed time point" in str(e):
            return times_used, np.full(len(times_used), np.nan)
        else:
            raise


# -----------------------------------------------------------
# 4. Dynamic landmark C-index using Harrell's C
# -----------------------------------------------------------

def dynamic_cindex_landmark(
    survival_train,   # kept for compatibility, not used with Harrell's C
    survival_test,
    pids_test,
    pred_df,
    tau_landmark,
):
    """
    Landmarked Harrell C-index at τ = tau_landmark:

        C_Harrell(τ) = P( higher risk at τ for the subject who fails first after τ )

    Implementation:
      - restrict test to subjects with T > τ (alive at τ)
      - use risk scores at eval_time == τ
      - compute Harrell's C on these truncated test outcomes
    """

    # 1) test subjects alive at τ
    at_risk_test = survival_test["time"] > tau_landmark
    y_test_k = survival_test[at_risk_test]
    pids_at_risk = pids_test[at_risk_test]

    if len(pids_at_risk) == 0:
        return np.nan

    # 2) risk scores at landmark τ (NOT at future t)
    df_k = pred_df[
        (pred_df["pid"].isin(pids_at_risk)) &
        (pred_df["eval_time"] == float(tau_landmark))
    ][["pid", "risk"]]

    if df_k.empty:
        return np.nan

    risk_series = df_k.set_index("pid")["risk"].reindex(pids_at_risk)
    estimate = risk_series.to_numpy()

    # if any missing risk, bail out
    if np.any(pd.isna(estimate)):
        return np.nan

    # 3) Harrell's C on truncated test data
    event_indicator = y_test_k["event"].astype(bool)
    event_time = y_test_k["time"]

    try:
        cindex, concordant, discordant, tied_risk, tied_time = (
            concordance_index_censored(
                event_indicator=event_indicator,
                event_time=event_time,
                estimate=estimate,
            )
        )
        return cindex
    except ValueError:
        # degenerate cases (e.g., no comparable pairs)
        return np.nan


# -----------------------------------------------------------
# 5. Dynamic IPCW C-index: C_IPCW(t | tau_landmark)
# -----------------------------------------------------------

def dynamic_cindex_ipcw(
    survival_train,
    survival_test,
    pids_test,
    pred_df,
    tau_landmark,
    eval_time
):
    """
    IPCW C-index: C_IPCW(t | τ)
    
    Measures concordance at evaluation time t, given survival past τ.
    
    Formula:
        C_IPCW(t | τ) = P(r_i(t) > r_j(t) | τ < T_i ≤ t < T_j)
    
    where:
        - τ = tau_landmark (conditioning time)
        - t = eval_time (evaluation/prediction horizon)
        - r_i(t) = risk score for patient i at time t
        - Weights adjusted by censoring distribution G(T_i | τ)
    
    Parameters:
    -----------
    survival_train : structured array
        Training survival data (for censoring distribution estimation)
    survival_test : structured array
        Test survival data
    pids_test : array
        Test patient IDs
    pred_df : DataFrame
        Predictions with columns: pid, eval_time, risk
    tau_landmark : float
        Landmark time (conditioning time τ)
    eval_time : float
        Evaluation/prediction horizon time t (must be > tau_landmark)
    
    Returns:
    --------
    float : C-index value or np.nan if cannot be computed
    """
    
    if eval_time <= tau_landmark:
        return np.nan
    
    # 1) Subset to test subjects alive at τ (T > τ)
    at_risk_test = survival_test["time"] > tau_landmark
    y_test_k = survival_test[at_risk_test]
    pids_at_risk = pids_test[at_risk_test]
    
    if len(pids_at_risk) == 0:
        return np.nan
    
    # 2) Subset training subjects alive at τ (for conditional censoring distribution)
    at_risk_train = survival_train["time"] > tau_landmark
    y_train_k = survival_train[at_risk_train]
    
    if len(y_train_k) == 0:
        return np.nan
    
    # 3) Get risk scores at eval_time for patients alive at τ
    df_k = pred_df[
        (pred_df["pid"].isin(pids_at_risk)) &
        (pred_df["eval_time"] == float(eval_time))
    ][["pid", "risk"]]
    
    if df_k.empty:
        return np.nan
    
    # Align risk scores with pids_at_risk
    risk_series = df_k.set_index("pid")["risk"].reindex(pids_at_risk)
    estimate = risk_series.to_numpy()
    
    # If any missing risk scores, bail out
    if np.any(pd.isna(estimate)):
        return np.nan
    
    # 4) Call sksurv's IPCW C-index
    try:
        cindex, concordant, discordant, tied_risk, tied_time = concordance_index_ipcw(
            survival_train=y_train_k,
            survival_test=y_test_k,
            estimate=estimate,
            tau=None
        )
        return cindex
    except ValueError as e:
        # Handle cases like insufficient data, censoring issues, etc.
        if "censoring survival function is zero" in str(e) or \
           "time must be smaller than largest observed time point" in str(e):
            return np.nan
        else:
            # Re-raise unexpected errors
            raise


# -----------------------------------------------------------
# 6. Loop through all models & datasets and collect scores
# -----------------------------------------------------------

data_dir = "data"
rows = []

models = [
    ("tabpfn", "dynamic_tabpfn_risks.csv"),
    ("landmark_cox", "landmark_cox_risk.csv"),
    ("deephit", "dynamic_deephit_risk.csv"),
]

for model_name, pred_path in models:
    print(f"\n=== Processing model: {model_name} ({pred_path}) ===")

    if not os.path.exists(pred_path):
        print(f"WARNING: missing predictions file {pred_path}, skipping model.")
        continue

    # load predictions for this model
    pred_df = pd.read_csv(pred_path)
    pred_df["eval_time"] = pred_df["eval_time"].astype(float)
    datasets = pred_df["dataset"].unique()

    for dname in datasets:
        print(f"\n### Processing dataset: {dname} (model: {model_name})")

        csv_path = os.path.join(data_dir, f"{dname}.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: missing file {csv_path}, skipping dataset.")
            continue

        pat_df = load_survival_dataset(csv_path)

        pred_df_ds = pred_df[pred_df["dataset"] == dname].copy()
        test_pids = np.sort(pred_df_ds["pid"].unique())

        is_test = pat_df["pid"].isin(test_pids)
        pat_test = pat_df[is_test].copy()
        pat_train = pat_df[~is_test].copy()

        if pat_test.empty or pat_train.empty:
            print(f"WARNING: no test or no train patients for {dname}. Skipping.")
            continue

        y_train = Surv.from_arrays(
            event=pat_train["event"].astype(bool).values,
            time=pat_train["time"].values,
        )
        y_test = Surv.from_arrays(
            event=pat_test["event"].astype(bool).values,
            time=pat_test["time"].values,
        )

        pids_train = pat_train["pid"].values
        pids_test = pat_test["pid"].values

        taus = np.sort(pred_df_ds["eval_time"].unique())
        if len(taus) < 3:
            print(f"Not enough τ for dataset {dname}. Need ≥ 3.")
            continue

        τ1, τ2, τ3 = taus[:3]

        # ----- Brier -----
        times_b1, bs_1 = dynamic_brier(
            survival_train=y_train,
            survival_test=y_test,
            pids_test=pids_test,
            pred_df=pred_df_ds,
            tau_landmark=τ1,
            eval_times=[τ2, τ3],
        )
        times_b2, bs_2 = dynamic_brier(
            survival_train=y_train,
            survival_test=y_test,
            pids_test=pids_test,
            pred_df=pred_df_ds,
            tau_landmark=τ2,
            eval_times=[τ3],
        )

        if bs_1.size > 0:
            rows.append({"model": model_name, "dataset": dname, "metric": "brier",
                         "tau_start": τ1, "tau_end": τ2, "value": bs_1[0]})
        if bs_1.size > 1:
            rows.append({"model": model_name, "dataset": dname, "metric": "brier",
                         "tau_start": τ1, "tau_end": τ3, "value": bs_1[1]})
        if bs_2.size > 0:
            rows.append({"model": model_name, "dataset": dname, "metric": "brier",
                         "tau_start": τ2, "tau_end": τ3, "value": bs_2[0]})

        # ----- AUC -----
        times_a1, auc_1 = dynamic_auc(
            survival_train=y_train,
            survival_test=y_test,
            pids_test=pids_test,
            pred_df=pred_df_ds,
            tau_landmark=τ1,
            eval_times=[τ2, τ3],
        )
        times_a2, auc_2 = dynamic_auc(
            survival_train=y_train,
            survival_test=y_test,
            pids_test=pids_test,
            pred_df=pred_df_ds,
            tau_landmark=τ2,
            eval_times=[τ3],
        )

        if auc_1.size > 0:
            rows.append({"model": model_name, "dataset": dname, "metric": "auc",
                         "tau_start": τ1, "tau_end": τ2, "value": auc_1[0]})
        if auc_1.size > 1:
            rows.append({"model": model_name, "dataset": dname, "metric": "auc",
                         "tau_start": τ1, "tau_end": τ3, "value": auc_1[1]})
        if auc_2.size > 0:
            rows.append({"model": model_name, "dataset": dname, "metric": "auc",
                         "tau_start": τ2, "tau_end": τ3, "value": auc_2[0]})

        # ----- Landmark Harrell C(τ_k) -----
        c_τ1 = dynamic_cindex_landmark(
            survival_train=y_train,
            survival_test=y_test,
            pids_test=pids_test,
            pred_df=pred_df_ds,
            tau_landmark=τ1,
        )
        c_τ2 = dynamic_cindex_landmark(
            survival_train=y_train,
            survival_test=y_test,
            pids_test=pids_test,
            pred_df=pred_df_ds,
            tau_landmark=τ2,
        )
        c_τ3 = dynamic_cindex_landmark(
            survival_train=y_train,
            survival_test=y_test,
            pids_test=pids_test,
            pred_df=pred_df_ds,
            tau_landmark=τ3,
        )

        # store as tau_start = tau_end = τ_k to keep schema
        rows.append({"model": model_name, "dataset": dname, "metric": "cindex",
                     "tau_start": τ1, "tau_end": τ1, "value": c_τ1})
        rows.append({"model": model_name, "dataset": dname, "metric": "cindex",
                     "tau_start": τ2, "tau_end": τ2, "value": c_τ2})
        rows.append({"model": model_name, "dataset": dname, "metric": "cindex",
                     "tau_start": τ3, "tau_end": τ3, "value": c_τ3})

        # ----- IPCW C-index: C(t | τ_k) -----
        c_τ2_given_τ1 = dynamic_cindex_ipcw(
            survival_train=y_train,
            survival_test=y_test,
            pids_test=pids_test,
            pred_df=pred_df_ds,
            tau_landmark=τ1,
            eval_time=τ2,
        )
        
        c_τ3_given_τ1 = dynamic_cindex_ipcw(
            survival_train=y_train,
            survival_test=y_test,
            pids_test=pids_test,
            pred_df=pred_df_ds,
            tau_landmark=τ1,
            eval_time=τ3,
        )
        
        c_τ3_given_τ2 = dynamic_cindex_ipcw(
                survival_train=y_train,
                survival_test=y_test,
                pids_test=pids_test,
                pred_df=pred_df_ds,
                tau_landmark=τ2,
                eval_time=τ3,
            )

        # Store results with tau_start = τ, tau_end = t
        rows.append({"model": model_name, "dataset": dname, "metric": "cindex_ipcw",
                     "tau_start": τ1, "tau_end": τ2, "value": c_τ2_given_τ1})
        rows.append({"model": model_name, "dataset": dname, "metric": "cindex_ipcw",
                     "tau_start": τ1, "tau_end": τ3, "value": c_τ3_given_τ1})
        rows.append({"model": model_name, "dataset": dname, "metric": "cindex_ipcw",
                     "tau_start": τ2, "tau_end": τ3, "value": c_τ3_given_τ2})

# -----------------------------------------------------------
# 7. Save final results
# -----------------------------------------------------------

results_df = pd.DataFrame(rows)
results_df = results_df.sort_values(
    ["model", "dataset", "metric", "tau_start", "tau_end"]
)
results_df.to_csv("metrics.csv", index=False)
print("\nSaved scores to metrics.csv")