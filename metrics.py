import os
import numpy as np
import pandas as pd

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc, brier_score

# -----------------------------------------------------------
# 1. Helper: read survival dataset and collapse to (T_i, δ_i)
# -----------------------------------------------------------

def load_survival_dataset(path):
    """
    Load survival dataset and get final outcome for each patient.
    Returns DataFrame with columns: pid, time, event
    """
    df = pd.read_csv(path)
    pat_df = (
        df.groupby("pid", as_index=False)
          .agg(time=("time2", "max"),
               event=("event", "max"))
    )
    return pat_df


# -----------------------------------------------------------
# 2. Harrell's C-index at each landmark time
# -----------------------------------------------------------

def compute_harrell_cindex_at_landmarks(
    survival_test,
    pred_df,
    landmark_times
):
    """
    Compute Harrell's C-index at each landmark time τ.
    
    For 3 landmarks [τ1, τ2, τ3], compute:
    - C-index for model trained using info up to τ1
    - C-index for model trained using info up to τ2
    - C-index for model trained using info up to τ3
    
    Formula: C(τ_k) = Σ δ_i * 1(τ_k < T_i < T_j) * 1(r_i > r_j) / Σ δ_i * 1(τ_k < T_i < T_j)
    
    Only includes patients with T > τ_k (alive at landmark).
    
    Parameters:
    -----------
    survival_test : DataFrame
        Test patient outcomes with columns: pid, time, event
    pred_df : DataFrame
        Predictions with columns: pid, tau, t, risk, surv_prob, dataset
    landmark_times : list
        List of landmark times [τ1, τ2, τ3]
    
    Returns:
    --------
    tuple: (list of C-index values, list of n_patients) for each landmark
    """
    
    cindices = []
    n_patients_list = []
    
    for tau in landmark_times:
        # 1) Subset to patients with T > τ_k (alive at landmark)
        at_risk = survival_test[survival_test["time"] > tau].copy()
        
        if len(at_risk) < 2:
            cindices.append(np.nan)
            n_patients_list.append(0)
            continue
        
        # 2) Get risk scores at this landmark (where tau matches and t is NaN)
        risk_data = pred_df[
            (pred_df["tau"] == tau) &
            (pred_df["t"].isna())
        ][["pid", "risk"]].copy()
        
        if risk_data.empty:
            cindices.append(np.nan)
            n_patients_list.append(0)
            continue
        
        # 3) Merge risk scores with at-risk patient outcomes
        merged = at_risk.merge(risk_data, on="pid", how="inner")
        
        if len(merged) < 2:
            cindices.append(np.nan)
            n_patients_list.append(len(merged))
            continue
        
        # Check if there's variation in events
        if merged["event"].sum() == 0 or merged["event"].sum() == len(merged):
            cindices.append(np.nan)
            n_patients_list.append(len(merged))
            continue
        
        # 4) Compute Harrell's C-index using patients with T > τ_k
        try:
            cindex, concordant, discordant, tied_risk, tied_time = (
                concordance_index_censored(
                    event_indicator=merged["event"].astype(bool).values,
                    event_time=merged["time"].values,
                    estimate=merged["risk"].values,
                )
            )
            cindices.append(cindex)
            n_patients_list.append(len(merged))
        except Exception as e:
            print(f"  Error computing C-index at τ={tau}: {e}")
            cindices.append(np.nan)
            n_patients_list.append(len(merged))
    
    return cindices, n_patients_list


# -----------------------------------------------------------
# 3. IPCW C-index at each landmark time
# -----------------------------------------------------------

def compute_ipcw_cindex_at_landmarks(
    survival_train,
    survival_test,
    pred_df,
    landmark_times
):
    """
    Compute IPCW C-index at each landmark time τ.
    
    For 3 landmarks [τ1, τ2, τ3], compute:
    - C-index for model trained using info up to τ1
    - C-index for model trained using info up to τ2
    - C-index for model trained using info up to τ3
    
    Formula: C_IPCW(τ_k) = Σ [δ_i * 1(τ_k < T_i < T_j) / G(T_i|τ_k)²] * 1(r_i > r_j) 
                           / Σ [δ_i * 1(τ_k < T_i < T_j) / G(T_i|τ_k)²]
    
    Only includes patients with T > τ_k (alive at landmark) in both train and test.
    
    Parameters:
    -----------
    survival_train : DataFrame
        Train patient outcomes with columns: pid, time, event (used for censoring weights)
    survival_test : DataFrame
        Test patient outcomes with columns: pid, time, event
    pred_df : DataFrame
        Predictions with columns: pid, tau, t, risk, surv_prob, dataset
    landmark_times : list
        List of landmark times [τ1, τ2, τ3]
    
    Returns:
    --------
    tuple: (list of IPCW C-index values, list of n_patients) for each landmark
    """
    
    cindices = []
    n_patients_list = []
    
    for tau in landmark_times:
        # 1) Subset TRAIN to patients with T > τ_k (alive at landmark)
        train_at_risk = survival_train[survival_train["time"] > tau].copy()
        
        if len(train_at_risk) < 2:
            cindices.append(np.nan)
            n_patients_list.append(0)
            continue
        
        # Create structured array for train set (for censoring weight estimation)
        y_train = Surv.from_dataframe("event", "time", train_at_risk)
        
        # 2) Subset TEST to patients with T > τ_k (alive at landmark)
        test_at_risk = survival_test[survival_test["time"] > tau].copy()
        
        if len(test_at_risk) < 2:
            cindices.append(np.nan)
            n_patients_list.append(0)
            continue
        
        # 3) Get risk scores at this landmark (where tau matches and t is NaN)
        risk_data = pred_df[
            (pred_df["tau"] == tau) &
            (pred_df["t"].isna())
        ][["pid", "risk"]].copy()
        
        if risk_data.empty:
            cindices.append(np.nan)
            n_patients_list.append(0)
            continue
        
        # 4) Merge risk scores with at-risk test patient outcomes
        merged = test_at_risk.merge(risk_data, on="pid", how="inner")
        
        if len(merged) < 2:
            cindices.append(np.nan)
            n_patients_list.append(len(merged))
            continue
        
        # Check if there's variation in events
        if merged["event"].sum() == 0 or merged["event"].sum() == len(merged):
            cindices.append(np.nan)
            n_patients_list.append(len(merged))
            continue
        
        # 5) Create structured array for test set
        y_test = Surv.from_dataframe("event", "time", merged)
        
        # 6) Compute IPCW C-index
        try:
            # Use tau as the evaluation time
            cindex, concordant, discordant, tied_risk, tied_time = (
                concordance_index_ipcw(
                    survival_train=y_train,
                    survival_test=y_test,
                    estimate=merged["risk"].values,
                    tau=None
                )
            )
            cindices.append(cindex)
            n_patients_list.append(len(merged))
        except Exception as e:
            print(f"  Error computing IPCW C-index at τ={tau}: {e}")
            cindices.append(np.nan)
            n_patients_list.append(len(merged))
    
    return cindices, n_patients_list


# -----------------------------------------------------------
# 4. IPCW AUC at each landmark time and horizon
# -----------------------------------------------------------

def compute_ipcw_auc_at_landmarks(
    survival_train,
    survival_test,
    pred_df,
    landmark_times
):
    """
    Compute IPCW AUC using risk scores from earlier landmarks to predict later times.
    
    For 3 landmarks [τ1, τ2, τ3], compute:
    - AUC(τ2), model trained using info up to τ1 
    - AUC(τ3), model trained using info up to τ1 
    - AUC(τ3), model trained using info up to τ2 
    
    Parameters:
    -----------
    survival_train : DataFrame
        Train patient outcomes with columns: pid, time, event
    survival_test : DataFrame
        Test patient outcomes with columns: pid, time, event
    pred_df : DataFrame
        Predictions with columns: pid, tau, t, risk, surv_prob, dataset
    landmark_times : list
        List of landmark times [τ1, τ2, τ3]
    
    Returns:
    --------
    list of dicts with keys: tau, t, auc, n_patients
    """
    
    results = []
    
    # For each landmark tau_i, use risk scores to predict AUC at later landmarks tau_j (j > i)
    for i, tau_risk in enumerate(landmark_times[:-1]):  # Exclude last landmark
        # Get risk scores at tau_risk
        risk_data = pred_df[
            (pred_df["tau"] == tau_risk) &
            (pred_df["t"].isna())
        ][["pid", "risk"]].copy()
        
        if risk_data.empty:
            continue
        
        # For each later landmark time
        for j in range(i+1, len(landmark_times)):
            tau_eval = landmark_times[j]
            
            # 1) Subset TRAIN to patients with T > tau_risk (alive at risk landmark)
            train_at_risk = survival_train[survival_train["time"] > tau_risk].copy()
            
            if len(train_at_risk) < 2:
                continue
            
            # Create structured array for train set
            y_train = Surv.from_dataframe("event", "time", train_at_risk)
            
            # 2) Subset TEST to patients with T > tau_risk (alive at risk landmark)
            test_at_risk = survival_test[survival_test["time"] > tau_risk].copy()
            
            if len(test_at_risk) < 2:
                continue
            
            # 3) Merge risk scores with test patient outcomes
            merged = test_at_risk.merge(risk_data, on="pid", how="inner")
            
            if len(merged) < 2:
                continue
            
            # Create structured array for test set
            y_test = Surv.from_dataframe("event", "time", merged)
            
            # 4) Compute AUC at tau_eval using risk scores from tau_risk
            try:
                auc_scores, mean_auc = cumulative_dynamic_auc(
                    survival_train=y_train,
                    survival_test=y_test,
                    estimate=merged["risk"].values,
                    times=[tau_eval]
                )
                
                results.append({
                    "tau": tau_risk,
                    "t": tau_eval,
                    "auc": auc_scores[0],
                    "n_patients": len(merged)
                })
            except Exception as e:
                print(f"  Error computing AUC at τ={tau_risk}, t={tau_eval}: {e}")
                results.append({
                    "tau": tau_risk,
                    "t": tau_eval,
                    "auc": np.nan,
                    "n_patients": len(merged)
                })
    
    return results


# -----------------------------------------------------------
# 5. Brier Score at each landmark time and horizon
# -----------------------------------------------------------

def compute_brier_score_at_landmarks(
    survival_train,
    survival_test,
    pred_df,
    landmark_times
):
    """
    Compute Brier Score using survival probabilities from earlier landmarks to predict later times.
    
    For 3 landmarks [τ1, τ2, τ3], compute:
    - BS(τ2) using survival prob at τ2, model trained using info up to τ1
    - BS(τ3) using survival prob at τ3, model trained using info up to τ1
    - BS(τ3) using survival prob at τ3, model trained using info up to τ2
    
    Parameters:
    -----------
    survival_train : DataFrame
        Train patient outcomes with columns: pid, time, event
    survival_test : DataFrame
        Test patient outcomes with columns: pid, time, event
    pred_df : DataFrame
        Predictions with columns: pid, tau, t, risk, surv_prob, dataset
    landmark_times : list
        List of landmark times [τ1, τ2, τ3]
    
    Returns:
    --------
    list of dicts with keys: tau, t, bs, n_patients
    """
    
    results = []
    
    # For each landmark tau_i, use survival probabilities to predict BS at later landmarks tau_j (j > i)
    for i, tau_risk in enumerate(landmark_times[:-1]):  # Exclude last landmark
        
        # For each later landmark time
        for j in range(i+1, len(landmark_times)):
            tau_eval = landmark_times[j]
            
            # 1) Get survival probabilities: trained at tau_risk, evaluated at tau_eval
            # These are in rows where tau=tau_risk and t=tau_eval
            surv_data = pred_df[
                (pred_df["tau"] == tau_risk) &
                (pred_df["t"] == tau_eval)
            ][["pid", "surv_prob"]].copy()
            
            if surv_data.empty:
                print(f"  WARNING: No survival probabilities found for τ={tau_risk}, t={tau_eval}")
                continue
            
            # 2) Subset TRAIN to patients with T > tau_risk (alive at risk landmark)
            train_at_risk = survival_train[survival_train["time"] > tau_risk].copy()
            
            if len(train_at_risk) < 2:
                print(f"  WARNING: Not enough train patients at τ={tau_risk}")
                continue
            
            # Create structured array for train set
            y_train = Surv.from_dataframe("event", "time", train_at_risk)
            
            # 3) Subset TEST to patients with T > tau_risk (alive at risk landmark)
            test_at_risk = survival_test[survival_test["time"] > tau_risk].copy()
            
            if len(test_at_risk) < 2:
                print(f"  WARNING: Not enough test patients at τ={tau_risk}")
                continue
            
            # 4) Merge survival probabilities with test patient outcomes
            merged = test_at_risk.merge(surv_data, on="pid", how="inner")
            
            if len(merged) < 2:
                print(f"  WARNING: Not enough merged patients at τ={tau_risk}, t={tau_eval}")
                continue
            
            # Create structured array for test set
            y_test = Surv.from_dataframe("event", "time", merged)
            
            # 5) Compute Brier Score at tau_eval
            try:
                # Create survival probability array - shape (n_samples, n_times)
                surv_probs = merged["surv_prob"].values.reshape(-1, 1)
                
                # brier_score returns (times, scores) tuple
                # scores is an array of length n_times
                times, bs_scores = brier_score(
                    survival_train=y_train,
                    survival_test=y_test,
                    estimate=surv_probs,
                    times=[tau_eval]
                )
                
                # Extract the score - it's just bs_scores[0] since we have one time point
                bs_value = bs_scores[0]
                
                results.append({
                    "tau": tau_risk,
                    "t": tau_eval,
                    "bs": bs_value,
                    "n_patients": len(merged)
                })
                print(f"  BS at τ={tau_risk}, t={tau_eval}: {bs_value:.4f} (n={len(merged)})")
                
            except Exception as e:
                print(f"  Error computing BS at τ={tau_risk}, t={tau_eval}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "tau": tau_risk,
                    "t": tau_eval,
                    "bs": np.nan,
                    "n_patients": len(merged)
                })
    
    return results


# -----------------------------------------------------------
# 6. Main evaluation loop
# -----------------------------------------------------------

data_dir = "data"
rows = []

models = [
    ("landmark_cox", "landmark_cox_risk.csv"),
    ("deephit", "dynamic_deephit_risk.csv"),
]

for model_name, pred_path in models:
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name} ({pred_path})")
    print(f"{'='*60}")

    if not os.path.exists(pred_path):
        print(f"WARNING: missing predictions file {pred_path}, skipping model.")
        continue

    # Load predictions
    pred_df = pd.read_csv(pred_path)
    datasets = pred_df["dataset"].unique()

    for dname in datasets:
        print(f"\n### Dataset: {dname}")

        csv_path = os.path.join(data_dir, f"{dname}.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: missing file {csv_path}, skipping dataset.")
            continue

        # Load patient outcomes
        pat_df = load_survival_dataset(csv_path)

        # Get predictions for this dataset
        pred_df_ds = pred_df[pred_df["dataset"] == dname].copy()
        test_pids = np.sort(pred_df_ds["pid"].unique())

        # Split into train and test patients
        is_test = pat_df["pid"].isin(test_pids)
        pat_train = pat_df[~is_test].copy()
        pat_test = pat_df[is_test].copy()

        if pat_test.empty:
            print(f"WARNING: no test patients for {dname}. Skipping.")
            continue
        
        if pat_train.empty:
            print(f"WARNING: no train patients for {dname}. Skipping.")
            continue

        # Get landmark times from predictions
        landmark_times = sorted(pred_df_ds[pred_df_ds["t"].isna()]["tau"].unique())
        
        if len(landmark_times) < 3:
            print(f"WARNING: Not enough landmarks for dataset {dname}. Need ≥ 3, got {len(landmark_times)}.")
            continue

        # Take first 3 landmark times
        landmark_times = landmark_times[:3]
        print(f"Landmark times: τ1={landmark_times[0]:.2f}, τ2={landmark_times[1]:.2f}, τ3={landmark_times[2]:.2f}")

        # Compute Harrell's C-index at each landmark
        print("\n  Computing Harrell's C-index...")
        cindices_harrell, n_patients_harrell = compute_harrell_cindex_at_landmarks(
            survival_test=pat_test,
            pred_df=pred_df_ds,
            landmark_times=landmark_times
        )

        # Store Harrell C-index results
        for tau, cindex, n_patients in zip(landmark_times, cindices_harrell, n_patients_harrell):
            rows.append({
                "model": model_name,
                "dataset": dname,
                "metric": "cindex_harrell",
                "tau": tau,
                "t": np.nan,
                "value": cindex,
                "n_patients": n_patients
            })
          
        # Compute IPCW C-index at each landmark
        print("\n  Computing IPCW C-index...")
        cindices_ipcw, n_patients_ipcw = compute_ipcw_cindex_at_landmarks(
            survival_train=pat_train,
            survival_test=pat_test,
            pred_df=pred_df_ds,
            landmark_times=landmark_times
        )

        # Store IPCW C-index results
        for tau, cindex, n_patients in zip(landmark_times, cindices_ipcw, n_patients_ipcw):
            rows.append({
                "model": model_name,
                "dataset": dname,
                "metric": "cindex_ipcw",
                "tau": tau,
                "t": np.nan,
                "value": cindex,
                "n_patients": n_patients
            })
      
        # Compute IPCW AUC at each landmark and horizon
        print("\n  Computing IPCW AUC...")
        auc_results = compute_ipcw_auc_at_landmarks(
            survival_train=pat_train,
            survival_test=pat_test,
            pred_df=pred_df_ds,
            landmark_times=landmark_times
        )
        
        # Store AUC results
        for auc_res in auc_results:
            rows.append({
                "model": model_name,
                "dataset": dname,
                "metric": "auc_ipcw",
                "tau": auc_res["tau"],
                "t": auc_res["t"],
                "value": auc_res["auc"],
                "n_patients": auc_res["n_patients"]
            })
            
        # Compute Brier Score at each landmark and horizon
        print("\n  Computing Brier Score...")
        bs_results = compute_brier_score_at_landmarks(
            survival_train=pat_train,
            survival_test=pat_test,
            pred_df=pred_df_ds,
            landmark_times=landmark_times
        )
        
        # Store Brier Score results
        for bs_res in bs_results:
            rows.append({
                "model": model_name,
                "dataset": dname,
                "metric": "brier_score",
                "tau": bs_res["tau"],
                "t": bs_res["t"],
                "value": bs_res["bs"],
                "n_patients": bs_res["n_patients"]
            })

# -----------------------------------------------------------
# 7. Save results
# -----------------------------------------------------------

results_df = pd.DataFrame(rows)
results_df = results_df.sort_values(["model", "dataset", "metric", "tau", "t"])
results_df.to_csv("metrics.csv", index=False)
print("\n" + "="*60)
print("Results saved to metrics.csv")
print("="*60)

# Summary statistics
print("\n" + "="*60)
print("Summary: Mean metrics by Model and Dataset")
print("="*60)
summary = results_df.groupby(["model", "dataset", "metric"])["value"].agg(["mean", "std", "count"])
print(summary)

print("\n" + "="*60)
print("Summary: Mean metrics by Model and Metric (across all datasets)")
print("="*60)
model_summary = results_df.groupby(["model", "metric"])["value"].agg(["mean", "std", "count"])
print(model_summary)