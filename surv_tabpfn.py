import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from sklearn.pipeline import make_pipeline
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, integrated_brier_score
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from pycox.models import DeepHitSingle, CoxPH
import torchtuples as tt
from tabpfn import TabPFNClassifier
import warnings
import matplotlib.pyplot as plt
import random
from sksurv.nonparametric import SurvivalFunctionEstimator 
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Directory containing the datasets
data_dir = os.path.join("test1")

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# No longer need bin_sizes since we use fixed evaluation times

# CSV path
csv_path = "tabpfn_evaluation.csv"

def construct_tabpfn_trainset(x_train_imputed, y_train, cuts):
    """
    Construct TabPFN training dataset by creating examples for each patient at cut timepoints:
    - Examples at timepoints where patient was observed (class C or D)
    - Examples at timepoints before/after patient time (class A or B)

    Parameters:
        x_train_imputed (pd.DataFrame): Imputed covariates (n_patients × features)
        y_train (structured array): Survival data with fields "event" and "time" (discretized indices)
        cuts (np.ndarray): The actual cut values used for discretization

    Returns:
        X_tabpfn_train (pd.DataFrame): Feature matrix with timepoints
        y_tabpfn_train (pd.Series): Corresponding class labels ("A", "B", "C", "D")
    """
    # Extract event/censoring times and status
    T_train = y_train["time"]  # These are discretized indices
    delta_train = y_train["event"]
    n_train = len(T_train)

    # Use the actual cut values as evaluation timepoints
    selected_times = cuts.copy()

    dataset_rows = []
    class_labels = []

    # Create examples for each patient at each cut timepoint
    for i in range(n_train):
        x_i = x_train_imputed.iloc[i].values
        T_i_idx = T_train[i]  # Discretized time index
        delta_i = delta_train[i]

        # Create examples at each cut timepoint
        for j, t_j in enumerate(selected_times):
            # Assign class label based on T_i_idx and current timepoint index j
            if T_i_idx < j:
                label = "A"  # Event/censoring happened before this timepoint
            elif T_i_idx > j:
                label = "B"  # Event/censoring will happen after this timepoint
            elif T_i_idx == j and delta_i == 0:
                label = "C"  # Censored at this timepoint
            elif T_i_idx == j and delta_i == 1:
                label = "D"  # Event at this timepoint
            else:
                # Fallback (shouldn't happen with proper discretization)
                if T_i_idx < j:
                    label = "A"
                else:
                    label = "B"

            # Feature = original + eval_time (use actual cut value)
            row = np.concatenate([x_i, [t_j]])
            dataset_rows.append(row)
            class_labels.append(label)

    # Convert to arrays for easier manipulation
    dataset_rows = np.array(dataset_rows)
    class_labels = np.array(class_labels)
    
    # Create DataFrame and Series for return
    feature_cols = list(x_train_imputed.columns) + ["eval_time"]
    X_tabpfn_train = pd.DataFrame(dataset_rows, columns=feature_cols)
    y_tabpfn_train = pd.Series(class_labels)
    
    return X_tabpfn_train, y_tabpfn_train

def construct_tabpfn_testset(x_test_imputed, times):
    n_test = x_test_imputed.shape[0]
    d = x_test_imputed.shape[1]

    test_rows = []
    patient_ids = []

    for i in range(n_test):
        x_i = x_test_imputed.iloc[i].values
        for t in times:
            row = np.concatenate([x_i, [t]])
            test_rows.append(row)
            patient_ids.append(i)

    feature_cols = list(x_test_imputed.columns) + ["eval_time"]
    X_tabpfn_test = pd.DataFrame(test_rows, columns=feature_cols)
    return X_tabpfn_test, np.array(patient_ids)

def _is_monotonic_increasing(x: np.ndarray) -> bool:
    return (x[1:] >= x[:-1]).all()

def bin_numerical(x: np.ndarray, right_cuts: np.ndarray, error_on_larger: bool = False) -> np.ndarray:
    """
    pycox.utils.bin_numerical: index of bins defined by right_cuts (sorted).
    If right_cuts = [c0, c1, ... c_{m-1}], bins are (-inf, c0], (c0, c1], ... (c_{m-2}, c_{m-1}], (c_{m-1}, inf)
    """
    assert _is_monotonic_increasing(right_cuts), "Need `right_cuts` to be sorted."
    bins = np.searchsorted(right_cuts, x, side="left")
    if error_on_larger and bins.max() == right_cuts.size:
        raise ValueError("x contains larger values than right_cuts.")
    return bins

def discretize(x: np.ndarray, cuts: np.ndarray, side: str = "right", error_on_larger: bool = False) -> np.ndarray:
    """
    pycox.preprocessing.discretize: map times to cut values (not indices).
    side='right'  -> round up to right cut (events)
    side='left'   -> round down to left cut (censorings)
    """
    if side not in ["right", "left"]:
        raise ValueError("side must be 'right' or 'left'.")
    bins = bin_numerical(x, cuts, error_on_larger=error_on_larger)
    if side == "right":
        # append +inf so the last open interval maps to inf (pycox does the same then caps elsewhere)
        cuts_ext = np.concatenate([cuts, np.array([np.inf])])
        return cuts_ext[bins]
    # side == 'left'
    bins_cut = bins.copy()
    bins_cut[bins_cut == cuts.size] = -1
    exact = cuts[bins_cut] == x
    left_bins = bins - 1 + exact
    vals = cuts[left_bins]
    vals[left_bins == -1] = -np.inf
    return vals

def km_quantile_cuts(durations: np.ndarray, events: np.ndarray, num: int, min_=0., dtype="float64") -> np.ndarray:
    """
    pycox.preprocessing.cuts_quantiles replicated with sksurv:
    Build `num` right-cuts (sorted), with equal drop in KM survival between min and max observed time.
    Ensures cuts[0] = min_ (or durations.min() if min_=None) and cuts[-1] = durations.max().
    """
    sfe = SurvivalFunctionEstimator().fit(
        Surv.from_arrays(event=events.astype(bool), time=durations.astype(float))
    )
    t_sorted = np.sort(np.unique(durations.astype(float)))
    if t_sorted.size < 2:
        return np.array([t_sorted.min(), t_sorted.max()], dtype=dtype)

    S_hat = sfe.predict_proba(t_sorted)
    # pycox line: s_cuts = linspace(km.values.min(), km.values.max(), num)
    s_cuts = np.linspace(S_hat.min(), S_hat.max(), num)

    # pycox logic on reversed arrays
    cuts_idx = np.searchsorted(S_hat[::-1], s_cuts)[::-1]
    cuts = t_sorted[::-1][cuts_idx]
    cuts = np.unique(cuts)
    if cuts.size != num:
        warnings.warn(f"cuts are not unique, continue with {cuts.size} cuts instead of {num}")
    # first cut: durations.min() if min_ is None else min_
    cuts = cuts.astype(dtype)
    cuts[0] = durations.min() if min_ is None else min_
    # pycox asserts last cut equals durations.max()
    cuts[-1] = durations.max()
    return cuts

def discretize_unknown_c(duration: np.ndarray,
                         event: np.ndarray,
                         cuts: np.ndarray,
                         right_censor: bool = True,
                         censor_side: str = "left"):
    """
    pycox.preprocessing.DiscretizeUnknownC.transform:
    - Optionally right-censor values > cuts.max() and set event=False there
    - Events -> side='right'
    - Censorings -> side=censor_side (default 'left')
    Returns (td, event_adj) where `td` are cut-values (right/left edge), not indices.
    """
    dtype_event = event.dtype
    event = event.astype(bool).copy()
    duration = duration.astype(float).copy()

    if right_censor:
        censor_mask = duration > cuts.max()
        if censor_mask.any():
            duration[censor_mask] = cuts.max()
            event[censor_mask] = False

    if duration.max() > cuts.max():
        raise ValueError("`duration` contains larger values than cuts. Set right_censor=True to cap these.")

    td = np.zeros_like(duration, dtype=float)
    cens_mask = ~event
    td[event] = discretize(duration[event], cuts, side="right", error_on_larger=True)
    if cens_mask.any():
        td[cens_mask] = discretize(duration[cens_mask], cuts, side=censor_side, error_on_larger=True)
    return td, event.astype(dtype_event)

def duration_to_index_map(cuts: np.ndarray):
    """Map each cut value to its index (0..len(cuts)-1)."""
    uniq = np.unique(cuts)
    order = np.argsort(uniq)
    idx_map = {float(uniq[o]): int(o) for o in order}
    
    def safe_map(x):
        if x is None or np.isnan(x):
            return 0  # Default to first index
        result = idx_map.get(float(x))
        if result is None:
            # Find closest cut point
            closest_idx = np.searchsorted(uniq, float(x))
            return min(closest_idx, len(uniq) - 1)
        return result
    
    return np.vectorize(safe_map, otypes=[int])


def cdi_interpolate(eval_times, grid_times, S_grid):
    """
    Constant Density Interpolation (CDI) for survival functions.
    
    Assumes uniform distribution of event times within each interval,
    leading to S(t) = α_j - β_j * t within interval (τ_{j-1}, τ_j].
    
    Parameters:
        eval_times: array of times to evaluate S(t) at
        grid_times: array of grid points τ_j (sorted)
        S_grid: survival probabilities at grid points (n_patients × n_grid)
    
    Returns:
        S_eval: survival probabilities at eval_times (n_patients × n_eval)
    """
    n_patients, n_grid = S_grid.shape
    n_eval = len(eval_times)
    S_eval = np.zeros((n_patients, n_eval))
    
    for i in range(n_patients):
        S_i = S_grid[i]
        
        for j, t in enumerate(eval_times):
            # Find interval: t ∈ (τ_{k-1}, τ_k]
            k = np.searchsorted(grid_times, t, side='right')
            
            if k == 0:
                # t <= τ_0, extrapolate as constant
                S_eval[i, j] = 1.0  # or S_i[0]
            elif k >= n_grid:
                # t > τ_{max}, extrapolate as constant
                S_eval[i, j] = S_i[-1]
            else:
                # t ∈ (τ_{k-1}, τ_k], apply CDI formula
                tau_prev = grid_times[k-1] if k > 0 else 0.0
                tau_curr = grid_times[k]
                S_prev = S_i[k-1] if k > 0 else 1.0
                S_curr = S_i[k]
                
                # CDI coefficients: S(t) = α_j - β_j * t
                delta_tau = tau_curr - tau_prev
                if delta_tau > 0:
                    alpha_j = (S_prev * tau_curr - S_curr * tau_prev) / delta_tau
                    beta_j = (S_prev - S_curr) / delta_tau
                    S_eval[i, j] = alpha_j - beta_j * t
                else:
                    # Degenerate interval, use endpoint value
                    S_eval[i, j] = S_curr
    
    # Ensure monotonicity and bounds
    S_eval = np.clip(S_eval, 0.0, 1.0)
    return S_eval


for file_name in dataset_files:
    try:
        dataset_name = file_name.replace(".csv", "")
        file_path = os.path.join(data_dir, file_name)
        print("="*50)
        print(f"\nProcessing dataset: {dataset_name}")

        # Track best per dataset
        best_row = None

        # Load datasets
        df = pd.read_csv(file_path)
        df.drop(columns=['pid'], inplace=True)
        shape = df.shape
        print(f"Dataset shape: {shape}")

        # Define columns
        time_col = "time"
        event_col = "event"
        censored = (df["event"] == 0).sum()
        censored_percent = (censored)/len(df)*100
        print(f"Percentage of censored data: {censored_percent}%")
        covariates = df.columns.difference([time_col, event_col])

        # Define covariates and target variable
        x = df[covariates].copy()
        y = Surv.from_arrays(event=df[event_col].astype(bool), time=df[time_col])
        
        # --------------- 70-15-15 Split (same as baseline.py) ---------------
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x, y, test_size=0.15, stratify=y["event"], random_state=SEED
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=0.1765, stratify=y_trainval["event"], random_state=SEED
        )
        # 0.1765 * 0.85 ≈ 0.15, so test/val are both 15%

        # One-hot encode using get_dummies (fit on TRAIN only!)
        x_train_ohe = pd.get_dummies(x_train, drop_first=True)
        x_val_ohe   = pd.get_dummies(x_val, drop_first=True)
        x_test_ohe  = pd.get_dummies(x_test, drop_first=True)

        # Align columns so train/val/test match
        x_train_ohe, x_val_ohe = x_train_ohe.align(x_val_ohe, join="left", axis=1, fill_value=0)
        x_train_ohe, x_test_ohe = x_train_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)

        covariates_ohe = x_train_ohe.columns  # all columns are covariates after OHE

        # Impute missing values in covariates (fit on train only, same as baseline.py)
        imputer = SimpleImputer().fit(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = imputer.transform(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = pd.DataFrame(x_train_imputed, columns=covariates_ohe, index=x_train.index)
        x_val_imputed = imputer.transform(x_val_ohe.loc[:, covariates_ohe.tolist()])
        x_val_imputed = pd.DataFrame(x_val_imputed, columns=covariates_ohe, index=x_val.index)
        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # Evaluation time points (use train+val, same as baseline.py)
        times = np.percentile(y_trainval["time"], np.arange(10, 100, 10))
        times = np.unique(times)
        max_trainval_time = y_trainval["time"].max()
        times = times[times < max_trainval_time]
        test_mask = y_test["time"] < max_trainval_time
        y_test_filtered = y_test[test_mask]
        x_test_filtered = x_test_imputed[test_mask]
        times = times[(times > y_test_filtered["time"].min()) & (times < y_test_filtered["time"].max())]
        print(f"Evaluation time points: {times}")

        # Keep the masked test index for alignment inside the n_bins loop
        masked_test_index = x_test_filtered.index

        # Keep originals for metrics (explicit names for clarity)
        y_train_orig = y_train
        y_test_orig = y_test_filtered

        # ========================= Train on fixed evaluation times =========================
        print("Training TabPFN with fixed evaluation times...")
        
        # Use fixed number of cuts based on evaluation times
        n_bins = len(times)
        print(f"Using {n_bins} cuts based on evaluation times.")

        # ========== First train on train set and validate ==========
        # Use evaluation times directly as cuts for discretization
        cuts = times.copy()
        print(f"Using evaluation times as cuts: {cuts}")

        # Discretize TRAIN
        t_tr = df.loc[x_train.index, time_col].to_numpy()
        e_tr = df.loc[x_train.index, event_col].astype(bool).to_numpy()
        td_tr, e_tr_adj = discretize_unknown_c(t_tr, e_tr, cuts, right_censor=True, censor_side="left")

        # Discretize VAL
        t_val = df.loc[x_val.index, time_col].to_numpy()
        e_val = df.loc[x_val.index, event_col].astype(bool).to_numpy()
        td_val, e_val_adj = discretize_unknown_c(t_val, e_val, cuts, right_censor=True, censor_side="left")

        # Map discrete cut-values -> integer indices
        to_idx = duration_to_index_map(cuts)
        t_train_bin = to_idx(td_tr)
        t_val_bin = to_idx(td_val)

        # Build Surv objects in model time (indices)
        y_train_model = Surv.from_arrays(event=e_tr_adj, time=t_train_bin)
        y_val_model = Surv.from_arrays(event=e_val_adj, time=t_val_bin)

        # Train TabPFN on train data
        X_tabpfn_train, y_tabpfn_train = construct_tabpfn_trainset(
            x_train_imputed, y_train_model, cuts
        )

        tabpfn_model = TabPFNClassifier(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ignore_pretraining_limits=True
        )
        tabpfn_model.fit(X_tabpfn_train, y_tabpfn_train)

        # Predict on validation set
        X_tabpfn_val, val_patient_ids = construct_tabpfn_testset(x_val_imputed, cuts)
        probs = tabpfn_model.predict_proba(X_tabpfn_val)

        # Convert A/B/C/D -> discrete hazards
        def hazards_from_probs(X_test, pids, probs, cut_values, eps=1e-12):
            n_patients = pids.max() + 1
            m = len(cut_values)
            h = np.zeros((n_patients, m), dtype=float)
            for j, t in enumerate(cut_values):
                idx_at_t = np.where(np.isclose(X_test["eval_time"], t))[0]
                for idx in idx_at_t:
                    pid = pids[idx]
                    pA, pB, pC, pD = probs[idx]
                    denom = pB + pC + pD + eps   # at-risk at t
                    h[pid, j] = pD / denom
            return np.clip(h, 0.0, 1.0)

        h_val = hazards_from_probs(X_tabpfn_val, val_patient_ids, probs, cuts)

        # Build survival function
        S_val = np.clip(np.cumprod(1.0 - h_val, axis=1), 0.0, 1.0)
        S_val = np.minimum.accumulate(S_val, axis=1)

        # Risk for ranking: probability of event by horizon
        risk_scores_val = 1.0 - S_val[:, -1]

        # Validation C-index
        val_c_index, *_ = concordance_index_censored(y_val["event"], y_val["time"], risk_scores_val)
        print(f"Validation C-index: {val_c_index:.4f}")

        # ========== Retrain on train+val, evaluate on test (same as baseline.py) ==========
        # OHE and impute trainval/test
        x_trainval_ohe = pd.get_dummies(x_trainval, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)
        x_trainval_ohe, x_test_ohe = x_trainval_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)
        covariates_ohe = x_trainval_ohe.columns
        imputer = SimpleImputer().fit(x_trainval_ohe.loc[:, covariates_ohe.tolist()])
        x_trainval_imputed = imputer.transform(x_trainval_ohe.loc[:, covariates_ohe.tolist()])
        x_trainval_imputed = pd.DataFrame(x_trainval_imputed, columns=covariates_ohe, index=x_trainval.index)
        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)
        test_mask = y_test["time"] < max_trainval_time
        y_test_filtered = y_test[test_mask]
        x_test_filtered = x_test_imputed[test_mask]
        times = times[(times > y_test_filtered["time"].min()) & (times < y_test_filtered["time"].max())]
        
        # Use evaluation times directly as cuts for discretization
        cuts = times.copy()
        print(f"Using evaluation times as cuts: {cuts}")

        # 2) Discretize TRAINVAL
        t_trainval = df.loc[x_trainval.index, time_col].to_numpy()
        e_trainval = df.loc[x_trainval.index, event_col].astype(bool).to_numpy()
        td_trainval, e_trainval_adj = discretize_unknown_c(t_trainval, e_trainval, cuts, right_censor=True, censor_side="left")

        # 3) Discretize TEST
        t_te_full = df.loc[x_test.index, time_col].to_numpy()
        e_te_full = df.loc[x_test.index, event_col].astype(bool).to_numpy()
        td_te_full, e_te_adj_full = discretize_unknown_c(t_te_full, e_te_full, cuts, right_censor=True, censor_side="left")

        # 4) Map discrete cut-values -> integer indices
        to_idx = duration_to_index_map(cuts)
        t_trainval_bin = to_idx(td_trainval)
        t_test_bin_full = to_idx(td_te_full)

        # 5) Build Surv objects
        y_trainval_model = Surv.from_arrays(event=e_trainval_adj, time=t_trainval_bin)
        y_test_model = Surv.from_arrays(
            event=e_te_adj_full[x_test.index.get_indexer(masked_test_index)],
            time=t_test_bin_full[x_test.index.get_indexer(masked_test_index)]
        )

        # 6) Map evaluation times to model indices
        times_eval = bin_numerical(times, cuts, error_on_larger=False)
        times_eval = np.clip(times_eval, 0, len(cuts) - 1).astype(int)
        
        # Train TabPFN on trainval data
        X_tabpfn_train, y_tabpfn_train = construct_tabpfn_trainset(
            x_trainval_imputed, y_trainval_model, cuts
        )

        tabpfn_model = TabPFNClassifier(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ignore_pretraining_limits=True
        )
        tabpfn_model.fit(X_tabpfn_train, y_tabpfn_train)

        # Work on unique evaluation bins to avoid duplicate counting
        unique_bins, inv = np.unique(times_eval, return_inverse=True)

        # Predict on unique bins only
        X_tabpfn_test, test_patient_ids = construct_tabpfn_testset(x_test_filtered, cuts[unique_bins])
        probs = tabpfn_model.predict_proba(X_tabpfn_test)

        # Convert A/B/C/D -> discrete hazards per unique bin
        def hazards_from_probs(X_test, pids, probs, unique_cut_values, eps=1e-12):
            n_patients = pids.max() + 1
            m = len(unique_cut_values)
            h = np.zeros((n_patients, m), dtype=float)
            for j, t in enumerate(unique_cut_values):
                idx_at_t = np.where(np.isclose(X_test["eval_time"], t))[0]
                for idx in idx_at_t:
                    pid = pids[idx]
                    pA, pB, pC, pD = probs[idx]
                    denom = pB + pC + pD + eps   # at-risk at t
                    h[pid, j] = pD / denom
            return np.clip(h, 0.0, 1.0)

        h_unique = hazards_from_probs(X_tabpfn_test, test_patient_ids, probs, cuts[unique_bins])

        # Build survival on the model's native grid
        S_unique = np.clip(np.cumprod(1.0 - h_unique, axis=1), 0.0, 1.0)

        # Interpolate to ORIGINAL 'times' grid using CDI
        bin_t_grid = cuts[unique_bins]
        S_full = cdi_interpolate(times, bin_t_grid, S_unique)
        S_full = np.minimum.accumulate(S_full, axis=1)

        # Calculate time-dependent risk scores using cumulative hazard
        H = -np.log(S_full + 1e-8)  # Add small epsilon to avoid log(0)
        # Risk for ranking: probability of event by horizon
        risk_scores_ranking = 1.0 - S_full[:, -1]

        # Final metrics on test set
        c_index, *_ = concordance_index_censored(y_test_orig["event"], y_test_orig["time"], risk_scores_ranking)
        ibs = integrated_brier_score(y_trainval, y_test_orig, S_full, times)
        # For AUC, use time-dependent risk scores (2D array: n_samples x n_times)
        _, mean_auc = cumulative_dynamic_auc(y_trainval, y_test_orig, H, times=times)

        best_row = {
            "dataset": dataset_name,
            "n_eval_times": len(times),
            "score": round(float(val_c_index), 4),
            "c_index": round(float(c_index), 4),
            "ibs": round(float(ibs), 4),
            "mean_auc": round(float(mean_auc), 4),
        }
        
        print("="*50)
        print(f"Final test results for dataset {dataset_name}:")
        print(f"Number of eval time points: {best_row['n_eval_times']}")
        print(f"Validation C-index (Score): {best_row['score']:.4f}")
        print(f"Test C-index: {best_row['c_index']:.4f}")
        print(f"Test interval Brier Score (IBS): {best_row['ibs']:.4f}")
        print(f"Test mean AUC: {best_row['mean_auc']:.4f}")

        # ========================= write results =========================
        if best_row is not None:
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["dataset", "n_eval_times", "score", "c_index", "ibs", "mean_auc"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(best_row)

    except Exception as e:
        print(f"Error: {e}")
        continue