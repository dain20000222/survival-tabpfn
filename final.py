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
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from tabpfn import TabPFNClassifier
import warnings
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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
data_dir = os.path.join("test")

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# Sweep configs
neighbors_list = [1, 2, 3, 4]
bin_sizes = [None, 5, 10, 15, 20]

# CSV path
csv_path = "tabpfn_results.csv"

def construct_tabpfn_trainset(x_train_imputed, y_train, n_neighbors=1):
    """
    Construct TabPFN training dataset by creating classification examples for each patient
    at n timepoints before/after their event/censoring time, including the exact time if it exists.

    Parameters:
        x_train_imputed (pd.DataFrame): Imputed covariates (n_patients × features)
        y_train (structured array): Survival data with fields "event" and "time"
        n_neighbors (int): Number of timepoints to include before and after the event time

    Returns:
        X_tabpfn_train (pd.DataFrame): Feature matrix with timepoints
        y_tabpfn_train (pd.Series): Corresponding class labels ("A", "B", "C", "D")
    """
    # Extract event/censoring times and status
    T_train = y_train["time"]
    delta_train = y_train["event"]
    n_train = len(T_train)

    # Sorted unique event/censoring times
    unique_times = np.sort(np.unique(T_train))
    dataset_rows = []
    class_labels = []

    for i in range(n_train):
        x_i = x_train_imputed.iloc[i].values
        T_i = T_train[i]
        delta_i = delta_train[i]

        # Index of T_i in unique_times
        idx = np.searchsorted(unique_times, T_i)

        # Collect candidate timepoints: n before, at, n after
        start_idx = max(0, idx - n_neighbors)
        end_idx = min(len(unique_times), idx + n_neighbors + 1)

        timepoints = unique_times[start_idx:end_idx]

        for t_j in timepoints:
            # Assign class label based on T_i and t_j
            if T_i < t_j:
                label = "A"
            elif T_i > t_j:
                label = "B"
            elif np.isclose(T_i, t_j) and delta_i == 0:
                label = "C"
            elif np.isclose(T_i, t_j) and delta_i == 1:
                label = "D"
            else:
                raise ValueError(f"Unexpected condition: T_i={T_i}, t_j={t_j}, delta_i={delta_i}")

            # Feature = original + eval_time
            row = np.concatenate([x_i, [t_j]])
            dataset_rows.append(row)
            class_labels.append(label)

    # Create DataFrame
    timepoint_name = "eval_time"
    feature_cols = list(x_train_imputed.columns) + [timepoint_name]
    X_tabpfn_train = pd.DataFrame(dataset_rows, columns=feature_cols)
    y_tabpfn_train = pd.Series(class_labels, name="label")

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
    return np.vectorize(idx_map.get, otypes=[int])


for file_name in dataset_files:
    try:
        dataset_name = file_name.replace(".csv", "")
        file_path = os.path.join(data_dir, file_name)
        print("="*50)
        print(f"\nProcessing dataset: {dataset_name}")

        # Track best per dataset
        best_cindex = -1.0
        best_row = None

        # Load datasets
        df = pd.read_csv(file_path)
        df.drop(columns=['pid'], inplace=True)
        shape = df.shape
        print(f"Dataset shape: {shape}")

        # One-hot encode categorical variables
        df = pd.get_dummies(df, drop_first=True)
        censored = (df["event"] == 0).sum()
        censored_percent = (censored)/len(df)*100
        print(f"Percentage of censored data: {censored_percent}%")

        # Define columns
        time_col = "time"
        event_col = "event"
        covariates = df.columns.difference([time_col, event_col])

        # Define covariates and target variable
        x = df[covariates].copy()
        y = Surv.from_arrays(event=df[event_col].astype(bool), time=df[time_col])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y["event"], random_state=0)

        # Impute missing values in covariates
        imputer = SimpleImputer().fit(x_train.loc[:, covariates.tolist()])
        x_train_imputed = imputer.transform(x_train.loc[:, covariates.tolist()])
        x_train_imputed = pd.DataFrame(x_train_imputed, columns=covariates, index=x_train.index)
        x_test_imputed = imputer.transform(x_test.loc[:, covariates.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates, index=x_test.index)

        # Evaluation time points
        times = np.percentile(y["time"], np.arange(10, 100, 10))
        times = np.unique(times)

        # Clip 2
        # Get max time
        max_train_time = y_train["time"].max()

        # Clip evaluation time grid
        times = times[times < max_train_time]

        # Filter out test samples with time beyond train horizon
        test_mask = y_test["time"] < max_train_time
        y_test = y_test[test_mask]
        x_test_imputed = x_test_imputed[test_mask]

        # Clip 3
        times = times[(times > y_test["time"].min()) & (times < y_test["time"].max())]
        print(f"Evaluation time points: {times}")

        # Keep the masked test index for alignment inside the n_bins loop
        masked_test_index = x_test_imputed.index

        # Keep originals for metrics (explicit names for clarity)
        y_train_orig = y_train
        y_test_orig = y_test

        # ========================= n_bins / n_neighbors sweep =========================
        for n_bins in bin_sizes:

            if n_bins is not None:
                print(f"Using KM-based discretization with {n_bins} cuts.")
                # 1) Build KM-quantile right-cuts (length = n_bins, includes min and max)
                cuts = km_quantile_cuts(
                    durations=df.loc[x_train.index, time_col].to_numpy(),
                    events=df.loc[x_train.index, event_col].to_numpy(),
                    num=n_bins,
                    min_=df[time_col].min(),
                    dtype="float64",
                )
                if (np.diff(cuts) == 0).any():
                    print(f"⚠️ Skipping n_bins={n_bins}: non-unique KM cuts.")
                    continue

                # 2) Discretize TRAIN (pycox §4.1): events -> right edge, censor -> left edge; right-censor beyond last cut
                t_tr = df.loc[x_train.index, time_col].to_numpy()
                e_tr = df.loc[x_train.index, event_col].astype(bool).to_numpy()
                td_tr, e_tr_adj = discretize_unknown_c(t_tr, e_tr, cuts, right_censor=True, censor_side="left")

                # 3) Discretize TEST the same way (using TRAIN cuts!)
                t_te_full = df.loc[x_test.index, time_col].to_numpy()
                e_te_full = df.loc[x_test.index, event_col].astype(bool).to_numpy()
                td_te_full, e_te_adj_full = discretize_unknown_c(t_te_full, e_te_full, cuts, right_censor=True, censor_side="left")

                # 4) Map discrete cut-values -> integer indices 0..len(cuts)-1
                to_idx = duration_to_index_map(cuts)
                t_train_bin = to_idx(td_tr)
                t_test_bin_full = to_idx(td_te_full)

                # 5) Build Surv objects in *model time* (indices), aligned to your masked test
                y_train_model = Surv.from_arrays(event=e_tr_adj, time=t_train_bin)
                y_test_model = Surv.from_arrays(
                    event=e_te_adj_full[x_test.index.get_indexer(masked_test_index)],
                    time=t_test_bin_full[x_test.index.get_indexer(masked_test_index)]
                )

                # 6) Map continuous evaluation 'times' -> model indices via right rounding (like events)
                #    This is equivalent to discretize(..., side='right'), but we only need the index.
                times_eval = bin_numerical(times, cuts, error_on_larger=False)  # 0..len(cuts) ; clip to last valid index
                times_eval = np.clip(times_eval, 0, len(cuts) - 1).astype(int)
            else:
                print("No binning (using original time).")
                y_train_model = y_train_orig
                y_test_model  = y_test_orig
                times_eval = times
            
            for n_neighbors in neighbors_list:
                # Train set uses model-time (binned or original)
                X_tabpfn_train, y_tabpfn_train = construct_tabpfn_trainset(
                    x_train_imputed, y_train_model, n_neighbors=n_neighbors
                )

                tabpfn_model = TabPFNClassifier(
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    ignore_pretraining_limits=True
                )
                tabpfn_model.fit(X_tabpfn_train, y_tabpfn_train)

                # Work on unique evaluation bins to avoid duplicate counting
                unique_bins, inv = np.unique(times_eval, return_inverse=True)

                # Predict on unique bins only
                X_tabpfn_test, test_patient_ids = construct_tabpfn_testset(x_test_imputed, unique_bins)
                probs = tabpfn_model.predict_proba(X_tabpfn_test)

                # Convert A/B/C/D -> discrete hazards per unique bin
                def hazards_from_probs(X_test, pids, probs, unique_bins, eps=1e-12):
                    n_patients = pids.max() + 1
                    m = len(unique_bins)
                    h = np.zeros((n_patients, m), dtype=float)
                    for j, t in enumerate(unique_bins):
                        idx_at_t = np.where(np.isclose(X_test["eval_time"], t))[0]
                        for idx in idx_at_t:
                            pid = pids[idx]
                            pA, pB, pC, pD = probs[idx]
                            denom = pB + pC + pD + eps   # at-risk at t
                            h[pid, j] = pD / denom
                    return np.clip(h, 0.0, 1.0)

                h_unique = hazards_from_probs(X_tabpfn_test, test_patient_ids, probs, unique_bins)

                # Build survival on the model’s native grid
                S_unique = np.clip(np.cumprod(1.0 - h_unique, axis=1), 0.0, 1.0)   # (n_test, len(unique_bins))

                # Interpolate to ORIGINAL 'times' grid
                if n_bins is None:
                    S_full = S_unique[:, inv]
                else:
                    bin_t_grid = cuts[unique_bins]  # <-- use KM right cuts
                    S_full = np.vstack([
                        np.interp(times, bin_t_grid, S_unique[i], left=1.0, right=S_unique[i, -1])
                        for i in range(S_unique.shape[0])
                    ])
                S_full = np.minimum.accumulate(S_full, axis=1)

                # Risk for ranking: probability of event by horizon
                risk_scores = 1.0 - S_full[:, -1]

                # Metrics
                c_index, *_ = concordance_index_censored(y_test_orig["event"], y_test_orig["time"], risk_scores)
                ibs = integrated_brier_score(y_train_orig, y_test_orig, S_full, times)
                _, mean_auc = cumulative_dynamic_auc(y_train_orig, y_test_orig, risk_scores, times=times)

                if c_index > best_cindex:
                    best_cindex = c_index
                    best_row = {
                        "dataset": dataset_name,
                        "n_bins": n_bins if n_bins is not None else "None",
                        "n_neighbors": n_neighbors,
                        "c_index": round(float(c_index), 4),
                        "ibs": round(float(ibs), 4),
                        "mean_auc": round(float(mean_auc), 4),
                    }
        print("="*50)
        print(f"Best results for dataset {dataset_name}:")
        print(f"Best n_bins: {best_row['n_bins']}, Best n_neighbors: {best_row['n_neighbors']}")
        print(f"Best C-index: {best_row['c_index']:.4f}")
        print(f"Best interval Brier Score (IBS): {best_row['ibs']:.4f}")
        print(f"Best mean AUC: {best_row['mean_auc']:.4f}")

        # ========================= write best per dataset =========================
        if best_row is not None:
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["dataset", "n_bins", "n_neighbors", "c_index", "ibs", "mean_auc"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(best_row)

    except Exception as e:
        print(f"Error: {e}")
        continue
