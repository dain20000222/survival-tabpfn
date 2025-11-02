import pandas as pd
import numpy as np
import os
import csv
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tabpfn import TabPFNClassifier
from sksurv.metrics import concordance_index_ipcw, brier_score
from sksurv.util import Surv
import warnings
import random
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Directory containing the datasets
data_dir = os.path.join("data")

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# CSV path
csv_path = "dynamic_tabpfn_evaluation.csv"

def prepare_data(df: pd.DataFrame):
    df = df.copy()

    # Ensure sorting within patient
    sort_cols = ["pid", "time"] + (["time2"] if "time2" in df.columns else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)

    num_cols = [c for c in df.columns if c.startswith("num_")]
    fac_cols = [c for c in df.columns if c.startswith("fac_")]

    # Ordinal-encode categoricals to integers (TabPFN expects numeric inputs)
    enc = None
    if fac_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
        df[fac_cols] = enc.fit_transform(df[fac_cols].astype("category"))

    feature_cols = num_cols + fac_cols + ["time"] + (["time2"] if "time2" in df.columns else [])

    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df["event"].to_numpy(dtype=int)

    return df, feature_cols, X_all, y_all, enc

def split_by_pid(df: pd.DataFrame, test_size=0.3, random_state=42):
    pids = df["pid"].unique()
    pid_train, pid_test = train_test_split(pids, test_size=test_size, random_state=random_state)

    return pid_train, pid_test

def build_xy_for_pids(df: pd.DataFrame, feature_cols, pids):
    sub = df[df["pid"].isin(pids)].copy()
    X = sub[feature_cols].to_numpy(dtype=float)
    y = sub["event"].to_numpy(dtype=int)
    idx = sub.index.to_numpy()

    return X, y, idx

def train_base(df, feature_cols, pid_train, device="cuda"):
    X_train, y_train, _ = build_xy_for_pids(df, feature_cols, pid_train)
    clf = TabPFNClassifier(
        device=device
    )
    clf.fit(X_train, y_train)

    return clf, X_train, y_train

def online_predict_autoregressive(df, feature_cols, pid_test, base_X, base_y,
                                  device="cuda", history_window=None):
    """
    For each patient:
      - predict interval 1 using base train
      - then append TRUE (X, y) of interval 1 to the context
      - predict interval 2, append its TRUE (X, y), ...
    If history_window is not None, keep only the last K revealed intervals per patient.
    Returns a dataframe with y_pred, y_proba for all test rows.
    """
    results = []

    # We re-create a classifier handle per patient-step to ensure fresh conditioning.
    test_df = df[df["pid"].isin(pid_test)].copy()
    test_df = test_df.sort_values(["pid", "time"] + (["time2"] if "time2" in test_df.columns else []))

    # group rows by patient in chronological order
    for pid, grp in test_df.groupby("pid", sort=False):
        grp = grp.copy()
        revealed_X_list = []
        revealed_y_list = []

        for ridx, row in grp.iterrows():
            x_query = row[feature_cols].to_numpy(dtype=float)[None, :]

            # Build augmented context: base + revealed prefix for THIS patient
            if revealed_X_list:
                if (history_window is not None) and (len(revealed_X_list) > history_window):
                    revealed_X_list = revealed_X_list[-history_window:]
                    revealed_y_list = revealed_y_list[-history_window:]

                X_aug = np.vstack([base_X] + revealed_X_list)
                y_aug = np.concatenate([base_y] + revealed_y_list)
            else:
                X_aug, y_aug = base_X, base_y
            
            print(f"Patient {pid}, row {ridx}: Augmented context size: {X_aug.shape[0]}")

            # Light "re-conditioning": fit on augmented context, then predict current query
            clf = TabPFNClassifier(device=device)
            clf.fit(X_aug, y_aug)
            proba = clf.predict_proba(x_query)[0]
            yhat = int(proba.argmax())

            results.append({
                "pid": pid,
                "row_index": ridx,
                "time": row["time"],
                **({"time2": row["time2"]} if "time2" in row else {}),
                "y_true": int(row["event"]),
                "y_pred": yhat,
                "p_event": float(proba[1]) if proba.shape[0] > 1 else float(proba[0])  # binary
            })

            # APPEND THE TRUE (X,y) from this interval to the revealed prefix
            x_true = x_query  # identical features at this row
            y_true = np.array([row["event"]], dtype=int)
            revealed_X_list.append(x_true)
            revealed_y_list.append(y_true)

    out = pd.DataFrame(results).sort_values(["pid", "time"])
    return out

def subject_time_event(df: pd.DataFrame, pids):
    """Return Surv array and an index of pids in order, aggregated to subject-level."""
    sub = df[df["pid"].isin(pids)].copy()
    has_time2 = "time2" in sub.columns

    recs = []
    for pid, g in sub.sort_values(["pid", "time"] + (["time2"] if has_time2 else [])).groupby("pid", sort=False):
        if (g["event"] == 1).any():
            # take the (last) event row as the event time
            ge = g[g["event"] == 1].iloc[-1]
            t = float(ge["time2"] if has_time2 else ge["time"])
            e = True
        else:
            # censored at the last observed end time
            t = float((g["time2"] if has_time2 else g["time"]).max())
            e = False
        recs.append((pid, e, t))

    order_pids = [r[0] for r in recs]
    y_surv = Surv.from_arrays(event=[r[1] for r in recs], time=[r[2] for r in recs])
    return y_surv, order_pids

def patient_risk_from_online(online_out: pd.DataFrame):
    """
    Convert per-interval p_event into a single per-patient risk score:
      risk_i = 1 - prod_t (1 - p_event_{i,t})
    """
    risk_rows = []
    has_time2 = "time2" in online_out.columns
    for pid, g in online_out.sort_values(["pid", "time"] + (["time2"] if has_time2 else [])).groupby("pid", sort=False):
        s = np.prod(1.0 - g["p_event"].values)  # cumulative survival up to the last interval
        risk = float(1.0 - s)
        risk_rows.append((pid, risk))
    pid_to_risk = dict(risk_rows)
    return pid_to_risk

for file_name in dataset_files:
    dataset_name = file_name.replace(".csv", "")
    file_path = os.path.join(data_dir, file_name)
    print("="*50)
    print(f"\nProcessing dataset: {dataset_name}")

    # Load datasets
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")

    # Define evaluation times (unique event times)
    event_times = df[df['event'] == 1]['time2'].values
    times = np.quantile(event_times, [0.25, 0.5, 0.75])

    # Prepare data
    df, feature_cols, X_all, y_all, enc = prepare_data(df)

    # Split by patient ID
    pid_train, pid_test = split_by_pid(df, test_size=0.3, random_state=42)

    # Train base TabPFN model
    clf_base, base_X, base_y = train_base(df, feature_cols, pid_train, device="cuda")
    
    # Online autoregressive prediction
    online_out = online_predict_autoregressive(
        df, feature_cols, pid_test, base_X, base_y,
        device="cuda", history_window=None
    )

    print(online_out.head())

     # 1) Subject-level (event, time) for train/test
    y_train_surv, train_pid_order = subject_time_event(df, pid_train)
    y_test_surv, test_pid_order = subject_time_event(df, pid_test)

    print(f"Example y_train_surv: {y_train_surv[:5]}")
    print(f"Example y_test_surv: {y_test_surv[:5]}")

    # 2) Convert interval probabilities to patient-level risks
    pid2risk = patient_risk_from_online(online_out)
    risk_test = np.array([pid2risk[pid] for pid in test_pid_order], dtype=float)

    print(f"Example test risks: {[pid2risk[pid] for pid in test_pid_order[:5]]}")
    print(f"Example test risks: {risk_test[:5]}")

    # 3) IPCW C-index (uses training set to estimate censoring weights)
    c_ipcw, *_ = concordance_index_ipcw(y_train_surv, y_test_surv, risk_test)

    print(f"IPCW C-index: {c_ipcw:.4f}")

    # Compute time-specific C-index for each evaluation time
    c_ipcw_times = []
    for t in times:
        try:
            c_t, *_ = concordance_index_ipcw(y_train_surv, y_test_surv, risk_test, tau=t)
            c_ipcw_times.append(c_t)
            print(f"IPCW C-index at time {t:.2f}: {c_t:.4f}")
        except Exception as e:
            print(f"Could not compute C-index at time {t:.2f}: {e}")
            c_ipcw_times.append(np.nan)

    # You can also compute the mean time-specific C-index
    mean_c_ipcw_time = np.nanmean(c_ipcw_times) if c_ipcw_times else np.nan
    print(f"Mean time-specific IPCW C-index: {mean_c_ipcw_time:.4f}")

    # 4) Append to CSV
    out_row = {
        "dataset": dataset_name,
        "n_train_pids": len(pid_train),
        "n_test_pids": len(pid_test),
        "ipcw_cindex": float(c_ipcw),
    }

    # write header if file doesn't exist
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(out_row)

