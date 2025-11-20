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

# CSV paths and checkpoint setup
csv_path = "dynamic_tabpfn_evaluation_3timepoint.csv"
checkpoint_file = "dynamic_tabpfn_checkpoint.txt"
risk_csv_path = "dynamic_tabpfn_risks_3timepoint.csv"

# Load processed datasets from checkpoint
def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed = [line.strip() for line in f.readlines()]
        print(f"Loaded checkpoint: {len(processed)} datasets already processed")
        return set(processed)
    return set()

# Save checkpoint
def save_checkpoint(dataset_name):
    with open(checkpoint_file, 'a') as f:
        f.write(f"{dataset_name}\n")
    print(f"Checkpoint saved for dataset: {dataset_name}")

# Load already processed datasets
processed_datasets = load_checkpoint()

# Filter out already processed datasets
remaining_datasets = [f for f in dataset_files if f.replace(".csv", "") not in processed_datasets]

print(f"Total datasets: {len(dataset_files)}")
print(f"Already processed: {len(processed_datasets)}")
print(f"Remaining to process: {len(remaining_datasets)}")

if remaining_datasets:
    print(f"Next datasets to process: {remaining_datasets}")  # Show all remaining datasets
else:
    print("All datasets have been processed!")

def prepare_evaluation_data(df: pd.DataFrame, times):
    """
    For each evaluation time, select the latest observed row for each patient
    before that evaluation time. If no row exists, create a default row with
    average values from all patients.
    """
    eval_rows = []
    has_time2 = "time2" in df.columns
    time_col = "time2" if has_time2 else "time"
    
    # Calculate default values (averages) for all features
    num_cols = [c for c in df.columns if c.startswith("num_")]
    fac_cols = [c for c in df.columns if c.startswith("fac_")]
    
    default_values = {}
    for col in num_cols:
        default_values[col] = df[col].mean()
    for col in fac_cols:
        default_values[col] = df[col].mode().iloc[0] if not df[col].mode().empty else df[col].iloc[0]
    
    # Set default event to 0 (no event)
    default_values["event"] = 0
    
    for eval_time in times:
        for pid in df["pid"].unique():
            patient_data = df[df["pid"] == pid].copy()
            # Get rows observed before or at eval_time
            eligible_rows = patient_data[patient_data[time_col] <= eval_time]
            
            if len(eligible_rows) > 0:
                # Select the latest observed row
                latest_row = eligible_rows.loc[eligible_rows[time_col].idxmax()].copy()
                latest_row["eval_time"] = eval_time
                eval_rows.append(latest_row)
            else:
                # Create default row with average values
                default_row = default_values.copy()
                default_row["pid"] = pid
                default_row["eval_time"] = eval_time
                default_row[time_col] = eval_time  # Set time to eval_time
                
                # Convert to Series to match expected format
                default_row = pd.Series(default_row)
                eval_rows.append(default_row)
                print(f"Patient {pid} has no data before eval_time {eval_time}, using default row.")
    
    eval_df = pd.DataFrame(eval_rows).reset_index(drop=True)
    return eval_df

def prepare_data_for_evaluation(df: pd.DataFrame):
    """Modified prepare_data function for evaluation dataset"""
    df = df.copy()
    
    # Sort by patient and evaluation time
    df = df.sort_values(["pid", "eval_time"]).reset_index(drop=True)
    
    num_cols = [c for c in df.columns if c.startswith("num_")]
    fac_cols = [c for c in df.columns if c.startswith("fac_")]
    
    # Ordinal-encode categoricals to integers
    enc = None
    if fac_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
        df[fac_cols] = enc.fit_transform(df[fac_cols].astype("category"))
    
    # Use eval_time instead of time/time2
    feature_cols = num_cols + fac_cols + ["eval_time"]
    
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
    clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
    clf.fit(X_train, y_train)
    return clf, X_train, y_train

def online_predict_autoregressive_eval(eval_df, feature_cols, pid_test, base_X, base_y,
                                      device="cuda", history_window=None):
    """
    Autoregressive prediction on evaluation dataset.
    For each patient, predict eval_time 1, then eval_time 2, then eval_time 3.
    """
    results = []

    # Filter to test patients and sort by patient and eval_time
    test_df = eval_df[eval_df["pid"].isin(pid_test)].copy()
    test_df = test_df.sort_values(["pid", "eval_time"])

    # Group by patient and predict each eval_time sequentially
    for pid, grp in test_df.groupby("pid", sort=False):
        grp = grp.copy().sort_values("eval_time")
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

            # Fit on augmented context, then predict current query
            clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
            clf.fit(X_aug, y_aug)
            proba = clf.predict_proba(x_query)[0]
            yhat = int(proba.argmax())

            results.append({
                "pid": pid,
                "row_index": ridx,
                "eval_time": row["eval_time"],
                "y_true": int(row["event"]),
                "y_pred": yhat,
                "p_event": float(proba[1]) if proba.shape[0] > 1 else float(proba[0])
            })

            # APPEND THE TRUE (X,y) from this evaluation time to the revealed prefix
            x_true = x_query
            y_true = np.array([row["event"]], dtype=int)
            revealed_X_list.append(x_true)
            revealed_y_list.append(y_true)

    out = pd.DataFrame(results).sort_values(["pid", "eval_time"])
    return out

def subject_time_event_eval(df: pd.DataFrame, pids, original_df: pd.DataFrame):
    """
    Return Surv array for evaluation, using original survival times from original_df
    """
    has_time2 = "time2" in original_df.columns
    time_col = "time2" if has_time2 else "time"
    
    recs = []
    for pid in pids:
        patient_orig = original_df[original_df["pid"] == pid].copy()
        
        if (patient_orig["event"] == 1).any():
            # Take the (last) event row as the event time
            event_rows = patient_orig[patient_orig["event"] == 1]
            ge = event_rows.loc[event_rows[time_col].idxmax()]
            t = float(ge[time_col])
            e = True
        else:
            # Censored at the last observed time
            t = float(patient_orig[time_col].max())
            e = False
        recs.append((pid, e, t))
    
    order_pids = [r[0] for r in recs]
    y_surv = Surv.from_arrays(event=[r[1] for r in recs], time=[r[2] for r in recs])
    return y_surv, order_pids

def patient_risk_from_online_eval_time_dependent(online_out: pd.DataFrame, times):
    """
    Convert per-eval-time p_event into time-dependent risk scores.
    Each patient gets a separate risk score for each evaluation time.
    """
    risk_rows = []
    for pid, g in online_out.sort_values(["pid", "eval_time"]).groupby("pid", sort=False):
        g = g.sort_values("eval_time")
        cumulative_survival = 1.0
        
        for i, (_, row) in enumerate(g.iterrows()):
            # Update cumulative survival up to this time point
            cumulative_survival *= (1.0 - row["p_event"])
            # Risk at this time point
            risk_at_time = 1.0 - cumulative_survival
            
            risk_rows.append({
                "pid": pid,
                "eval_time": row["eval_time"],
                "risk": risk_at_time
            })
    
    risk_df = pd.DataFrame(risk_rows)
    return risk_df

for file_name in dataset_files:
    dataset_name = file_name.replace(".csv", "")
    file_path = os.path.join(data_dir, file_name)
    print("="*50)
    print(f"\nProcessing dataset: {dataset_name} ({remaining_datasets.index(file_name) + 1}/{len(remaining_datasets)})")

    # Load original dataset
    original_df = pd.read_csv(file_path)
    print(f"Original dataset shape: {original_df.shape}")

    # Define evaluation times (unique event times)
    has_time2 = "time2" in original_df.columns
    time_col = "time2" if has_time2 else "time"
    event_times = original_df[original_df['event'] == 1][time_col].values
    times = np.quantile(event_times, [0.25, 0.5, 0.75])
    print(f"Evaluation times: {times}")

    # Create evaluation dataset with latest rows before each eval time
    eval_df = prepare_evaluation_data(original_df, times)
    print(f"Evaluation dataset shape: {eval_df.shape}")
    print(f"Rows per patient: {len(eval_df) / len(original_df['pid'].unique()):.1f}")

    # Prepare evaluation data
    eval_df, feature_cols, X_all, y_all, enc = prepare_data_for_evaluation(eval_df)

    # Split by patient ID (using original patient IDs)
    original_pids = original_df["pid"].unique()
    pid_train, pid_test = train_test_split(original_pids, test_size=0.3, random_state=42)

    # Train base TabPFN model on training evaluation data
    train_eval_df = eval_df[eval_df["pid"].isin(pid_train)].copy()
    X_base, y_base, _ = build_xy_for_pids(train_eval_df, feature_cols, pid_train)
    
    clf_base = TabPFNClassifier(device="cuda")
    clf_base.fit(X_base, y_base)
    
    # Autoregressive prediction on test evaluation data
    online_out = online_predict_autoregressive_eval(
        eval_df, feature_cols, pid_test, X_base, y_base,
        device="cuda", history_window=None
    )

    print("Online prediction results:")
    print(online_out.head())

    # Get survival data from original dataset
    y_train_surv, train_pid_order = subject_time_event_eval(train_eval_df, pid_train, original_df)

    # Only use test patients that actually have evaluation data
    actual_test_pids = eval_df[eval_df["pid"].isin(pid_test)]["pid"].unique()
    y_test_surv, test_pid_order = subject_time_event_eval(
        eval_df[eval_df["pid"].isin(actual_test_pids)], 
        actual_test_pids,  # Use only patients with eval data
        original_df
    )

    print(f"Original test patients: {len(pid_test)}")
    print(f"Test patients with eval data: {len(actual_test_pids)}")
    print(f"Example y_train_surv: {y_train_surv[:5]}")
    print(f"Example y_test_surv: {y_test_surv[:5]}")

    # Convert interval probabilities to time-dependent patient-level risks
    risk_df = patient_risk_from_online_eval_time_dependent(online_out, times)
    print(f"Example time-dependent risks:")
    print(risk_df.head(10))

    # Save risk_df to CSV
    risk_df['dataset'] = dataset_name
    risk_df.to_csv(risk_csv_path, mode='a', header=not os.path.exists(risk_csv_path), index=False)

    # Compute time-specific C-index for each evaluation time
    c_ipcw_times = []
    for t in times:
        try:
            # Get risks at this specific evaluation time
            risks_at_t = risk_df[risk_df["eval_time"] == t]
            
            # Align with test patient order
            risk_test_at_t = np.array([
                risks_at_t[risks_at_t["pid"] == pid]["risk"].iloc[0] 
                for pid in test_pid_order
            ], dtype=float)
            
            c_t, *_ = concordance_index_ipcw(y_train_surv, y_test_surv, risk_test_at_t, tau=t)
            c_ipcw_times.append(c_t)
            print(f"IPCW C-index at time {t:.2f}: {c_t:.4f}")
        except Exception as e:
            print(f"Could not compute C-index at time {t:.2f}: {e}")
            c_ipcw_times.append(np.nan)

    # Save to CSV with uniform format
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['dataset', 'time', 'cindex']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Write each evaluation time result
        for i, (eval_time, c_index) in enumerate(zip(times, c_ipcw_times)):
            uniform_result = {
                'dataset': dataset_name,
                'time': eval_time,
                'cindex': c_index
            }
            writer.writerow(uniform_result)
    
    save_checkpoint(dataset_name)