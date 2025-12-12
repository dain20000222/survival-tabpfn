import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tabpfn import TabPFNClassifier
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
checkpoint_file = "dynamic_tabpfn_kelly_checkpoint.txt"
risk_csv_path = "dynamic_tabpfn_kelly_risks.csv"

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

def prepare_evaluation_data(df: pd.DataFrame, times, default_values):
    """
    For each evaluation time, select the latest observed row for each patient
    before that evaluation time. For the FIRST eval_time per patient, always
    use default values. For subsequent eval_times, use latest observed data.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (processed, numeric)
    times : array-like
        Evaluation times
    default_values : dict
        Dictionary of default values for features (calculated from processed train data)
    """
    eval_rows = []
    has_time2 = "time2" in df.columns
    time_col = "time2" if has_time2 else "time"
    
    # Sort times to ensure we process them in order
    sorted_times = sorted(times)
    
    for eval_time_idx, eval_time in enumerate(sorted_times):
        for pid in df["pid"].unique():
            patient_data = df[df["pid"] == pid].copy()
            
            # For the first evaluation time, always use default values
            if eval_time_idx == 0:
                default_row = default_values.copy()
                default_row["pid"] = pid
                default_row["eval_time"] = eval_time
                default_row[time_col] = eval_time  # Set time to eval_time
                
                # Convert to Series to match expected format
                default_row = pd.Series(default_row)
                eval_rows.append(default_row)

            else:
                # For subsequent evaluation times, use latest observed row
                # Get the previous eval_time as the cutoff
                prev_eval_time = sorted_times[eval_time_idx - 1]
                
                # Get rows observed before or at previous eval_time AND before current eval_time
                eligible_rows = patient_data[
                    (patient_data[time_col] <= prev_eval_time) & 
                    (patient_data[time_col] < eval_time)
                ]
                
                if len(eligible_rows) > 0:
                    # Select the latest observed row
                    latest_row = eligible_rows.loc[eligible_rows[time_col].idxmax()].copy()
                    latest_row["eval_time"] = eval_time
                    eval_rows.append(latest_row)

                else:
                    # Create default row if no data exists before this eval_time
                    default_row = default_values.copy()
                    default_row["pid"] = pid
                    default_row["eval_time"] = eval_time
                    default_row[time_col] = eval_time  # Set time to eval_time
                    
                    # Convert to Series to match expected format
                    default_row = pd.Series(default_row)
                    eval_rows.append(default_row)
                    print(f"Patient {pid} has no data before prev_eval_time {prev_eval_time}, using default row.")
    
    eval_df = pd.DataFrame(eval_rows).reset_index(drop=True)
    return eval_df


def build_xy_for_pids(df: pd.DataFrame, feature_cols, pids):
    sub = df[df["pid"].isin(pids)].copy()
    X = sub[feature_cols].to_numpy(dtype=float)
    y = sub["event"].to_numpy(dtype=int)
    idx = sub.index.to_numpy()
    return X, y, idx

def online_predict_autoregressive_eval(eval_df, feature_cols, pid_test, base_X, base_y):
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
        patient_died = False  # Track if patient has died
        p_events = []  # Store p_event for each eval_time for this patient

        for ridx, row in grp.iterrows():
            x_query = row[feature_cols].to_numpy(dtype=float)[None, :]

            # If patient died in previous eval_time, set p_event = 1
            if patient_died:
                p_event = 1.0
                yhat = 1
            else:
                # Build augmented context: base + revealed prefix for THIS patient
                if revealed_X_list:
                    X_aug = np.vstack([base_X] + revealed_X_list)
                    y_aug = np.concatenate([base_y] + revealed_y_list)
                else:
                    X_aug, y_aug = base_X, base_y

                print(f"Augmented context for patient {pid} at eval_time {row['eval_time']}:")
                print("X_aug:")
                print(X_aug)
                print("y_aug:")
                print(y_aug)

                # Fit on augmented context, then predict current query
                clf = TabPFNClassifier(device="cuda", ignore_pretraining_limits=True)
                clf.fit(X_aug, y_aug)
                proba = clf.predict_proba(x_query)[0]
                yhat = int(proba.argmax())
                
                # p_event is the probability of event at this timepoint
                p_event = float(proba[1]) if proba.shape[0] > 1 else float(proba[0])

            # Store p_event for survival probability calculation
            p_events.append(p_event)
            
            # Calculate survival probabilities
            eval_time_idx = len(p_events) - 1  # 0-indexed
            
            # Calculate all survival probabilities based on current eval_time_idx
            if eval_time_idx == 0:
                # At τ1: S_τ0(τ1) = 1 - p_1
                surv_prob_tau0_tau1 = 1.0 - p_events[0]
                surv_prob = surv_prob_tau0_tau1
                
            elif eval_time_idx == 1:
                # At τ2: 
                # S_τ0(τ2) = (1-p_2) * S_τ0(τ1) = (1-p_2) * (1-p_1)
                # S_τ1(τ2) = 1 - p_2
                surv_prob_tau0_tau2 = (1.0 - p_events[1]) * (1.0 - p_events[0])
                surv_prob_tau1_tau2 = 1.0 - p_events[1]
                surv_prob = surv_prob_tau0_tau2  # Use τ0 as default
                
            elif eval_time_idx == 2:
                # At τ3:
                # S_τ0(τ3) = (1-p_3) * S_τ0(τ2) = (1-p_3) * (1-p_2) * (1-p_1)
                # S_τ1(τ3) = (1-p_3) * S_τ1(τ2) = (1-p_3) * (1-p_2)
                # S_τ2(τ3) = 1 - p_3
                surv_prob_tau0_tau3 = (1.0 - p_events[2]) * (1.0 - p_events[1]) * (1.0 - p_events[0])
                surv_prob_tau1_tau3 = (1.0 - p_events[2]) * (1.0 - p_events[1])
                surv_prob_tau2_tau3 = 1.0 - p_events[2]
                surv_prob = surv_prob_tau0_tau3  # Use τ0 as default

            # Create result entry with all relevant survival probabilities
            result_entry = {
                "pid": pid,
                "row_index": ridx,
                "eval_time": row["eval_time"],
                "y_true": int(row["event"]),
                "y_pred": yhat,
                "surv_prob": surv_prob,  # Default survival probability (from τ0)
                "risk": p_event  # Risk at this timepoint is just p_event
            }
            
            results.append(result_entry)

            # Check if patient died at this eval_time (y_true = 1)
            if int(row["event"]) == 1:
                patient_died = True

            # APPEND THE TRUE (X,y) from this evaluation time to the revealed prefix
            x_true = x_query
            y_true = np.array([row["event"]], dtype=int)
            revealed_X_list.append(x_true)
            revealed_y_list.append(y_true)

    out = pd.DataFrame(results).sort_values(["pid", "eval_time"])
    return out


for i, file_name in enumerate(remaining_datasets, start=1):
    dataset_name = file_name.replace(".csv", "")
    file_path = os.path.join(data_dir, file_name)
    print("=" * 50)
    print(f"\nProcessing dataset: {dataset_name} ({i}/{len(remaining_datasets)})")

    # Load original dataset
    original_df = pd.read_csv(file_path)
    print(f"Original dataset shape: {original_df.shape}")

    # Define evaluation times (quantiles of event times from ORIGINAL data)
    has_time2 = "time2" in original_df.columns
    time_col = "time2" if has_time2 else "time"
    event_times = original_df[original_df['event'] == 1][time_col].values
    times = np.quantile(event_times, [0.25, 0.5, 0.75])
    print(f"Evaluation times: {times}")

    # 1) TRAIN/TEST SPLIT (by patient ID)
    original_pids = original_df["pid"].unique()
    pid_train, pid_test = train_test_split(original_pids, test_size=0.3, random_state=SEED)
    print(f"Train patients: {len(pid_train)}, Test patients: {len(pid_test)}")

    # Identify raw feature columns (before preprocessing)
    num_cols = [c for c in original_df.columns if c.startswith("num_")]
    fac_cols = [c for c in original_df.columns if c.startswith("fac_")]
    raw_feature_cols = num_cols + fac_cols

    # 2) DATA PREPROCESSING (OHE + SimpleImputer, fit on TRAIN, transform TRAIN and TEST)

    train_df = original_df[original_df["pid"].isin(pid_train)].copy()
    test_df  = original_df[original_df["pid"].isin(pid_test)].copy()

    x_train_raw = train_df[raw_feature_cols]
    x_test_raw  = test_df[raw_feature_cols]

    # One-hot encode categorical features (pd.get_dummies)
    x_train_ohe = pd.get_dummies(x_train_raw, drop_first=True)
    x_test_ohe  = pd.get_dummies(x_test_raw,  drop_first=True)

    # Align columns between train and test
    all_columns = x_train_ohe.columns.union(x_test_ohe.columns)
    x_train_ohe = x_train_ohe.reindex(columns=all_columns, fill_value=0)
    x_test_ohe  = x_test_ohe.reindex(columns=all_columns, fill_value=0)

    # Impute missing values with median (fit on TRAIN, transform TRAIN and TEST)
    imputer = SimpleImputer(strategy="median")
    x_train_proc = pd.DataFrame(
        imputer.fit_transform(x_train_ohe),
        columns=all_columns,
        index=x_train_ohe.index,
    )
    x_test_proc = pd.DataFrame(
        imputer.transform(x_test_ohe),
        columns=all_columns,
        index=x_test_ohe.index,
    )

    # Rebuild fully processed longitudinal dataframes
    non_feature_cols = [c for c in original_df.columns if c not in raw_feature_cols]

    train_processed = pd.concat(
        [train_df[non_feature_cols].reset_index(drop=True),
         x_train_proc.reset_index(drop=True)],
        axis=1,
    )
    test_processed = pd.concat(
        [test_df[non_feature_cols].reset_index(drop=True),
         x_test_proc.reset_index(drop=True)],
        axis=1,
    )

    # Combine back into one processed longitudinal dataframe
    df_processed = pd.concat([train_processed, test_processed], ignore_index=True)

    # 3) CALCULATE DEFAULT VALUES (means from PROCESSED TRAIN patients)
    feature_cols_processed = all_columns.tolist()
    train_mask = df_processed["pid"].isin(pid_train)

    default_values = {}
    for col in feature_cols_processed:
        default_values[col] = df_processed.loc[train_mask, col].mean()

    # For event, explicitly set default to 0 (no event)
    default_values["event"] = 0

    print(f"Default values calculated from processed train data ({len(pid_train)} patients)")

    # 4) BUILD EVALUATION DATA (from PROCESSED longitudinal data)
    eval_df = prepare_evaluation_data(df_processed, times, default_values)
    print(f"Evaluation dataset shape: {eval_df.shape}")
    print(f"Rows per patient: {len(eval_df) / len(original_df['pid'].unique()):.1f}")

    # Sort evaluation dataframe and define final feature columns (processed features + eval_time)
    eval_df = eval_df.sort_values(["pid", "eval_time"]).reset_index(drop=True)
    feature_cols = feature_cols_processed + ["eval_time"]

    # Build base context from TRAIN patients using processed features
    train_eval_df = eval_df[eval_df["pid"].isin(pid_train)].copy()
    X_base, y_base, _ = build_xy_for_pids(train_eval_df, feature_cols, pid_train)
    
    # Autoregressive prediction on test evaluation data
    risk_df = online_predict_autoregressive_eval(
        eval_df, feature_cols, pid_test, X_base, y_base
    )

    print("Risk predictions:")
    print(risk_df.head(10))

    # Save risk_df to CSV
    risk_df['dataset'] = dataset_name
    risk_df.to_csv(risk_csv_path, mode='a', header=not os.path.exists(risk_csv_path), index=False)
    
    save_checkpoint(dataset_name)