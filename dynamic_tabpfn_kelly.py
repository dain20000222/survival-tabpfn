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
    before that evaluation time. If no row exists, create a default row with
    provided default values (already in processed feature space).

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
                # Create default row with provided default values (features + event)
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


def build_xy_for_pids(df: pd.DataFrame, feature_cols, pids):
    sub = df[df["pid"].isin(pids)].copy()
    X = sub[feature_cols].to_numpy(dtype=float)
    y = sub["event"].to_numpy(dtype=int)
    idx = sub.index.to_numpy()
    return X, y, idx

def online_predict_survival_conditional(eval_df, feature_cols, pid_test, base_X, base_y,
                                       device="cuda", history_window=None):
    """
    Survival prediction using conditional probabilities.
    For each patient, compute p_k = P(Y_k = 1 | X_k, Y_1=0, ..., Y_{k-1}=0)
    and derive survival probabilities S(τ_k).
    """
    results = []

    # Filter to test patients and sort by patient and eval_time
    test_df = eval_df[eval_df["pid"].isin(pid_test)].copy()
    test_df = test_df.sort_values(["pid", "eval_time"])

    # Group by patient and predict each eval_time sequentially
    for pid, grp in test_df.groupby("pid", sort=False):
        grp = grp.copy().sort_values("eval_time")
        survival_prob = 1.0  # S(τ_0) = 1 (initially alive)
        
        # Track revealed data for this patient (only non-event observations)
        revealed_X_list = []
        revealed_y_list = []

        for k, (ridx, row) in enumerate(grp.iterrows(), 1):
            x_query = row[feature_cols].to_numpy(dtype=float)[None, :]

            # Build augmented context: base + revealed prefix (only survival observations)
            if revealed_X_list:
                if (history_window is not None) and (len(revealed_X_list) > history_window):
                    revealed_X_list = revealed_X_list[-history_window:]
                    revealed_y_list = revealed_y_list[-history_window:]

                X_aug = np.vstack([base_X] + revealed_X_list)
                y_aug = np.concatenate([base_y] + revealed_y_list)
            else:
                X_aug, y_aug = base_X, base_y

            # Fit on augmented context (conditioned on survival), then predict current query
            clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
            clf.fit(X_aug, y_aug)
            proba = clf.predict_proba(x_query)[0]
            
            # p_k = P(Y_k = 1 | X_k, Y_1=0, ..., Y_{k-1}=0)
            p_k = float(proba[1]) if proba.shape[0] > 1 else float(proba[0])
            
            # Solve for S(τ_k) using: 1 - S(τ_k) = p_k * S(τ_{k-1}) + (1 - S(τ_{k-1}))
            # Rearranging: 1 - S(τ_k) = p_k * S(τ_{k-1}) + 1 - S(τ_{k-1})
            # => 1 - S(τ_k) = 1 - S(τ_{k-1}) * (1 - p_k)
            # => S(τ_k) = S(τ_{k-1}) * (1 - p_k)
            new_survival_prob = survival_prob * (1 - p_k)
            
            # Risk at this timepoint is the cumulative hazard up to τ_k
            cumulative_risk = 1 - new_survival_prob
            
            # Discrete hazard (instantaneous risk) at this interval
            if survival_prob > 0:
                discrete_hazard = p_k
            else:
                discrete_hazard = 1.0  # If already dead, hazard is 1
            
            yhat = 1 if p_k > 0.5 else 0

            results.append({
                "pid": pid,
                "row_index": ridx,
                "eval_time": row["eval_time"],
                "time_step": k,
                "y_true": int(row["event"]),
                "y_pred": yhat,
                "p_k": p_k,  # Conditional probability of event at time k
                "surv_prob": new_survival_prob,  # S(τ_k)
                "cumulative_risk": cumulative_risk,  # 1 - S(τ_k)
                "discrete_hazard": discrete_hazard,  # Instantaneous risk
                "risk": cumulative_risk  # For compatibility with existing code
            })

            # Update survival probability for next iteration
            survival_prob = new_survival_prob

            # IMPORTANT: Only add to revealed data if patient survived (y=0)
            # This maintains the conditioning Y_1=0, ..., Y_{k-1}=0 for future predictions
            y_true = int(row["event"])
            if y_true == 0:  # Patient survived this time step
                x_true = x_query
                y_true_array = np.array([y_true], dtype=int)
                revealed_X_list.append(x_true)
                revealed_y_list.append(y_true_array)
            else:
                # Patient had event - stop adding to revealed data but continue prediction
                # (in practice, this patient would not have future observations)
                pass

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
    
    # Autoregressive prediction on test evaluation data using survival conditioning
    risk_df = online_predict_survival_conditional(
        eval_df, feature_cols, pid_test, X_base, y_base,
        device="cuda", history_window=None
    )

    print("Survival predictions:")
    print(risk_df.head(10))
    print(f"Columns: {risk_df.columns.tolist()}")

    # Save risk_df to CSV
    risk_df['dataset'] = dataset_name
    risk_df.to_csv(risk_csv_path, mode='a', header=not os.path.exists(risk_csv_path), index=False)
    
    save_checkpoint(dataset_name)
