import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.linear_model import CoxPHSurvivalAnalysis
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

# CSV paths
risk_csv_path = "landmark_cox_risk_5.csv"

# Initialize risk CSV file
risk_file_exists = os.path.isfile(risk_csv_path)
if not risk_file_exists:
    with open(risk_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['pid', 'eval_time', 'risk', 'surv_prob', 'dataset']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def build_patient_outcome(df):
    """
    For each pid, get final follow-up time (t_event) and status_event (1 if event, else 0).
    Assumes last row per pid contains correct final event/censor info.
    """
    outcome = (
        df.sort_values("time2")
        .groupby("pid")
        .agg(
            t_event = ("time2", "max"),
            status_event = ("event", "last")
        )
    )
    return outcome

def apply_locf_landmark(df, landmark_time, covariates, patient_outcome, default_values):
    """
    Apply LOCF landmarking for a given landmark_time.

    - If t_event > landmark_time: residual_time = t_event - landmark_time, event as original
    - If t_event <= landmark_time: residual_time = 0 (or ~0), event = 0 (not at risk after landmark)
    - Covariates taken from last observation before (or at) landmark; if none, use defaults.
    """
    rows = []
    for pid, patient in df.groupby("pid"):
        patient = patient.sort_values("time2")
        t_event = patient_outcome.loc[pid, "t_event"]
        status_event = patient_outcome.loc[pid, "status_event"]

        # Case A: patient experiences event AFTER the landmark -> normal residual time
        if t_event > landmark_time:
            residual_time = float(t_event - landmark_time)
            event_indicator = int(status_event)
        else:
            # Case B: event before (or at) landmark -> keep but set as NOT at risk
            residual_time = 0.0
            event_indicator = 0

        # LOCF covariates up to landmark_time
        valid = patient[patient["time2"] <= landmark_time]
        if not valid.empty:
            latest = valid.iloc[-1]
            row = {"pid": pid, "time": residual_time, "event": event_indicator}
            for c in covariates:
                row[c] = latest[c]
        else:
            # No observation → use defaults
            row = {"pid": pid, "time": residual_time, "event": event_indicator}
            for c in covariates:
                row[c] = default_values[c]

        rows.append(row)

    return pd.DataFrame(rows)


for file_name in dataset_files:
    try:
        dataset_name = file_name.replace(".csv", "")
        file_path = os.path.join(data_dir, file_name)
        print("=" * 50)
        print(f"\nProcessing dataset: {dataset_name}")

        # Load dataset
        df = pd.read_csv(file_path)
        shape = df.shape
        print(f"Dataset shape: {shape}")

        # Define columns
        time_col = "time"
        time2_col = "time2"
        event_col = "event"

        covariates = df.columns.difference([time_col, time2_col, event_col, 'pid']).tolist()

        # Get event times for landmark calculation (on original scale)
        event_times = df[df[event_col] == 1]["time2"].values
        if len(event_times) == 0:
            print(f"Skipping {dataset_name}: No events observed")
            continue

        # Pick three landmark times (25, 50, 75 quantiles of event times)
        times = np.quantile(event_times, [0.1, 0.3, 0.5, 0.7, 0.9])
        print(f"Landmark times: {times}")

        # Split patient IDs into train (70%) and test (30%)
        all_pids = df['pid'].unique()
        pids_train, pids_test = train_test_split(
            all_pids, test_size=0.3, random_state=SEED
        )

        # Split dataframe by patient IDs
        df_train = df[df['pid'].isin(pids_train)].copy()
        df_test = df[df['pid'].isin(pids_test)].copy()

        print(f"Train patients: {len(pids_train)}, Test patients: {len(pids_test)}")

        # ========== PREPROCESS DATA BEFORE LANDMARKING ==========
        # --- One-hot encode categorical variables ---
        # Fit on TRAIN only
        x_train = pd.get_dummies(df_train[covariates], drop_first=True)
        x_test = pd.get_dummies(df_test[covariates], drop_first=True)

        # Align columns with union of train+test
        all_columns = x_train.columns.union(x_test.columns)
        x_train = x_train.reindex(columns=all_columns, fill_value=0)
        x_test = x_test.reindex(columns=all_columns, fill_value=0)

        # --- Impute missing values ---
        imputer = SimpleImputer(strategy='median')

        x_train = pd.DataFrame(imputer.fit_transform(x_train),
                            columns=x_train.columns,
                            index=x_train.index)

        x_test = pd.DataFrame(imputer.transform(x_test),
                            columns=x_test.columns,
                            index=x_test.index)

        # Calculate default values from PROCESSED train set
        default_values = {}
        for col in x_train.columns:
            default_values[col] = x_train[col].mean()

        print(f"Default values calculated from processed train set ({len(default_values)} features)")

        # Update dataframes with processed features
        df_train_processed = df_train[['pid', 'time', 'time2', 'event']].copy()
        df_train_processed = df_train_processed.join(x_train)

        df_test_processed = df_test[['pid', 'time', 'time2', 'event']].copy()
        df_test_processed = df_test_processed.join(x_test)

        processed_feature_cols = list(x_train.columns)

        # Build patient outcome DataFrame (from full data)
        patient_outcome = build_patient_outcome(df)

        # Dictionary to store risk scores and survival probabilities for all test patients
        patient_risks_dict = {pid: [] for pid in pids_test}

        for i, landmark_time in enumerate(times):
            print(f"\nProcessing landmark time {i + 1}: {landmark_time:.2f}")
                        
            mask = patient_outcome["t_event"] > landmark_time
            residual_times = patient_outcome.loc[mask, "t_event"] - landmark_time
            prediction_horizon = float(np.median(residual_times))

            # Apply LOCF landmarking for both train and test (using processed features)
            train_df = apply_locf_landmark(
                df_train_processed, landmark_time, processed_feature_cols, 
                patient_outcome, default_values
            )

            test_df = apply_locf_landmark(
                df_test_processed, landmark_time, processed_feature_cols, 
                patient_outcome, default_values
            )

            print(f"Train: {len(train_df)} patients, Events: {train_df['event'].sum()}")
            print(f"Test:  {len(test_df)} patients, Events: {test_df['event'].sum()}")

            if len(train_df) < 10 or train_df['event'].sum() < 5:
                print(f"Skipping landmark {landmark_time:.2f}: insufficient train data")
                continue

            if len(test_df) < 5:
                print(f"Skipping landmark {landmark_time:.2f}: insufficient test data")
                continue

            # Extract covariates (already processed)
            x_train_landmark = train_df[processed_feature_cols].copy()
            x_test_landmark = test_df[processed_feature_cols].copy()

            # Remove constant columns to avoid singularity
            constant_cols = x_train_landmark.columns[x_train_landmark.std() == 0]
            if len(constant_cols) > 0:
                print(f"Removing {len(constant_cols)} constant columns")
                x_train_landmark = x_train_landmark.drop(columns=constant_cols)
                x_test_landmark = x_test_landmark.drop(columns=constant_cols)

            # Prepare survival data (residual time since landmark)
            y_train = Surv.from_dataframe('event', 'time', train_df)
            y_test = Surv.from_dataframe('event', 'time', test_df)

            # Fit Cox model at this landmark
            cox_model = CoxPHSurvivalAnalysis()
            try:
                cox_model.fit(x_train_landmark, y_train)
            except Exception as e:
                print(f"Error fitting Cox model at landmark {landmark_time:.2f}: {e}")
                continue

            # Get linear predictors (risk scores) for test set
            test_risk_scores = cox_model.predict(x_test_landmark)

            # Get survival functions for test patients at this landmark
            surv_funcs = cox_model.predict_survival_function(x_test_landmark)

            # Store risk scores and survival probabilities for each test patient at this landmark time
            for idx, pid in enumerate(test_df['pid'].values):
                risk_score = test_risk_scores[idx]
                surv_func = surv_funcs[idx]
                surv_prob = surv_func(prediction_horizon)

                patient_risks_dict[pid].append((landmark_time, risk_score, surv_prob))

            print(f"Predictions saved for {len(test_df)} patients at landmark {landmark_time:.2f}")

        # Save all risk predictions to CSV (grouped by patient)
        with open(risk_csv_path, 'a', newline='') as csvfile:
            fieldnames = ['pid', 'eval_time', 'risk', 'surv_prob', 'dataset']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for pid in sorted(pids_test):
                if pid in patient_risks_dict and len(patient_risks_dict[pid]) > 0:
                    for eval_time, risk, surv_prob in patient_risks_dict[pid]:
                        writer.writerow({
                            'pid': pid,
                            'eval_time': eval_time,  # landmark time
                            'risk': risk if np.isfinite(risk) else np.nan,
                            'surv_prob': surv_prob if np.isfinite(surv_prob) else np.nan,
                            'dataset': dataset_name,
                        })

        print(f"\nRisk predictions saved to {risk_csv_path}")

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue
