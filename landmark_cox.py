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
risk_csv_path = "landmark_cox_risk.csv"

# Initialize risk CSV file
risk_file_exists = os.path.isfile(risk_csv_path)
if not risk_file_exists:
    with open(risk_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['pid', 'tau', 't', 'risk', 'surv_prob', 'dataset']
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
              t_event=("time2", "max"),
              status_event=("event", "last")
          )
    )
    return outcome

def apply_locf_landmark(df, landmark_time, covariates, patient_outcome, default_values):
    """
    Apply LOCF landmarking for a given landmark_time = tau_k (START of prediction window).

    - Keep only patients with t_event > landmark_time (at risk at tau_k)
    - Residual time = t_event - landmark_time
    - Covariates taken from last observation at or before landmark_time (LOCF).
      If none, use defaults.
    """
    rows = []
    for pid, patient in df.groupby("pid"):
        patient = patient.sort_values("time")  # covariate observation time

        t_event = patient_outcome.loc[pid, "t_event"]
        status_event = patient_outcome.loc[pid, "status_event"]

        # at risk at tau_k
        if t_event <= landmark_time:
            continue

        residual_time = float(t_event - landmark_time)
        event_indicator = int(status_event)

        # LOCF covariates up to landmark_time (NOT previous tau)
        valid = patient[patient["time"] <= landmark_time]
        row = {"pid": pid, "time": residual_time, "event": event_indicator}

        if not valid.empty:
            latest = valid.iloc[-1]
            for c in covariates:
                row[c] = latest[c]
        else:
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
        print(f"Dataset shape: {df.shape}")

        # Define columns
        time_col = "time"
        time2_col = "time2"
        event_col = "event"

        covariates = df.columns.difference([time_col, time2_col, event_col, 'pid']).tolist()

        # Build patient outcome DataFrame (from full data)
        patient_outcome = build_patient_outcome(df)

        # Get event times for landmark calculation (on original scale)
        event_times = df[df[event_col] == 1]["time2"].values
        if len(event_times) == 0:
            print(f"Skipping {dataset_name}: No events observed")
            continue

        # Pick three landmark times (25, 50, 75 quantiles of event times)
        τ1, τ2, τ3 = np.quantile(event_times, [0.25, 0.5, 0.75])

        # IMPORTANT: add tau0=0 as the baseline landmark
        τ0 = 0.0
        landmark_times = [τ0, float(τ1), float(τ2), float(τ3)]
        print(f"Landmark times: τ0={τ0:.2f}, τ1={τ1:.2f}, τ2={τ2:.2f}, τ3={τ3:.2f}")

        # Store predictions for all test patients across all landmarks
        patient_predictions = {}

        # ========== Process each landmark time ==========
        for idx, tau in enumerate(landmark_times):  # idx: 0..3 corresponds to tau0..tau3
            print(f"\n{'='*50}")
            print(f"Processing landmark time τ{idx} = {tau:.2f}")
            print(f"{'='*50}")

            # Filter patients who are still at risk at this landmark time
            at_risk_pids = patient_outcome[patient_outcome['t_event'] > tau].index.values
            print(f"Patients at risk at τ{idx}: {len(at_risk_pids)}")

            if len(at_risk_pids) < 10:
                print(f"✗ Skipping τ{idx}: not enough patients at risk")
                continue

            # Filter dataframe to only include at-risk patients
            df_at_risk = df[df['pid'].isin(at_risk_pids)].copy()
            patient_outcome_at_risk = patient_outcome.loc[at_risk_pids]

            # Split at-risk patient IDs into train (70%) and test (30%)
            pids_train, pids_test = train_test_split(
                at_risk_pids, test_size=0.3, random_state=SEED
            )

            df_train = df_at_risk[df_at_risk['pid'].isin(pids_train)].copy()
            df_test = df_at_risk[df_at_risk['pid'].isin(pids_test)].copy()

            print(f"Train patients: {len(pids_train)}, Test patients: {len(pids_test)}")

            # ========== PREPROCESS DATA BEFORE LANDMARKING ==========
            # One-hot encode categorical variables (fit on TRAIN only)
            x_train = pd.get_dummies(df_train[covariates], drop_first=True)
            x_test = pd.get_dummies(df_test[covariates], drop_first=True)

            # Align columns
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

            # Default values from processed TRAIN
            default_values = {col: x_train[col].mean() for col in x_train.columns}
            print(f"Default values calculated from processed train set ({len(default_values)} features)")

            # Update dataframes with processed features
            df_train_processed = df_train[['pid', 'time', 'time2', 'event']].copy().join(x_train)
            df_test_processed = df_test[['pid', 'time', 'time2', 'event']].copy().join(x_test)

            processed_feature_cols = list(x_train.columns)

            # Apply LOCF landmarking using covariates up to *tau* 
            train_df_tau = apply_locf_landmark(
                df_train_processed,
                landmark_time=tau,
                covariates=processed_feature_cols,
                patient_outcome=patient_outcome_at_risk,
                default_values=default_values
            )

            test_df_tau = apply_locf_landmark(
                df_test_processed,
                landmark_time=tau,
                covariates=processed_feature_cols,
                patient_outcome=patient_outcome_at_risk,
                default_values=default_values
            )

            print(f"Train: {len(train_df_tau)} patients, Events: {train_df_tau['event'].sum()}")
            print(f"Test:  {len(test_df_tau)} patients, Events: {test_df_tau['event'].sum()}")

            if len(train_df_tau) >= 10 and train_df_tau['event'].sum() >= 5 and len(test_df_tau) >= 5:
                x_train_tau = train_df_tau[processed_feature_cols].copy()
                x_test_tau = test_df_tau[processed_feature_cols].copy()

                # Remove constant columns
                constant_cols = x_train_tau.columns[x_train_tau.std() == 0]
                if len(constant_cols) > 0:
                    print(f"Removing {len(constant_cols)} constant columns")
                    x_train_tau = x_train_tau.drop(columns=constant_cols)
                    x_test_tau = x_test_tau.drop(columns=constant_cols)

                y_train_tau = Surv.from_dataframe('event', 'time', train_df_tau)

                cox_model_tau = CoxPHSurvivalAnalysis()
                try:
                    cox_model_tau.fit(x_train_tau, y_train_tau)

                    # Risk scores for test set
                    test_risk_scores_tau = cox_model_tau.predict(x_test_tau)

                    # Survival functions for test set (argument is residual time since tau)
                    surv_funcs_tau = cox_model_tau.predict_survival_function(x_test_tau)

                    # Store predictions
                    for test_idx, pid in enumerate(test_df_tau['pid'].values):
                        if pid not in patient_predictions:
                            patient_predictions[pid] = {}

                        risk = float(test_risk_scores_tau[test_idx])
                        surv_func = surv_funcs_tau[test_idx]

                        # Survival probabilities at all future landmark times
                        # NOTE: you can calculate survival probabilities at ANY future time t > tau
                        # by calling: surv_func(t - tau)
                        surv_probs = {}
                        for future_idx in range(idx + 1, len(landmark_times)):
                            future_tau = landmark_times[future_idx]
                            surv_probs[f'surv_tau{future_idx}'] = float(surv_func(future_tau - tau))

                        patient_predictions[pid][f'tau{idx}'] = {
                            'risk': risk,
                            **surv_probs
                        }

                except Exception as e:
                    print(f"✗ Error fitting Cox model at τ{idx}: {e}")
            else:
                print(f"✗ Skipping τ{idx}: insufficient data")
                continue

        # ========== Save all predictions to CSV ==========
        print(f"\n{'='*50}")
        print("Saving predictions to CSV...")
        print(f"{'='*50}")

        with open(risk_csv_path, 'a', newline='') as csvfile:
            fieldnames = ['pid', 'tau', 't', 'risk', 'surv_prob', 'dataset']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for pid in sorted(patient_predictions.keys()):
                pred = patient_predictions[pid]

                # Write all risk scores (tau0..tau3 if available)
                for idx, tau in enumerate(landmark_times):
                    key = f'tau{idx}'
                    if key in pred:
                        writer.writerow({
                            'pid': pid,
                            'tau': tau,
                            't': np.nan,
                            'risk': pred[key]['risk'] if np.isfinite(pred[key]['risk']) else np.nan,
                            'surv_prob': np.nan,
                            'dataset': dataset_name,
                        })

                # Write survival probabilities at future landmark times
                for idx, tau in enumerate(landmark_times):
                    key = f'tau{idx}'
                    if key in pred:
                        for future_idx in range(idx + 1, len(landmark_times)):
                            future_tau = landmark_times[future_idx]
                            surv_key = f'surv_tau{future_idx}'
                            if surv_key in pred[key]:
                                writer.writerow({
                                    'pid': pid,
                                    'tau': tau,
                                    't': future_tau,
                                    'risk': np.nan,
                                    'surv_prob': pred[key][surv_key] if np.isfinite(pred[key][surv_key]) else np.nan,
                                    'dataset': dataset_name,
                                })

        print(f"✓ Risk predictions saved to {risk_csv_path}")

    except Exception as e:
        print(f"✗ Error processing dataset {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*50)
print("All datasets processed!")
print("="*50)
