import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw
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
csv_path = "landmark_cox_evaluation.csv"
risk_csv_path = "landmark_cox_risk.csv"

# Initialize risk CSV file
risk_file_exists = os.path.isfile(risk_csv_path)
if not risk_file_exists:
    with open(risk_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['pid', 'eval_time', 'risk', 'dataset']
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

def apply_locf_landmark_train(df, landmark_time, covariates, patient_outcome):
    """
    Landmarking with LOCF for TRAINING set.
    Keep only patients still at risk at landmark_time.
    """
    rows = []
    for pid, patient in df.groupby("pid"):
        patient = patient.sort_values("time2")
        t_event = patient_outcome.loc[pid, "t_event"]
        status_event = patient_outcome.loc[pid, "status_event"]

        # must be under follow-up at (strictly after) the landmark
        if t_event <= landmark_time:
            continue

        # last observation carried forward up to the landmark
        valid = patient[patient["time2"] <= landmark_time]
        if valid.empty:
            continue
        latest = valid.iloc[-1]

        new_row = {"pid": pid,
                   "time": float(t_event - landmark_time),
                   "event": int(status_event)}
        for c in covariates:
            new_row[c] = latest[c]
        rows.append(new_row)

    return pd.DataFrame(rows)

def apply_locf_landmark_test(df, landmark_time, covariates, patient_outcome, default_values):
    """
    Landmarking with LOCF for TEST set.
    Keep ALL test patients.
    - If patient has visits before landmark_time: use LOCF
    - If patient has no visits before landmark_time: use default (mean) values
    """
    rows = []
    for pid, patient in df.groupby("pid"):
        patient = patient.sort_values("time2")
        t_event = patient_outcome.loc[pid, "t_event"]
        status_event = patient_outcome.loc[pid, "status_event"]

        # For test set, we want to predict for all patients
        # Calculate time to event from landmark
        time_to_event = float(t_event - landmark_time)
        
        # Check if patient has visits before landmark
        valid = patient[patient["time2"] <= landmark_time]
        
        if not valid.empty:
            # Use LOCF: last observation before landmark
            latest = valid.iloc[-1]
            new_row = {"pid": pid,
                       "time": time_to_event,
                       "event": int(status_event) if t_event > landmark_time else 0}
            for c in covariates:
                new_row[c] = latest[c]
        else:
            # No visits before landmark: use default values
            new_row = {"pid": pid,
                       "time": time_to_event,
                       "event": int(status_event) if t_event > landmark_time else 0}
            for c in covariates:
                new_row[c] = default_values[c]
        
        rows.append(new_row)

    return pd.DataFrame(rows)

for file_name in dataset_files:
    try:
        dataset_name = file_name.replace(".csv", "")
        file_path = os.path.join(data_dir, file_name)
        print("="*50)
        print(f"\nProcessing dataset: {dataset_name}")

        # Load datasets
        df = pd.read_csv(file_path)
        shape = df.shape
        print(f"Dataset shape: {shape}")

        # Define columns
        time_col = "time"
        time2_col = "time2"
        event_col = "event"

        covariates = df.columns.difference([time_col, time2_col, event_col, 'pid']).tolist()

        # Get event times for landmark calculation
        event_times = df[df[event_col] == 1]["time2"].values
        if len(event_times) == 0:
            print(f"Skipping {dataset_name}: No events observed")
            continue

        # Pick three landmark times (25, 50, 75 quantiles of event times)
        times = np.quantile(event_times, [0.25, 0.5, 0.75])
        print(f"Landmark times: {times}")

        # Split patient IDs into train (70%), test (30%)
        all_pids = df['pid'].unique()
        pids_train, pids_test = train_test_split(
            all_pids, test_size=0.3, random_state=SEED
        )
        
        # Split dataframe by patient IDs
        df_train = df[df['pid'].isin(pids_train)]
        df_test = df[df['pid'].isin(pids_test)]
        
        print(f"Train patients: {len(pids_train)}, Test patients: {len(pids_test)}")
        
        # Build patient outcome DataFrame (from full data)
        patient_outcome = build_patient_outcome(df)
        
        # Identify categorical and numerical columns
        categorical_cols = []
        numerical_cols = []
        for col in covariates:
            if df_train[col].dtype == 'object' or df_train[col].dtype.name == 'category':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        # Calculate default values from TRAIN set for test imputation
        default_values = {}
        for col in numerical_cols:
            default_values[col] = df_train[col].mean()
        for col in categorical_cols:
            mode_val = df_train[col].mode()
            default_values[col] = mode_val.iloc[0] if not mode_val.empty else df_train[col].iloc[0]
        
        results = []
        
        # Dictionary to store risk scores for all patients at all timepoints
        patient_risks_dict = {pid: [] for pid in pids_test}
        
        for i, landmark_time in enumerate(times):
            print(f"\nProcessing landmark time {i+1}: {landmark_time:.2f}")
            
            # Apply LOCF for train (only patients at risk)
            train_df = apply_locf_landmark_train(
                df_train, landmark_time, covariates, patient_outcome
            )
            
            # Apply LOCF for test (ALL patients, with default imputation)
            test_df = apply_locf_landmark_test(
                df_test, landmark_time, covariates, patient_outcome, default_values
            )

            print(f"Train: {len(train_df)} patients, Events: {train_df['event'].sum()}")
            print(f"Test: {len(test_df)} patients, Events: {test_df['event'].sum()}")

            if len(train_df) < 10 or train_df['event'].sum() < 5:
                print(f"Skipping landmark {landmark_time:.2f}: insufficient train data")
                continue

            if len(test_df) < 5:
                print(f"Skipping landmark {landmark_time:.2f}: insufficient test data")
                continue
            
            # Extract covariates
            x_train = train_df[covariates].copy()
            x_test = test_df[covariates].copy()
            
            # One-hot encode categorical variables
            x_train_ohe = pd.get_dummies(x_train, drop_first=True)
            x_test_ohe = pd.get_dummies(x_test, drop_first=True)
            
            # Align columns between train and test
            all_columns = x_train_ohe.columns.union(x_test_ohe.columns)
            for col in all_columns:
                if col not in x_train_ohe.columns:
                    x_train_ohe[col] = 0
                if col not in x_test_ohe.columns:
                    x_test_ohe[col] = 0
            
            x_train_ohe = x_train_ohe.reindex(columns=sorted(all_columns))
            x_test_ohe = x_test_ohe.reindex(columns=sorted(all_columns))
            
            # Impute missing values
            imputer = SimpleImputer(strategy='median')
            x_train_imputed = pd.DataFrame(
                imputer.fit_transform(x_train_ohe), 
                columns=x_train_ohe.columns,
                index=x_train_ohe.index
            )
            x_test_imputed = pd.DataFrame(
                imputer.transform(x_test_ohe),
                columns=x_test_ohe.columns,
                index=x_test_ohe.index
            )
            
            # Remove constant columns to avoid singularity
            constant_cols = x_train_imputed.columns[x_train_imputed.std() == 0]
            if len(constant_cols) > 0:
                print(f"Removing {len(constant_cols)} constant columns")
                x_train_imputed = x_train_imputed.drop(columns=constant_cols)
                x_test_imputed = x_test_imputed.drop(columns=constant_cols)
                
            # Prepare survival data
            y_train = Surv.from_dataframe('event', 'time', train_df)
            y_test = Surv.from_dataframe('event', 'time', test_df)
                
            # Fit Cox model
            cox_model = CoxPHSurvivalAnalysis()
            cox_model.fit(x_train_imputed, y_train)

            # Evaluate on test set
            test_risk_scores = cox_model.predict(x_test_imputed)
            test_cindex, *_ = concordance_index_ipcw(y_train, y_test, test_risk_scores, landmark_time)

            # Store risk scores for each test patient at this landmark time
            for idx, pid in enumerate(test_df['pid'].values):
                risk_score = test_risk_scores[idx]
                patient_risks_dict[pid].append((landmark_time, risk_score))

            # Save result
            result = {
                'dataset': dataset_name,
                'landmark_time': landmark_time,
                'n_train': len(train_df),
                'n_test': len(test_df),
                'n_events_train': train_df['event'].sum(),
                'test_cindex': test_cindex,
            }
                    
            results.append(result)
                    
            print(f"Test C-index: {test_cindex:.4f}")

        # Save all risk predictions to CSV (grouped by patient)
        with open(risk_csv_path, 'a', newline='') as csvfile:
            fieldnames = ['pid', 'eval_time', 'risk', 'dataset']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write all evaluation times for each patient together
            for pid in sorted(pids_test):
                if pid in patient_risks_dict and len(patient_risks_dict[pid]) > 0:
                    for eval_time, risk in patient_risks_dict[pid]:
                        writer.writerow({
                            'pid': pid,
                            'eval_time': eval_time,
                            'risk': risk if np.isfinite(risk) else np.nan,
                            'dataset': dataset_name
                        })

        # Save results for this dataset
        if results:
            # Save to CSV
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, 'a', newline='') as csvfile:
                fieldnames = ['dataset', 'time', 'cindex']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                # Write each landmark time result
                for result in results:
                    uniform_result = {
                        'dataset': result['dataset'],
                        'time': result['landmark_time'],
                        'cindex': result['test_cindex']
                    }
                    writer.writerow(uniform_result)
        else:
            print(f"No valid results for dataset {dataset_name}")
            
        print(f"\nRisk predictions saved to {risk_csv_path}")
            
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue