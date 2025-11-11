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

# CSV path
csv_path = "landmark_cox_evaluation_5.csv"

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

def apply_locf_landmark(df, landmark_time, covariates, patient_outcome):
    """
    Landmarking with LOCF (no tau truncation).
    Keep patients still at risk at landmark_time.
    time = t_event - landmark_time
    event = status_event (1 if event happens after the landmark, else 0)
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

        covariates = df.columns.difference([time_col, time2_col, event_col, 'pid'])

        # Get event times for landmark calculation
        event_times = df[df[event_col] == 1]["time2"].values
        if len(event_times) == 0:
            print(f"Skipping {dataset_name}: No events observed")
            continue

        # Pick three landmark times (25, 50, 75 quantiles of event times)
        times = np.quantile(event_times, [0.1, 0.3, 0.5, 0.7, 0.9]) 
        print(f"Landmark times: {times}")

        # Split patient IDs into train, test
        all_pids = df['pid'].unique()
        pids_train, pids_test = train_test_split(
            all_pids, test_size=0.3, random_state=SEED
        )
        # Build patient outcome DataFrame
        patient_outcome = build_patient_outcome(df)
        
        results = []
        
        for i, landmark_time in enumerate(times):
            print(f"\nProcessing landmark time {i+1}: {landmark_time:.2f}")
            
            # Apply LOCF for this landmark
            landmark_df = apply_locf_landmark(
                df, landmark_time, covariates, patient_outcome
            )

            print(landmark_df.head())
            print(f"Events: {landmark_df['event'].sum()}, Censored: {len(landmark_df) - landmark_df['event'].sum()}")

            # Split data based on patient IDs
            train_df = landmark_df[landmark_df["pid"].isin(pids_train)].copy()
            test_df = landmark_df[landmark_df["pid"].isin(pids_test)].copy()

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
            test_cindex,  *_ = concordance_index_ipcw(y_train, y_test, test_risk_scores, landmark_time)

            # Save result if we have valid metrics
            result = {
                'dataset': dataset_name,
                'landmark_time': landmark_time,
                'n_patients': len(landmark_df),
                'n_events': landmark_df['event'].sum(),
                'test_cindex': test_cindex,
            }
                    
            results.append(result)
                    
            print(f"Test C-index: {test_cindex:.4f}")

        # Save results for this dataset
        if results:
            # Find best model based on test C-index
            best_result = max(results, key=lambda x: x['test_cindex'] if not np.isnan(x['test_cindex']) else 0)
            print(f"\nBest landmark time: {best_result['landmark_time']:.2f}")
            print(f"Best test C-index: {best_result['test_cindex']:.4f}")

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
            
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        continue
