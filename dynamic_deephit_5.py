import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
import warnings
import random
from ddh.ddh_api import DynamicDeepHit
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
import torch
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Directory containing the datasets
data_dir = os.path.join("data")

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# CSV paths
risk_csv_path = "dynamic_deephit_risk_5.csv"

def convert_to_sequential_format(df, covariates):
    """
    Convert longitudinal data to sequential format for DDH.
    """
    x_data = []
    t_data = []
    e_data = []
    pid_data = []
    
    # Sort PIDs to ensure consistent order
    for pid in sorted(df['pid'].unique()):  
        patient_data = df[df['pid'] == pid].sort_values('time2')
        
        # Extract features for this patient (all time points)
        patient_features = []
        patient_times = []
        patient_events = []
        
        for _, row in patient_data.iterrows():
            features = []
            for cov in covariates:
                val = row[cov]
                if pd.isna(val):
                    val = 0.0  # Replace NaN with 0
                features.append(float(val))
            
            patient_features.append(features)
            patient_times.append(float(row['time2']))
            patient_events.append(float(row['event']))
        
        if len(patient_features) == 0:
            continue  # Skip patients with no valid data
            
        # Convert to numpy arrays with correct dtypes
        x_data.append(np.array(patient_features, dtype=np.float32))
        t_data.append(np.array(patient_times, dtype=np.float32))
        e_data.append(np.array(patient_events, dtype=np.float32))
        pid_data.append(pid)  # Store the patient ID

    return (np.array(x_data, dtype=object), np.array(t_data, dtype=object), 
            np.array(e_data, dtype=object), np.array(pid_data))

# Initialize risk CSV file
risk_file_exists = os.path.isfile(risk_csv_path)
if not risk_file_exists:
    with open(risk_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['pid', 'tau', 't', 'risk', 'surv_prob', 'dataset']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

for file_name in dataset_files:
    dataset_name = file_name.replace(".csv", "")
    file_path = os.path.join(data_dir, file_name)
    print("="*50)
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
    
    # Handle categorical variables
    categorical_cols = []
    numerical_cols = []
    
    for col in covariates:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    # Get event times for landmark calculation
    event_times = df[df[event_col] == 1]["time2"].values
    if len(event_times) == 0:
        print(f"Skipping {dataset_name}: No events observed")
        continue
    
    # Calculate 5 landmark times (10, 30, 50, 70, 90 quantiles)
    τ1, τ2, τ3, τ4, τ5 = np.quantile(event_times, [0.1, 0.3, 0.5, 0.7, 0.9])
    landmark_times = [τ1, τ2, τ3, τ4, τ5]
    print(f"Landmark times: τ1={τ1:.2f}, τ2={τ2:.2f}, τ3={τ3:.2f}, τ4={τ4:.2f}, τ5={τ5:.2f}")

    # Split by patient ID (70-30 train-test split)
    original_pids = df["pid"].unique()
    pid_train, pid_test = train_test_split(original_pids, test_size=0.3, random_state=SEED)
    
    # Create train/test dataframes
    df_train = df[df['pid'].isin(pid_train)]
    df_test = df[df['pid'].isin(pid_test)]
    
    print(f"Train patients: {len(pid_train)}, Test patients: {len(pid_test)}")

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

    # Store back into df_train/df_test while preserving time2/event/pid
    df_train = pd.concat([df_train[['pid','time','time2','event']].reset_index(drop=True),
                                    x_train.reset_index(drop=True)],
                                    axis=1)
    
    df_test = pd.concat([df_test[['pid','time','time2','event']].reset_index(drop=True),
                                x_test.reset_index(drop=True)],
                                axis=1)

    # Update covariates after encoding
    covariates = df_train.columns.difference(['pid','time','time2','event']).tolist()

    # Convert to sequential format for each split
    x_train, t_train, e_train, train_pids = convert_to_sequential_format(df_train, covariates)
    x_test, t_test, e_test, test_pids = convert_to_sequential_format(df_test, covariates)
    
    print(f"Train shape: {len(x_train)} patients, Test shape: {len(x_test)} patients")

    # Calculate default values from TRAIN set only for imputation
    default_values = {}
    for col in covariates:
        default_values[col] = df_train[col].mean()
    print(f"Default values calculated from train set")

    # Add default rows to TRAIN set
    first_eval_time = landmark_times[0]
    default_rows_train = []
    for pid in train_pids:
        patient_data = df_train[df_train['pid'] == pid]
        min_visit_time = patient_data['time2'].min()
        
        if min_visit_time > first_eval_time:
            default_row = default_values.copy()
            default_row["pid"] = pid
            default_row["time"] = 0.0
            default_row["time2"] = first_eval_time
            default_row["event"] = 0.0
            default_rows_train.append(default_row)
    
    if len(default_rows_train) > 0:
        default_df = pd.DataFrame(default_rows_train)
        df_train = pd.concat([df_train, default_df], ignore_index=True)
        df_train = df_train.sort_values(['pid', 'time2']).reset_index(drop=True)
        print(f"Added {len(default_rows_train)} default rows to train set")
        
        x_train, t_train, e_train, train_pids = convert_to_sequential_format(df_train, covariates)
        print(f"Updated train shape: {len(x_train)} patients")
    
    # Add default rows to TEST set
    default_rows_test = []
    for pid in test_pids:
        patient_data = df_test[df_test['pid'] == pid]
        min_visit_time = patient_data['time2'].min()
        
        if min_visit_time > first_eval_time:
            default_row = default_values.copy()
            default_row["pid"] = pid
            default_row["time"] = 0.0
            default_row["time2"] = first_eval_time
            default_row["event"] = 0.0
            default_rows_test.append(default_row)
    
    if len(default_rows_test) > 0:
        default_df = pd.DataFrame(default_rows_test)
        df_test = pd.concat([df_test, default_df], ignore_index=True)
        df_test = df_test.sort_values(['pid', 'time2']).reset_index(drop=True)
        print(f"Added {len(default_rows_test)} default rows to test set")
        
        x_test, t_test, e_test, test_pids = convert_to_sequential_format(df_test, covariates)
        print(f"Updated test shape: {len(x_test)} patients")

    layers = [[100], [100, 100], [100, 100, 100]]

    # Parameter grid
    param_grid = {
            'layers_rnn': [2, 3],
            'hidden_long': layers,
            'hidden_rnn': [50, 100],
            'hidden_att': layers,
            'hidden_cs': layers,
            'sigma': [0.1, 1, 3],
            'learning_rate' : [1e-3],
            }
            
    params = ParameterGrid(param_grid)
    
    # Train first successful model
    model = None
    for param_idx, param in enumerate(params):
        try:
            print(f"\n[{param_idx+1}/{len(list(params))}] Training with params: {param}")
            
            model = DynamicDeepHit(
                        layers_rnn = param['layers_rnn'],
                        hidden_rnn = param['hidden_rnn'], 
                        long_param = {'layers': param['hidden_long'], 'dropout': 0.3}, 
                        att_param = {'layers': param['hidden_att'], 'dropout': 0.3}, 
                        cs_param = {'layers': param['hidden_cs'], 'dropout': 0.3},
                        sigma = param['sigma'],
                        split = [0] + landmark_times + [np.max([t_.max() for t_ in t_train])])
            
            # Train the model
            model.fit(x_train, t_train, e_train, iters = 10, 
                    learning_rate = param['learning_rate'])
            
            print(f"Model trained successfully!")
            break  # Use first successful model
            
        except Exception as e:
            print(f"Error training model: {e}")
            model = None
            continue

    if model is None:
        print(f"Skipping {dataset_name}: No models were successfully trained")
        continue

    # ========== Dynamic Landmark Evaluation (aligned with landmark_cox.py) ==========
    print(f"\n{'='*50}")
    print("Performing dynamic landmark evaluation...")
    print(f"{'='*50}")

    # Dictionary to store predictions for all test patients
    patient_predictions = {pid: {} for pid in test_pids}

    # For each landmark time
    for idx, tau in enumerate(landmark_times, 1):
        print(f"\n{'='*50}")
        print(f"Processing landmark time τ{idx} = {tau:.2f}")
        print(f"{'='*50}")
        
        # Predict risk at all future landmark times using information only up to tau
        future_times = [landmark_times[i] for i in range(idx-1, len(landmark_times))]
        
        # Get risk predictions - shape: (n_test, n_future_times, max_seq_len)
        out_risk = model.predict_risk(x_test, future_times, all_step=True)
        
        # For each test patient, extract the risk score at tau
        for patient_idx, pid in enumerate(test_pids):
            patient_visit_times = t_test[patient_idx]
            
            # Find the most recent visit at or before tau
            valid_visits_mask = patient_visit_times <= tau
            
            if not np.any(valid_visits_mask):
                # No visits before tau - skip this patient for this landmark
                continue
            
            valid_visit_indices = np.where(valid_visits_mask)[0]
            most_recent_visit_idx = valid_visit_indices[-1]
            
            # Extract risk score at tau (first future time, which is tau itself or next landmark)
            # Risk score at tau for predicting event by tau
            risk_at_tau = out_risk[patient_idx, 0, most_recent_visit_idx]
            
            # Extract survival probabilities for future landmarks
            surv_probs = {}
            for future_idx in range(idx, len(landmark_times)):
                # Index in out_risk for this future time
                risk_idx = future_idx - idx + 1
                if risk_idx < out_risk.shape[1]:
                    risk_future = out_risk[patient_idx, risk_idx, most_recent_visit_idx]
                    # Survival probability = 1 - risk
                    surv_probs[f'surv_tau{future_idx+1}'] = 1.0 - risk_future
            
            # Store predictions
            patient_predictions[pid][f'tau{idx}'] = {
                'risk': risk_at_tau,
                **surv_probs
            }
            
        print(f"✓ Predictions extracted for {len([p for p in patient_predictions.values() if f'tau{idx}' in p])} patients at τ{idx}")

    # ========== Save all predictions to CSV (same format as landmark_cox.py) ==========
    print(f"\n{'='*50}")
    print("Saving predictions to CSV...")
    print(f"{'='*50}")

    with open(risk_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['pid', 'tau', 't', 'risk', 'surv_prob', 'dataset']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for pid in sorted(test_pids):
            pred = patient_predictions[pid]
            
            # First, write all risk scores (τ1, τ2, τ3, τ4, τ5)
            for idx, tau in enumerate(landmark_times, 1):
                if f'tau{idx}' in pred:
                    writer.writerow({
                        'pid': pid,
                        'tau': tau,
                        't': np.nan,
                        'risk': pred[f'tau{idx}']['risk'] if np.isfinite(pred[f'tau{idx}']['risk']) else np.nan,
                        'surv_prob': np.nan,
                        'dataset': dataset_name,
                    })
            
            # Then, write all survival probabilities
            for idx, tau in enumerate(landmark_times, 1):
                if f'tau{idx}' in pred:
                    # Write survival probabilities for all future timepoints
                    for future_idx in range(idx, len(landmark_times)):
                        future_tau = landmark_times[future_idx]
                        surv_key = f'surv_tau{future_idx+1}'
                        if surv_key in pred[f'tau{idx}']:
                            writer.writerow({
                                'pid': pid,
                                'tau': tau,
                                't': future_tau,
                                'risk': np.nan,
                                'surv_prob': pred[f'tau{idx}'][surv_key] if np.isfinite(pred[f'tau{idx}'][surv_key]) else np.nan,
                                'dataset': dataset_name,
                            })

    print(f"✓ Risk predictions saved to {risk_csv_path}")

print("\n" + "="*50)
print("Processing complete!")
print("="*50)