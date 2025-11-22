import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
import warnings
import random
from ddh.ddh_api import DynamicDeepHit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from sksurv.metrics import concordance_index_ipcw
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
csv_path = "dynamic_deephit_evaluation_5.csv"
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
        fieldnames = ['pid', 'eval_time', 'risk', 'dataset']
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
    
    # Handle categorical variables with label encoding
    categorical_cols = []
    numerical_cols = []
    
    for col in covariates:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    # Use label encoding instead of one-hot encoding
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna('Unknown')
        df[col] = le.fit_transform(df[col])
    
    # Handle missing values in numerical columns
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, NaN for invalid
        df[col] = df[col].fillna(df[col].median())
        
    # Ensure all covariates are numeric
    for col in covariates:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0.0)  # Fill any remaining NaN with 0

    # Get event times for evaluation time calculation
    event_times = df[df[event_col] == 1]["time2"].values
    if len(event_times) == 0:
        print(f"Skipping {dataset_name}: No events observed")
        continue
    
    # Calculate evaluation times from ALL data (before splitting)
    horizons = [0.1, 0.3, 0.5, 0.7, 0.9]
    times = np.quantile(event_times, horizons).tolist()
    print(f"Evaluation times: {times}")

    # Split by patient ID (70-30 train-test split)
    original_pids = df["pid"].unique()
    pid_train, pid_test = train_test_split(original_pids, test_size=0.3, random_state=SEED)
    
    # Create train/test dataframes
    df_train = df[df['pid'].isin(pid_train)]
    df_test = df[df['pid'].isin(pid_test)]
    
    print(f"Train patients: {len(pid_train)}, Test patients: {len(pid_test)}")

    # Convert to sequential format for each split
    x_train, t_train, e_train, train_pids = convert_to_sequential_format(df_train, covariates)
    x_test, t_test, e_test, test_pids = convert_to_sequential_format(df_test, covariates)
    
    print(f"Train shape: {len(x_train)} patients, Test shape: {len(x_test)} patients")

    # Calculate default values from TRAIN set for imputation
    default_values = {}
    for col in numerical_cols:
        default_values[col] = df_train[col].mean()
    for col in categorical_cols:
        default_values[col] = df_train[col].mode().iloc[0] if not df_train[col].mode().empty else df_train[col].iloc[0]
    
    # Add default rows to TEST set
    first_eval_time = times[0]
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

    layers = [[], [100], [100, 100], [100, 100, 100]]

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
                        split = [0] + times + [np.max([t_.max() for t_ in t_train])])
            
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

    # Use the model for test predictions
    out_risk = model.predict_risk(x_test, times, all_step = True)
    print(f"Risk prediction shape: {out_risk.shape}")

    cis = []

    et_train = np.array([(e_train[i][j], t_train[i][j]) for i in range(len(e_train)) for j in range(len(e_train[i]))],
                        dtype = [('e', bool), ('t', float)])

    et_test = np.array([(e_test[i][j], t_test[i][j]) for i in range(len(e_test)) for j in range(len(e_test[i]))],
                        dtype = [('e', bool), ('t', float)])

    for i, eval_time in enumerate(times):
        risk_scores = out_risk[:, i]  # Shape: (n_test, max_seq_len)
        
        # Extract risk score from most recent visit BEFORE eval_time
        final_risk_scores = []
        for patient_idx, patient_risks in enumerate(risk_scores):
            # Get this patient's visit times
            patient_visit_times = t_test[patient_idx]  # Array of visit times
            
            # Find visits that occurred before or at eval_time
            valid_visits_mask = patient_visit_times <= eval_time
            
            # Get the index of the most recent valid visit
            valid_visit_indices = np.where(valid_visits_mask)[0]
            most_recent_visit_idx = valid_visit_indices[-1]
            
            # Get the risk score from that visit
            risk_at_visit = patient_risks[most_recent_visit_idx]
            
            final_risk_scores.append(risk_at_visit)
        
        final_risk_scores = np.array(final_risk_scores)
        
        # Store risk scores for this evaluation time
        if i == 0:
            # Initialize dictionary to store all risks for each patient
            patient_risks_dict = {pid: [] for pid in test_pids}
        
        # Append risks for this evaluation time
        for pid, risk in zip(test_pids, final_risk_scores):
            patient_risks_dict[pid].append((eval_time, risk))
    
        et_test_final = np.array([(e_test[i][-1], t_test[i][-1]) for i in range(len(e_test))],
                                dtype = [('e', bool), ('t', float)])

        c_index, *_ = concordance_index_ipcw(et_train, et_test_final, 
                                            final_risk_scores, eval_time)
                                            
        cis.append(c_index)
        print(f"C-index at time {eval_time:.2f}: {c_index:.4f}")

    # Save all risk predictions to CSV (grouped by patient)
    with open(risk_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['pid', 'eval_time', 'risk', 'dataset']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write all evaluation times for each patient together
        for pid in test_pids:
            for eval_time, risk in patient_risks_dict[pid]:
                writer.writerow({
                    'pid': pid,
                    'eval_time': eval_time,
                    'risk': risk if np.isfinite(risk) else np.nan,
                    'dataset': dataset_name
                })

    # Save to CSV with uniform format (same as other two models)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['dataset', 'time', 'cindex']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Write each evaluation time result
        for i, (eval_time, c_index) in enumerate(zip(times, cis)):
            uniform_result = {
                'dataset': dataset_name,
                'time': eval_time,
                'cindex': c_index
            }
            writer.writerow(uniform_result)
    
    print(f"\nResults saved to {csv_path}")
    print(f"Risk predictions saved to {risk_csv_path}")