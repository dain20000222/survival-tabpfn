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
csv_path = "dynamic_deephit_evaluation.csv"

def convert_to_sequential_format(df, covariates):
    """
    Convert longitudinal data to sequential format for DDH.
    Use full dynamic data, not landmark approach.
    """
    x_data = []
    t_data = []
    e_data = []
    
    for pid in df['pid'].unique():
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

    return np.array(x_data, dtype=object), np.array(t_data, dtype=object), np.array(e_data, dtype=object)

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

    # Get event times for landmark calculation
    event_times = df[df[event_col] == 1]["time2"].values
    if len(event_times) == 0:
        print(f"Skipping {dataset_name}: No events observed")
        continue

    # Convert to sequential format for DDH - use full dynamic data
    x, t, e = convert_to_sequential_format(df, covariates)

    n = len(x)
    
    tr_size = int(n*0.70)
    te_size = n - tr_size

    x_train = np.array(x[:tr_size], dtype=object)
    x_test = np.array(x[tr_size:], dtype=object)
    t_train = np.array(t[:tr_size], dtype=object) 
    t_test = np.array(t[tr_size:], dtype=object)
    e_train = np.array(e[:tr_size], dtype=object)
    e_test = np.array(e[tr_size:], dtype=object)
        
    horizons = [0.25, 0.5, 0.75]
    times = np.quantile([t_[-1] for t_, e_ in zip(t, e) if e_[-1] == 1], horizons).tolist()
    print(f"Evaluation times: {times}")

    layers = [[], [100], [100, 100], [100, 100, 100]]

    # Simplified parameter grid (no validation tuning)
    param_grid = {
            'layers_rnn': [2],
            'hidden_long': [[100]],
            'hidden_rnn': [100],
            'hidden_att': [[100]],
            'hidden_cs': [[100]],
            'sigma': [1],
            'learning_rate' : [1e-3],
            }
    params = ParameterGrid(param_grid)
    
    # Train single model (no validation-based selection)
    model = None
    for param in params:
        try: 
            model = DynamicDeepHit(
                        layers_rnn = param['layers_rnn'],
                        hidden_rnn = param['hidden_rnn'], 
                        long_param = {'layers': param['hidden_long'], 'dropout': 0.3}, 
                        att_param = {'layers': param['hidden_att'], 'dropout': 0.3}, 
                        cs_param = {'layers': param['hidden_cs'], 'dropout': 0.3},
                        sigma = param['sigma'],
                        split = [0] + times + [np.max([t_.max() for t_ in t])])
            # The fit method is called to train the model
            model.fit(x_train, t_train, e_train, iters = 10, 
                    learning_rate = param['learning_rate'])
            break  # Use first successful model
        except Exception as e:
            print(f"Error training model with params {param}: {e}")
            model = None
            continue

    if model is None:
        print(f"Skipping {dataset_name}: No models were successfully trained")
        continue

    out_risk = model.predict_risk(x_test, times, all_step = True)

    cis = []

    et_train = np.array([(e_train[i][j], t_train[i][j]) for i in range(len(e_train)) for j in range(len(e_train[i]))],
                        dtype = [('e', bool), ('t', float)])

    et_test = np.array([(e_test[i][j], t_test[i][j]) for i in range(len(e_test)) for j in range(len(e_test[i]))],
                        dtype = [('e', bool), ('t', float)])


    for i, eval_time in enumerate(times):
        risk_scores = out_risk[:, i]
        
        # Extract the final risk score for each patient (last non-NaN value)
        final_risk_scores = []
        for patient_risks in risk_scores:
            # Get the last non-NaN risk score for this patient
            valid_risks = patient_risks[~np.isnan(patient_risks)]
            if len(valid_risks) > 0:
                final_risk_scores.append(valid_risks[-1])  # Take the last valid risk
            else:
                final_risk_scores.append(np.nan)
        
        final_risk_scores = np.array(final_risk_scores)
        
        # Handle NaN values
        valid_mask = np.isfinite(final_risk_scores)
        
        if np.sum(valid_mask) > 5:  # Need at least some valid predictions
            et_test_final = np.array([(e_test[i][-1], t_test[i][-1]) for i in range(len(e_test))],
                                    dtype = [('e', bool), ('t', float)])

            c_index, *_ = concordance_index_ipcw(et_train, et_test_final[valid_mask], 
                                                final_risk_scores[valid_mask], eval_time)
                                                
            cis.append(c_index)
            print(f"C-index at time {eval_time:.2f}: {c_index:.4f}")

        else:
            print(f"Warning: Too many NaN predictions for time {eval_time:.2f}")
            cis.append(np.nan)

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
