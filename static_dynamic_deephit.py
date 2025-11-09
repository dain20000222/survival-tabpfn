import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
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
data_dir = os.path.join("data_static")

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# CSV path
csv_path = "dynamic_deephit_static_evaluation.csv"

def convert_static_to_sequential_format(df, covariates, time_col="time", event_col="event"):
    """
    Convert a static (one-row-per-patient) table into a variable-length sequential format.
    Implementation follows the MITRA trainset construction used in autoregressive_mitra.py:
      - evaluation horizons = dataset percentiles (10..90) U patient's own time
      - if patient is censored, keep horizons only up to T_i
      - for each horizon produce a step with repeated static features, the horizon time and the binary label
        y_i(t) = 1{T_i <= t} * delta_i
    Returns:
      x_data: object array of shape (n_patients,) where each element is (seq_len, n_features)
      t_data: object array of shape (n_patients,) where each element is (seq_len,)
      e_data: object array of shape (n_patients,) where each element is (seq_len,)
    This mirrors construct_mitra_binary_trainset logic but returns per-patient sequences for DynamicDeepHit.
    """
    x_data = []
    t_data = []
    e_data = []

    # Build eval_times from dataset percentiles (same as autoregressive_mitra)
    eval_times = np.percentile(df[time_col].values, np.arange(10, 100, 10))
    eval_times = np.unique(eval_times)

    for _, row in df.iterrows():
        T_i = float(row[time_col])
        delta_i = int(row[event_col])

        # horizons = eval_times U {T_i}
        horizons = np.append(eval_times.copy(), T_i)
        horizons = np.unique(np.sort(horizons))

        # if censored, remove horizons after censor time (match MITRA behavior)
        if delta_i == 0:
            horizons = horizons[horizons <= T_i]

        # build repeated feature matrix (seq_len x n_features)
        feats = []
        for cov in covariates:
            v = row[cov]
            if pd.isna(v):
                v = 0.0
            feats.append(float(v))
        feats = np.array(feats, dtype=np.float32)
        x_seq = np.repeat(feats[np.newaxis, :], repeats=len(horizons), axis=0)

        # times sequence and binary event labels per horizon
        t_seq = np.array(horizons, dtype=np.float32)
        e_seq = np.array([int(bool(delta_i) and (T_i <= t)) for t in horizons], dtype=np.float32)

        x_data.append(x_seq)
        t_data.append(t_seq)
        e_data.append(e_seq)

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
    event_col = "event"

    covariates = df.columns.difference([time_col, event_col, 'pid']).tolist()
    
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

    # Convert STATIC dataset to sequential format (per-patient variable-length sequences)
    x, t, e = convert_static_to_sequential_format(df, covariates, time_col, event_col)

    # Train/val/test split (maintain stratification on event)
    n = len(x)
    indices = np.arange(n)
    strat = df[event_col].values
    train_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=strat, random_state=SEED)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1765, stratify=strat[train_idx], random_state=SEED)

    x_train = x[train_idx]; x_val = x[val_idx]; x_test = x[test_idx]
    t_train = t[train_idx]; t_val = t[val_idx]; t_test = t[test_idx]
    e_train = e[train_idx]; e_val = e[val_idx]; e_test = e[test_idx]

    # Model split: use full follow-up range (no landmarking). evaluate a single final time.
    max_time = float(np.max([t_[-1] for t_ in t]))
    times = [max_time]

    layers = [[], [100], [100, 100], [100, 100, 100]]

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
    
    models = []
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
            models.append([[model.compute_nll(x_val, t_val, e_val), model]])
        except Exception as e:
            print(f"Error training model with params {param}: {e}")
            continue

    if not models:
        print(f"Skipping {dataset_name}: No models were successfully trained")
        continue

    best_model = min(models)
    model = best_model[0][1]

    # Predict risk for the test set at the final follow-up time (no landmarking)
    max_time = float(np.max([t_seq[-1] for t_seq in t]))
    out_risk = model.predict_risk(x_test, [max_time], all_step=True)

    # out_risk shape: (n_test, n_times, n_steps_per_patient)
    risk_scores = out_risk[:, 0]  # first (and only) requested time

    # extract per-patient final risk value safely
    final_risk_scores = []
    for patient_risks in risk_scores:
        valid = patient_risks[~np.isnan(patient_risks)]
        final_risk_scores.append(valid[-1] if len(valid) > 0 else np.nan)
    final_risk_scores = np.array(final_risk_scores, dtype=float)

    # Prepare ground truth for concordance_index_censored (use last time/event in each patient's seq)
    y_test_times = np.array([t_test[i][-1] for i in range(len(t_test))], dtype=float)
    y_test_events = np.array([bool(e_test[i][-1]) for i in range(len(e_test))], dtype=bool)

    # Single concordance per dataset (no landmark times)
    c_index, *_ = concordance_index_censored(y_test_events, y_test_times, final_risk_scores)
    print(f"Dataset {dataset_name} -- test C-index: {c_index:.4f}")

    # Save single-row result (one c-index per dataset)
    result = {
        'dataset': dataset_name,
        'n_patients': int(len(x_test)),
        'n_events': int(y_test_events.sum()),
        'test_cindex': float(c_index),
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['dataset', 'n_patients', 'n_events', 'test_cindex']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    print(f"\nResults saved to {csv_path}")
