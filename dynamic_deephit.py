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

TAU = 10

def build_patient_outcome(df):
    """
    For each pid, get final follow-up time (t_event) and status_event (1 if event, else 0).
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
        x_data.append(np.array(patient_features, dtype=np.float64))
        t_data.append(np.array(patient_times, dtype=np.float64))
        e_data.append(np.array(patient_events, dtype=np.float64))
    
    return np.array(x_data, dtype=object), np.array(t_data, dtype=object), np.array(e_data, dtype=object)

def evaluate_at_landmark_time(model, x_test, t_test, e_test, landmark_time, patient_outcome, tau):
    """
    Evaluate DDH model at a specific landmark time for patients who survive past that time.
    This mimics the landmark analysis evaluation without using landmark training data.
    """
    try:
        # Filter patients who survive past landmark_time
        valid_patients = []
        final_times = []
        final_events = []
        valid_x = []
        valid_t = []
        valid_e = []
        
        for i, (x_seq, t_seq, e_seq) in enumerate(zip(x_test, t_test, e_test)):
            # Get patient ID - we need to match with patient_outcome
            # For now, use the final time as a proxy to find matching patient
            final_time = t_seq[-1]
            final_event = bool(e_seq[-1])
            
            # Check if this patient survives past landmark_time
            if final_time > landmark_time:
                # Calculate remaining time and event status after landmark
                time_after_landmark = final_time - landmark_time
                remaining_time = min(time_after_landmark, tau)
                
                if final_event and (time_after_landmark <= tau):
                    landmark_event = 1
                else:
                    landmark_event = 0
                
                valid_patients.append(i)
                final_times.append(remaining_time)
                final_events.append(bool(landmark_event))
                valid_x.append(x_seq)
                valid_t.append(t_seq)
                valid_e.append(e_seq)
            
        final_times = np.array(final_times)
        final_events = np.array(final_events)
        
        # Use tau as horizon for prediction (consistent with landmark analysis)
        horizon = tau
        
        # Predict risk using DDH on valid patients
        try:
            valid_x = np.array(valid_x, dtype=object)
            risk_scores = model.predict_risk(valid_x, [horizon])
            
            # Handle different output formats
            if isinstance(risk_scores, (list, tuple)):
                risk_scores = np.array(risk_scores)
            elif not isinstance(risk_scores, np.ndarray):
                risk_scores = np.array([risk_scores])
                
            # Ensure we have a 1D array of risk scores
            if risk_scores.ndim > 1:
                if risk_scores.shape[1] == 1:
                    risk_scores = risk_scores.flatten()
                else:
                    # If multiple time points, use the first one or average
                    risk_scores = np.mean(risk_scores, axis=1) if risk_scores.shape[1] > 1 else risk_scores[:, 0]
                
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(risk_scores)
            if not np.all(valid_mask):
                print(f"Found {np.sum(~valid_mask)} invalid risk scores")
                if np.sum(valid_mask) < 5:
                    return np.nan, len(final_times), sum(final_events)
                # Filter to valid scores only
                risk_scores = risk_scores[valid_mask]
                final_times = final_times[valid_mask]
                final_events = final_events[valid_mask]
                
        except Exception as pred_error:
            print(f"Risk prediction error: {str(pred_error)}")
            return np.nan, len(final_times), sum(final_events)
        
        # Calculate C-index
        try:
            cindex_result = concordance_index_censored(final_events, final_times, risk_scores)
            if isinstance(cindex_result, tuple):
                cindex = cindex_result[0]
            else:
                cindex = float(cindex_result)
            return cindex, len(final_times), sum(final_events)
        except Exception as cindex_error:
            print(f"C-index calculation error: {str(cindex_error)}")
            return np.nan, len(final_times), sum(final_events)
        
    except Exception as e:
        print(f"Evaluation error at landmark {landmark_time:.2f}: {str(e)}")
        return np.nan, 0, 0

for file_name in dataset_files:
    try:
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
        print(f"Number of covariates: {len(covariates)}")
        
        # Handle categorical variables with label encoding
        categorical_cols = []
        numerical_cols = []
        
        for col in covariates:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        
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
        
        print(f"Final number of covariates: {len(covariates)}")

        # Get event times for landmark calculation
        event_times = df[df[event_col] == 1]["time2"].values
        if len(event_times) == 0:
            print(f"Skipping {dataset_name}: No events observed")
            continue

        # Convert to sequential format for DDH - use full dynamic data
        x, t, e = convert_to_sequential_format(df, covariates)
        
        # Split data - same approach as landmark_cox.py
        all_pids = df['pid'].unique()
        pids_trainval, pids_test = train_test_split(
            all_pids, test_size=0.15, random_state=SEED
        )
        pids_train, pids_val = train_test_split(
            pids_trainval, test_size=0.1765, random_state=SEED
        )
        
        # Map patient IDs to indices for splitting x, t, e
        pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}
        train_indices = [pid_to_idx[pid] for pid in pids_train if pid in pid_to_idx]
        val_indices = [pid_to_idx[pid] for pid in pids_val if pid in pid_to_idx]
        test_indices = [pid_to_idx[pid] for pid in pids_test if pid in pid_to_idx]
        
        x_train = x[train_indices]
        x_val = x[val_indices]  
        x_test = x[test_indices]
        
        t_train = t[train_indices]
        t_val = t[val_indices]
        t_test = t[test_indices]
        
        e_train = e[train_indices]
        e_val = e[val_indices]
        e_test = e[test_indices]
        
        print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        
        # Check for sufficient events
        train_events = sum(1 for e_seq in e_train if e_seq[-1] == 1)
        if train_events < 10:
            print(f"Skipping {dataset_name}: Too few events in training ({train_events})")
            continue
        
        # Build patient outcome DataFrame for evaluation
        patient_outcome = build_patient_outcome(df)
        
        # Compute horizons for DDH training
        all_final_times = [t_[-1] for t_, e_ in zip(t, e) if e_[-1] == 1]
        if len(all_final_times) < 3:
            print(f"Skipping {dataset_name}: Too few final events")
            continue
            
        horizons = [0.25, 0.5, 0.75]
        times = np.quantile(all_final_times, horizons).tolist()
        
        # Train ONE model on full dynamic data
        try:
            model = DynamicDeepHit(
                layers_rnn=2,
                hidden_rnn=50, 
                long_param={'layers': [50], 'dropout': 0.3}, 
                att_param={'layers': [], 'dropout': 0.3}, 
                cs_param={'layers': [], 'dropout': 0.3},
                sigma=0.1,
                split=[0] + times + [np.max([t_.max() for t_ in t])]
            )
            
            # Train model on full dynamic data
            model.fit(x_train, t_train, e_train, 
                     iters=10, 
                     learning_rate=1e-3)
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            # Try simpler configuration
            try:
                print("Retrying with simpler configuration...")
                model = DynamicDeepHit(
                    layers_rnn=1,
                    hidden_rnn=25, 
                    long_param={'layers': [], 'dropout': 0.2}, 
                    att_param={'layers': [], 'dropout': 0.2}, 
                    cs_param={'layers': [], 'dropout': 0.2},
                    sigma=0.1,
                    split=10  # Simple fixed bins
                )
                
                model.fit(x_train, t_train, e_train, iters=3, learning_rate=1e-3)
                print("Retry successful!")
            except Exception as e2:
                print(f"Retry also failed: {str(e2)}")
                continue
        
        # Now evaluate the SAME model at different landmark times
        landmark_times = np.quantile(event_times, [0.25, 0.5, 0.75])
        print(f"Landmark evaluation times: {landmark_times}")
        
        results = []
        
        for i, landmark_time in enumerate(landmark_times):
            print(f"\nEvaluating at landmark time {i+1}: {landmark_time:.2f}")
            
            # Evaluate on validation set at this landmark time
            val_cindex, n_val_patients, n_val_events = evaluate_at_landmark_time(
                model, x_val, t_val, e_val, landmark_time, patient_outcome, TAU
            )
            print(f"Validation C-index: {val_cindex:.4f} (n_patients={n_val_patients}, n_events={n_val_events})")
            
            # Evaluate on test set at this landmark time
            test_cindex, n_test_patients, n_test_events = evaluate_at_landmark_time(
                model, x_test, t_test, e_test, landmark_time, patient_outcome, TAU
            )
            print(f"Test C-index: {test_cindex:.4f} (n_patients={n_test_patients}, n_events={n_test_events})")
            
            # Save result for this landmark time
            result = {
                'dataset': dataset_name,
                'landmark_time': landmark_time,
                'tau': TAU,
                'n_patients': n_test_patients,
                'n_events': n_test_events,
                'val_cindex': val_cindex,
                'test_cindex': test_cindex,
            }
                    
            results.append(result)

        # Save results for this dataset - matching landmark_cox.py format
        if results:
            # Find best model based on validation C-index
            best_result = max(results, key=lambda x: x['val_cindex'] if not np.isnan(x['val_cindex']) else -np.inf)
            print(f"\nBest landmark time: {best_result['landmark_time']:.2f}")
            print(f"Best validation C-index: {best_result['val_cindex']:.4f}")
            print(f"Corresponding test C-index: {best_result['test_cindex']:.4f}")

            # Save to CSV
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, 'a', newline='') as csvfile:
                fieldnames = ['dataset', 'landmark_time', 'tau', 'n_patients', 'n_events', 
                             'val_cindex', 'test_cindex']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                for result in results:
                    writer.writerow(result)
        else:
            print(f"No valid results for dataset {dataset_name}")
            
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        continue

print("\nDynamic Deep Hit evaluation completed!")
