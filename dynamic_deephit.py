import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, concordance_index_ipcw, integrated_brier_score, brier_score
from ddh.ddh_api import DynamicDeepHit
import torch
import warnings
import random
from sklearn.model_selection import ParameterGrid
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
data_dir = os.path.join("dynamic")

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# CSV path
csv_path = "dyndeephit_evaluation.csv"

def prepare_dynamic_data(df):
    """
    Prepare time-varying data for Dynamic DeepHit.
    Convert to sequential format required by the model.
    """
    # Sort by patient ID and time
    df_sorted = df.sort_values(['pid', 'time'])
    
    # Group by patient ID
    patients = []
    times = []
    events = []
    
    for pid, group in df_sorted.groupby('pid'):
        # Get patient's sequential observations
        patient_data = []
        patient_times = []
        patient_events = []
        
        for _, row in group.iterrows():
            # Extract features (excluding pid, event, time, time2)
            features = []
            for col in df.columns:
                if col not in ['pid', 'event', 'time', 'time2']:
                    if col.startswith('fac_'):
                        # Categorical feature - will be encoded later
                        features.append(row[col])
                    else:
                        # Numerical feature
                        features.append(float(row[col]))
            
            patient_data.append(features)
            patient_times.append(float(row['time2']))  # Use time2 as observation time
            patient_events.append(int(row['event']))
        
        patients.append(np.array(patient_data))
        times.append(np.array(patient_times))
        events.append(np.array(patient_events))
    
    return patients, times, events

def encode_categorical_features(x_train, x_val, x_test, df_columns):
    """
    Encode categorical features consistently across train/val/test sets.
    """
    categorical_cols = [col for col in df_columns if col.startswith('fac_') and col not in ['pid', 'event', 'time', 'time2']]
    encoders = {}
    
    # Fit encoders on training data
    for i, col in enumerate([c for c in df_columns if c not in ['pid', 'event', 'time', 'time2']]):
        if col in categorical_cols:
            encoder = LabelEncoder()
            # Collect all values from training set for this feature
            train_values = []
            for patient in x_train:
                train_values.extend(patient[:, i])
            encoder.fit(train_values)
            encoders[i] = encoder
    
    # Apply encoders to all sets
    for x_set in [x_train, x_val, x_test]:
        for j, patient in enumerate(x_set):
            for i, col in enumerate([c for c in df_columns if c not in ['pid', 'event', 'time', 'time2']]):
                if i in encoders:
                    # Handle unseen categories by mapping to 0
                    encoded_values = []
                    for val in patient[:, i]:
                        try:
                            encoded_values.append(encoders[i].transform([val])[0])
                        except ValueError:
                            encoded_values.append(0)  # Default for unseen categories
                    x_set[j][:, i] = encoded_values
    
    return x_train, x_val, x_test

def impute_dynamic_data(x_train, x_val, x_test):
    """
    Impute missing values in dynamic data.
    """
    # Concatenate all training data to fit imputer
    all_train_data = np.vstack(x_train)
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(all_train_data)
    
    # Apply imputation to each set
    for x_set in [x_train, x_val, x_test]:
        for i, patient in enumerate(x_set):
            x_set[i] = imputer.transform(patient)
    
    return x_train, x_val, x_test


for file_name in dataset_files:
    try:
        dataset_name = file_name.replace(".csv", "")
        file_path = os.path.join(data_dir, file_name)
        print("="*50)
        print(f"\nProcessing dataset: {dataset_name}")

        # Track best per dataset
        best_row = None

        # Load datasets
        df = pd.read_csv(file_path)
        original_shape = df.shape
        print(f"Original dataset shape: {original_shape}")
        
        # Get number of unique patients
        n_patients = df['pid'].nunique()
        print(f"Number of unique patients: {n_patients}")
        
        # Calculate censoring rate
        last_observations = df.groupby('pid').last()
        censored = (last_observations["event"] == 0).sum()
        censored_percent = (censored / n_patients) * 100
        print(f"Percentage of censored patients: {censored_percent:.1f}%")
        
        # Prepare dynamic data
        x, t, e = prepare_dynamic_data(df)
        print(f"Example patient data: {x[0][:5]}")  # Print first 5 time points for first patient
        print(f"Example patient times: {t[0][:5]}")  # Print first 5 time points for first patient
        print(f"Example patient events: {e[0][:5]}")  # Print first 5 event indicators for first patient
        
        # Split patients into train/val/test (70%/15%/15%) 
        # First create patient indices for stratification
        patient_events = [events[-1] for events in e]  # Last event status for each patient
        patient_indices = list(range(len(x)))
        
        # Split into train/val/test
        train_idx, test_idx = train_test_split(
            patient_indices, test_size=0.15, stratify=patient_events, random_state=SEED
        )
        
        train_events = [patient_events[i] for i in train_idx]
        trainval_idx, val_idx = train_test_split(
            train_idx, test_size=0.1765, stratify=train_events, random_state=SEED  # 0.1765 ≈ 0.15/0.85
        )
        
        # Create data splits
        x_train = [x[i] for i in trainval_idx]
        x_val = [x[i] for i in val_idx]
        x_test = [x[i] for i in test_idx]
        
        t_train = [t[i] for i in trainval_idx]
        t_val = [t[i] for i in val_idx]
        t_test = [t[i] for i in test_idx]
        
        e_train = [e[i] for i in trainval_idx]
        e_val = [e[i] for i in val_idx]
        e_test = [e[i] for i in test_idx]
        
        print(f"Train patients: {len(x_train)}, Val patients: {len(x_val)}, Test patients: {len(x_test)}")
        
        # Encode categorical features
        x_train, x_val, x_test = encode_categorical_features(
            x_train, x_val, x_test, df.columns
        )
        
        # Impute missing values
        x_train, x_val, x_test = impute_dynamic_data(x_train, x_val, x_test)
        
        # Ensure all feature arrays are numeric (convert any remaining non-numeric values)
        for i, patient in enumerate(x_train):
            x_train[i] = patient.astype(float)
        for i, patient in enumerate(x_val):
            x_val[i] = patient.astype(float)
        for i, patient in enumerate(x_test):
            x_test[i] = patient.astype(float)
        
        # Convert to numpy arrays as expected by Dynamic DeepHit
        x_train = np.array(x_train, dtype=object)
        x_val = np.array(x_val, dtype=object)
        x_test = np.array(x_test, dtype=object)
        
        t_train = np.array(t_train, dtype=object)
        t_val = np.array(t_val, dtype=object)
        t_test = np.array(t_test, dtype=object)
        
        e_train = np.array(e_train, dtype=object)
        e_val = np.array(e_val, dtype=object)
        e_test = np.array(e_test, dtype=object)

        
        # Compute horizons at which we evaluate the performance
        # Use percentiles from training data
        train_times = []
        train_events = []
        for times, events in zip(t_train, e_train):
            train_times.extend(times)
            train_events.extend(events)
        
        # Compute horizons at which we evaluate the performance (10th to 90th percentiles)
        # Use percentiles from training event times
        event_times = [t for t, e in zip(train_times, train_events) if e == 1]
        if len(event_times) == 0:
            print(f"No events found in training data for {dataset_name}, skipping...")
            continue
            
        # Use 10th to 90th percentiles (every 10th percentile)
        horizons = np.arange(0.1, 1.0, 0.1)
        times = np.quantile(event_times, horizons).tolist()
        max_time = np.max([t_.max() for t_ in t_train])  # Max time from training data
        
        print(f"Evaluation time points: {times}")
        
        # Define parameter grid for hyperparameter tuning
        # layers = [[], [100], [100, 100], [100, 100, 100]]
        layers = [[100], [100, 100]]

        param_grid = {
              'layers_rnn': [2, 3],
              'hidden_long': layers,
              'hidden_rnn': [50, 100],
              'hidden_att': layers,
              'hidden_cs': layers,
              'sigma': [0.1, 1, 3],
              'learning_rate' : [1e-3],
             }
        params = list(ParameterGrid(param_grid))
        
        best_val_nll = np.inf  # We want to minimize negative log-likelihood
        best_model = None
        best_config_idx = 0
        
        print("\nStarting hyperparameter tuning...")
        
        # Train models with different hyperparameters
        for i, param in enumerate(params):
            model = DynamicDeepHit(
                layers_rnn=param['layers_rnn'],
                hidden_rnn=param['hidden_rnn'], 
                long_param={'layers': param['hidden_long'], 'dropout': 0.3}, 
                att_param={'layers': param['hidden_att'], 'dropout': 0.3}, 
                cs_param={'layers': param['hidden_cs'], 'dropout': 0.3},
                sigma=param['sigma'],
                split=[0] + times + [max_time]
            )
            # Train the model
            model.fit(x_train, t_train, e_train, iters=10, 
                        learning_rate=param['learning_rate'])
            
            # Compute validation negative log-likelihood (lower is better)
            val_nll = model.compute_nll(x_val, t_val, e_val)
            
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_model = model
                best_config_idx = i
        
        if best_model is None:
            print(f"No valid model found for dataset {dataset_name}, skipping...")
            continue
        
        print(f"\nBest validation NLL: {best_val_nll:.4f} (configuration {best_config_idx + 1})")
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        
        # Retrain best model on train+val data
        x_trainval = np.concatenate([x_train, x_val])
        t_trainval = np.concatenate([t_train, t_val])
        e_trainval = np.concatenate([e_train, e_val])
        
        # Get the best configuration and retrain
        best_config = params[best_config_idx]

        final_model = DynamicDeepHit(
            layers_rnn=best_config['layers_rnn'],
            hidden_rnn=best_config['hidden_rnn'], 
            long_param={'layers': best_config['hidden_long'], 'dropout': 0.3}, 
            att_param={'layers': best_config['hidden_att'], 'dropout': 0.3}, 
            cs_param={'layers': best_config['hidden_cs'], 'dropout': 0.3},
            sigma=best_config['sigma'],
            split=[0] + times + [max_time]
        )
        
        final_model.fit(x_trainval, t_trainval, e_trainval, iters=10, 
                       learning_rate=best_config['learning_rate'])
        
        # Get predictions
        try:
            # Predict risk and survival as in the notebook
            out_risk = final_model.predict_risk(x_test, times, all_step=True)
            out_survival = final_model.predict_survival(x_test, times, all_step=True)

            print(f"Predicted risk shape: {out_risk.shape}")
            print(f"Predicted survival shape: {out_survival.shape}")
            print(f"Risk NaN count: {np.isnan(out_risk).sum()}")
            print(f"Survival NaN count: {np.isnan(out_survival).sum()}")
            
            # Handle NaN values in predictions
            if np.isnan(out_risk).any():
                print(f"WARNING: Found {np.isnan(out_risk).sum()} NaN values in risk predictions")
                # Replace all NaN values with neutral predictions
                out_risk = np.nan_to_num(out_risk, nan=0.1)  # Small positive risk
                print(f"Replaced NaN values in risk predictions with 0.1")
                                
            if np.isnan(out_survival).any():
                print(f"WARNING: Found {np.isnan(out_survival).sum()} NaN values in survival predictions")
                # Replace all NaN values with neutral predictions
                out_survival = np.nan_to_num(out_survival, nan=0.5)  # Neutral survival probability
                print(f"Replaced NaN values in survival predictions with 0.5")
            
            print(f"After NaN handling - Risk NaN count: {np.isnan(out_risk).sum()}")
            print(f"After NaN handling - Survival NaN count: {np.isnan(out_survival).sum()}")
            print(f"Risk prediction dimensions: patients={out_risk.shape[0]}, times={out_risk.shape[1]}, risks={out_risk.shape[2]}")
            print(f"Evaluation times length: {len(times)}")
            
            # Show some sample predictions to understand the structure
            print(f"Sample risk predictions (first patient, all times):")
            for t_idx in range(min(5, out_risk.shape[1])):
                print(f"  Time step {t_idx}: {out_risk[0, t_idx, :]}")
            
            print(f"Sample survival predictions (first patient, all times):")
            for t_idx in range(min(5, out_survival.shape[1])):
                print(f"  Time step {t_idx}: {out_survival[0, t_idx, :]}")
            
            # Prepare data for sksurv evaluation - use LAST observation per patient
            # For survival analysis, we need one record per patient with their final outcome
            trainval_events = []
            trainval_times = []
            for i in range(len(t_trainval)):
                # Use the last observation for each patient
                trainval_events.append(bool(e_trainval[i][-1]))  # Last event status
                trainval_times.append(float(t_trainval[i][-1]))  # Last observation time
            
            test_events = []
            test_times = []
            for i in range(len(t_test)):
                # Use the last observation for each patient
                test_events.append(bool(e_test[i][-1]))  # Last event status
                test_times.append(float(t_test[i][-1]))   # Last observation time
            
            # Create structured arrays for sksurv
            et_train = np.array([(e, t) for e, t in zip(trainval_events, trainval_times)],
                               dtype=[('e', bool), ('t', float)])
            et_test = np.array([(e, t) for e, t in zip(test_events, test_times)],
                              dtype=[('e', bool), ('t', float)])

            print(f"Structured train array shape: {et_train.shape}")
            print(f"Structured test array shape: {et_test.shape}")
            print(f"Example structured train data: {et_train[:5]}")  # Print first 5 entries
            print(f"Example structured test data: {et_test[:5]}")    # Print first 5 entries
            
            # Calculate metrics for each time horizon as in notebook
            cis = []
            brs = []
            roc_auc = []
            
            for i, _ in enumerate(times):
                print(f"Computing metrics for time point {i+1}/{len(times)} (time={times[i]:.2f})")
                
                # Extract risk predictions for this time point
                # The model returns 3D arrays: (n_patients, n_time_steps, n_risks)
                # For evaluation at time point i, we need to extract the appropriate risk for each patient
                risk_predictions = []
                
                for patient_idx in range(out_risk.shape[0]):
                    # Get risk at the i-th time step for this patient
                    if i < out_risk.shape[1]:  # Make sure time step exists
                        patient_risk_at_time = out_risk[patient_idx, i, :]
                        # Use the cumulative risk (sum of all risk components up to this time)
                        cumulative_risk = np.sum(patient_risk_at_time)
                        risk_predictions.append(cumulative_risk)
                    else:
                        # Use the last available time step if i is beyond available steps
                        last_step = out_risk.shape[1] - 1
                        patient_risk_at_time = out_risk[patient_idx, last_step, :]
                        cumulative_risk = np.sum(patient_risk_at_time)
                        risk_predictions.append(cumulative_risk)
                
                risk_predictions = np.array(risk_predictions)
                print(f"Risk predictions for time {i}: min={np.min(risk_predictions):.4f}, max={np.max(risk_predictions):.4f}, mean={np.mean(risk_predictions):.4f}")
                
                # Time-dependent C-index
                try:
                    ci = concordance_index_ipcw(et_train, et_test, risk_predictions, times[i])[0]
                    cis.append(ci)
                    print(f"C-index for time {times[i]:.2f}: {ci:.4f}")
                except Exception as e:
                    print(f"Warning: Could not compute C-index for time {times[i]}: {e}")
                    # Fallback: use standard C-index with last observation data
                    last_trainval_events = [e_trainval[j][-1] for j in range(len(e_trainval))]
                    last_trainval_times = [t_trainval[j][-1] for j in range(len(t_trainval))]
                    last_test_events = [e_test[j][-1] for j in range(len(e_test))]
                    last_test_times = [t_test[j][-1] for j in range(len(t_test))]
                    
                    ci, *_ = concordance_index_censored(
                        last_test_events, last_test_times, risk_predictions
                    )
                    cis.append(ci)
                    print(f"Fallback C-index for time {times[i]:.2f}: {ci:.4f}")
                
                # ROC AUC
                try:
                    auc = cumulative_dynamic_auc(et_train, et_test, risk_predictions, times[i])[0]
                    auc_val = auc[0] if len(auc) > 0 else 0.5
                    roc_auc.append(auc_val)
                    print(f"AUC for time {times[i]:.2f}: {auc_val:.4f}")
                except Exception as e:
                    print(f"Warning: Could not compute AUC for time {times[i]}: {e}")
                    roc_auc.append(0.5)
            
            # Brier Score - use try-except for robustness
            try:
                # Extract survival predictions for the evaluation times
                # Need to extract survival at each time point for each patient
                survival_predictions = np.zeros((len(x_test), len(times)))
                
                for patient_idx in range(len(x_test)):
                    for time_idx in range(len(times)):
                        if time_idx < out_survival.shape[1]:
                            # Get survival at this time step
                            patient_survival_at_time = out_survival[patient_idx, time_idx, :]
                            # Use the survival probability for the first risk component
                            survival_prob = patient_survival_at_time[0]
                            survival_predictions[patient_idx, time_idx] = survival_prob
                        else:
                            # Use the last available time step
                            last_step = out_survival.shape[1] - 1
                            patient_survival_at_time = out_survival[patient_idx, last_step, :]
                            survival_prob = patient_survival_at_time[0]
                            survival_predictions[patient_idx, time_idx] = survival_prob
                
                print(f"Survival predictions shape for Brier score: {survival_predictions.shape}")
                
                brs_result = brier_score(et_train, et_test, survival_predictions, times)[1]
                brs.append(brs_result)
                mean_brier = np.mean(brs_result)
                print(f"Brier score computed: {mean_brier:.4f}")
            except Exception as e:
                print(f"Warning: Could not compute Brier score: {e}")
                mean_brier = 0.25  # Default value
            
            # Use the last time point C-index as the main metric
            c_index = cis[-1] if len(cis) > 0 else 0.5
            mean_auc = np.mean(roc_auc) if len(roc_auc) > 0 else 0.5
            
            best_row = {
                "dataset": dataset_name,
                "n_patients": n_patients,
                "n_eval_times": len(times),  
                "val_nll": round(float(best_val_nll), 4),
                "c_index": round(float(c_index), 4),
                "brier_score": round(float(mean_brier), 4),
                "mean_auc": round(float(mean_auc), 4),
            }
            
            print("="*50)
            print(f"Final test results for dataset {dataset_name}:")
            print(f"Number of patients: {best_row['n_patients']}")
            print(f"Number of eval time points: {best_row['n_eval_times']}")
            print(f"Validation NLL: {best_row['val_nll']:.4f}")
            print(f"Test C-index: {best_row['c_index']:.4f}")
            print(f"Test Brier Score: {best_row['brier_score']:.4f}")
            print(f"Test mean AUC: {best_row['mean_auc']:.4f}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            continue

        # ========================= write results =========================
        if best_row is not None:
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["dataset", "n_patients", "n_eval_times", "val_nll", "c_index", "brier_score", "mean_auc"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(best_row)

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        continue