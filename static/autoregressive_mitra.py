import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, integrated_brier_score
import torch
import warnings
from autogluon.tabular import TabularDataset, TabularPredictor
import random
import tempfile, shutil
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
data_dir = os.path.join("data_static")

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# CSV path
csv_path = "autoregressive_mitra_evaluation2.csv"

def construct_mitra_binary_trainset(x_train, y_train, times):
    """
    Construct MITRA training dataset using binary classification approach.
    Include patient's own horizon in addition to provided time points.
    
    Parameters:
    - x_train: DataFrame of imputed covariates for training set
    - y_train: Structured array with 'time' and 'event' fields 
    - times: Array of time points to consider for binary classification

    Returns:
    - X_mitra: DataFrame of features for MITRA
    - y_mitra: Series of binary labels for MITRA
    """

    # Extract event/censoring times and status
    T = y_train["time"]
    delta = y_train["event"]
    n_train = len(T)
    
    dataset_rows = []
    binary_labels = []
    
    for i in range(n_train):
        T_i = T[i]
        delta_i = delta[i]
        x_i = x_train.iloc[i].values
        
        # Add subject's own time to provided time points
        horizons = np.append(times.copy(), T_i)
        horizons.sort()
        
        if delta_i == 0:  # censored
            horizons = horizons[horizons <= T_i]

        # Create one row per admissible horizon
        for t in horizons:
            row = np.concatenate([x_i, [t]])
            label = int(delta_i and T_i <= t)  # Binary: 1 if event occurred by time t
            dataset_rows.append(row)
            binary_labels.append(label)
    
    feature_cols = list(x_train.columns) + ["eval_time"]
    X_mitra = pd.DataFrame(dataset_rows, columns=feature_cols)
    y_mitra = pd.Series(binary_labels)
    
    # Print class distribution
    n_ones = sum(binary_labels)
    n_zeros = len(binary_labels) - n_ones
    print(f"\nTraining set class distribution:")
    print(f"Class 0 (no event): {n_zeros} ({n_zeros/len(binary_labels)*100:.1f}%)")
    print(f"Class 1 (event):    {n_ones} ({n_ones/len(binary_labels)*100:.1f}%)")
    print(f"Total samples:      {len(binary_labels)}")

    return X_mitra, y_mitra

def construct_mitra_binary_testset(x_test, times):
    """Construct MITRA test dataset using evaluation times for binary classification.
    
    Parameters:
    - x_test: DataFrame of imputed covariates for test set
    - times: Array of time points to consider for binary classification

    Returns:
    - X_mitra_test: DataFrame of features for MITRA test set
    - patient_ids: Array mapping each row to the original patient index
    """
    n_test = x_test.shape[0]

    test_rows = []
    patient_ids = []
    
    for i in range(n_test):
        x_i = x_test.iloc[i].values

        # Create one row per evaluation time
        for t in times:
            row = np.concatenate([x_i, [t]])
            test_rows.append(row)
            patient_ids.append(i)

    feature_cols = list(x_test.columns) + ["eval_time"]
    X_mitra_test = pd.DataFrame(test_rows, columns=feature_cols)

    return X_mitra_test, np.array(patient_ids)

def autoregressive_predict(X_mitra_train, y_mitra_train, x_set_imputed, y_set_struct, eval_times):
    """
    Autoregressive prediction using MITRA construct functions for both train and test.
    - X_mitra_train: DataFrame returned by construct_mitra_binary_trainset (features + 'eval_time')
    - y_mitra_train: Series of binary labels returned by construct_mitra_binary_trainset
    - x_set_imputed: DataFrame of covariates (one row per patient)
    - y_set_struct: structured array with fields 'time' and 'event'
    - eval_times: array of times to evaluate (must be sorted)
    Returns:
    - S: (n_patients x n_eval_times) survival probability matrix
    """
    # Build the base training DataFrame from the construct output to guarantee identical feature ordering
    train_df_base = X_mitra_train.copy().reset_index(drop=True)
    train_df_base["target"] = y_mitra_train.reset_index(drop=True)
    feature_cols = [c for c in train_df_base.columns if c != "target"]

    n_patients = len(x_set_imputed)
    m = len(eval_times)
    S = np.zeros((n_patients, m))

    T = y_set_struct["time"]
    delta = y_set_struct["event"]

    # Build full test table using the construct helper (ensures same column set + ordering)
    X_mitra_test, patient_ids = construct_mitra_binary_testset(x_set_imputed, eval_times)

    for i in range(n_patients):
        # Each patient starts from identical base
        working_train = train_df_base.copy().reset_index(drop=True)

        for j, t in enumerate(eval_times):
            row_idx = i * m + j
            # select the test-row and ensure the same feature column order
            row_df = X_mitra_test.iloc[[row_idx]].reset_index(drop=True)[feature_cols]

            tmp_dir = tempfile.mkdtemp()
            try:
                mitra_predictor = TabularPredictor(label='target', path=tmp_dir)
                mitra_predictor.fit(
                    TabularDataset(working_train),
                    hyperparameters={'MITRA': {'fine_tune': False}}
                )
                probs = mitra_predictor.predict_proba(TabularDataset(row_df)).to_numpy()
                surv_prob = float(probs[0, 0])
            except Exception:
                surv_prob = 0.0
            finally:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass

            S[i, j] = surv_prob

            # --- original: append true (X,y) for this patient at time t to working_train
            label = int(bool(delta[i]) and (T[i] <= t))
            append_row = row_df.copy()
            append_row["target"] = label
            working_train = pd.concat([working_train, append_row], ignore_index=True)

            # PSEUDO-LABELING: append predicted probability (event prob) instead of true label
            # Use model's predicted probability for class '1' (event).
            # event_prob = 1 - surv_prob 
            # append_row = row_df.copy()
            # append_row["target"] = event_prob
            # working_train = pd.concat([working_train, append_row], ignore_index=True)

        # ensure non-increasing survival over eval times
        S[i, :] = np.minimum.accumulate(S[i, :])

    return S

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
        df.drop(columns=['pid'], inplace=True)
        shape = df.shape
        print(f"Dataset shape: {shape}")

        # Define columns
        time_col = "time"
        event_col = "event"
        censored = (df["event"] == 0).sum()
        censored_percent = (censored)/len(df)*100
        print(f"Percentage of censored data: {censored_percent}%")
        covariates = df.columns.difference([time_col, event_col])

        # Define covariates and target variable
        x = df[covariates].copy()
        y = Surv.from_arrays(event=df[event_col].astype(bool), time=df[time_col])
        
        # Split into train/val/test (70%/15%/15%) with stratification on event
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x, y, test_size=0.15, stratify=y["event"], random_state=SEED
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=0.1765, stratify=y_trainval["event"], random_state=SEED
        )

        # One-hot encode using get_dummies (fit on train only)
        x_train_ohe = pd.get_dummies(x_train, drop_first=True)
        x_val_ohe = pd.get_dummies(x_val, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)

        # Align columns so train/val/test match (add missing columns with 0s)
        x_train_ohe, x_val_ohe = x_train_ohe.align(x_val_ohe, join="left", axis=1, fill_value=0)
        x_train_ohe, x_test_ohe = x_train_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)

        covariates_ohe = x_train_ohe.columns  # all columns are covariates after OHE

        # Impute missing values in covariates (fit on train only)
        imputer = SimpleImputer().fit(x_train_ohe.loc[:, covariates_ohe.tolist()])

        x_train_imputed = imputer.transform(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = pd.DataFrame(x_train_imputed, columns=covariates_ohe, index=x_train.index)

        x_val_imputed = imputer.transform(x_val_ohe.loc[:, covariates_ohe.tolist()])
        x_val_imputed = pd.DataFrame(x_val_imputed, columns=covariates_ohe, index=x_val.index)

        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # Determine evaluation time points (eval_times)
        # Use percentiles from 10th to 90th of train + val times
        eval_times = np.percentile(y_trainval["time"], np.arange(10, 100, 10))
        eval_times = np.unique(eval_times)

        # Filter test set to only include times within the range of train+val
        max_trainval_time = y_trainval["time"].max()
        test_mask = y_test["time"] < max_trainval_time
        y_test_filtered = y_test[test_mask]
        x_test_filtered = x_test_imputed[test_mask]

        # Further filter eval_times to be within the range of the filtered test set
        eval_times = eval_times[
            (eval_times > y_test_filtered["time"].min()) & 
            (eval_times < y_test_filtered["time"].max())
        ]
        print(f"Evaluation time points: {eval_times}")

        # Construct MITRA training set
        X_mitra_train, y_mitra_train = construct_mitra_binary_trainset(
            x_train_imputed, y_train, eval_times
        )
        
        # Sample 10000 rows if dataset is larger
        if len(X_mitra_train) > 10000:
            print(f"\nSampling 10000 rows from {len(X_mitra_train)} total rows...")
            sample_idx = np.random.choice(len(X_mitra_train), size=10000, replace=False)
            X_mitra_train = X_mitra_train.iloc[sample_idx].reset_index(drop=True)
            y_mitra_train = y_mitra_train.iloc[sample_idx].reset_index(drop=True)

        # Retrain on train+val, evaluate on test 
        # Combine train and val sets
        x_trainval_ohe = pd.get_dummies(x_trainval, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)
        x_trainval_ohe, x_test_ohe = x_trainval_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)
        covariates_ohe = x_trainval_ohe.columns

        # Impute missing values in covariates (fit on train+val only)
        imputer = SimpleImputer().fit(x_trainval_ohe.loc[:, covariates_ohe.tolist()])

        x_trainval_imputed = imputer.transform(x_trainval_ohe.loc[:, covariates_ohe.tolist()])
        x_trainval_imputed = pd.DataFrame(x_trainval_imputed, columns=covariates_ohe, index=x_trainval.index)

        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # Filter test set to only include times within the range of train+val
        test_mask = y_test["time"] < max_trainval_time
        y_test_filtered = y_test[test_mask]
        x_test_filtered = x_test_imputed[test_mask]

        # Further filter eval_times to be within the range of the filtered test set
        eval_times = eval_times[
            (eval_times > y_test_filtered["time"].min()) & 
            (eval_times < y_test_filtered["time"].max())
        ]
        
        # Construct MITRA training set 
        X_mitra_train, y_mitra_train = construct_mitra_binary_trainset(
            x_trainval_imputed, y_trainval, eval_times
        )

        # Sample 10000 rows if dataset is larger
        if len(X_mitra_train) > 10000:
            print(f"\nSampling 10000 rows from {len(X_mitra_train)} total rows...")
            sample_idx = np.random.choice(len(X_mitra_train), size=10000, replace=False)
            X_mitra_train = X_mitra_train.iloc[sample_idx].reset_index(drop=True)
            y_mitra_train = y_mitra_train.iloc[sample_idx].reset_index(drop=True)

        # Autoregressive test predictions (train base built from construct function)
        S = autoregressive_predict(X_mitra_train, y_mitra_train, x_test_filtered, y_test_filtered, eval_times)

        # Calculate time-dependent risk scores
        # Avoid -inf from log(0) by clipping survival probabilities to a small positive epsilon.
        # Keep original line commented for traceability.
        # risk_scores= -np.log(S)
        S_clipped = np.clip(S, 1e-12, 1.0)  # prevent zeros/negatives/infs
        risk_scores = -np.log(S_clipped)

        # Final metrics on test set
        c_index, *_ = concordance_index_censored(
            y_test_filtered["event"], 
            y_test_filtered["time"], 
            risk_scores[:, -1]  # Use last time-dependent risk score for ranking
        )

        ibs = integrated_brier_score(y_trainval, y_test_filtered, S, eval_times)
        
        _, mean_auc = cumulative_dynamic_auc(
            y_trainval, y_test_filtered, risk_scores, eval_times
        )

        best_row = {
            "dataset": dataset_name,
            "n_eval_times": len(eval_times),  
            "c_index": round(float(c_index), 4),
            "ibs": round(float(ibs), 4),
            "mean_auc": round(float(mean_auc), 4),
        }
        
        print("="*50)
        print(f"Final test results for dataset {dataset_name}:")
        print(f"Number of eval time points: {best_row['n_eval_times']}")
        print(f"Test C-index: {best_row['c_index']:.4f}")
        print(f"Test Integrated Brier Score (IBS): {best_row['ibs']:.4f}")
        print(f"Test mean AUC: {best_row['mean_auc']:.4f}")

        # ========================= write results =========================
        if best_row is not None:
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["dataset", "n_eval_times", "c_index", "ibs", "mean_auc"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(best_row)

    except Exception as e:
        print(f"Error: {e}")
        continue
