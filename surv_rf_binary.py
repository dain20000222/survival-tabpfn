import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from sklearn.pipeline import make_pipeline
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, integrated_brier_score
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from pycox.models import DeepHitSingle, CoxPH
import torchtuples as tt
from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
import matplotlib.pyplot as plt
import random
from sksurv.nonparametric import SurvivalFunctionEstimator 
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Directory containing the datasets
data_dir = os.path.join("test1")

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# CSV path
csv_path = "rf_binary_evaluation.csv"

def construct_tabpfn_binary_trainset(x_train_imputed, y_train, cuts):
    """
    Construct TabPFN training dataset using binary classification approach.
    Include patient's own horizon in addition to provided time points.
    
    Args:
        cuts: Time points to evaluate at
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
        x_i = x_train_imputed.iloc[i].values
        
        # Add subject's own time to provided time points
        horizons = np.append(cuts.copy(), T_i)
        horizons.sort()
        
        if delta_i == 0:  # censored
            horizons = horizons[horizons <= T_i]

        # Create one row per admissible horizon
        for t in horizons:
            row = np.concatenate([x_i, [t]])
            label = int(delta_i and T_i <= t)  # Binary: 1 if event occurred by time t
            dataset_rows.append(row)
            binary_labels.append(label)
    
    feature_cols = list(x_train_imputed.columns) + ["eval_time"]
    X_tabpfn = pd.DataFrame(dataset_rows, columns=feature_cols)
    y_tabpfn = pd.Series(binary_labels)

    print(f"[DEBUG] Number of training samples for TabPFN: {X_tabpfn.shape[0]}")
    print(f"[DEBUG] Number of features (including eval_time): {X_tabpfn.shape[1]}")
    print(f"[DEBUG] Sample of features:\n{X_tabpfn.head()}")
    print(f"[DEBUG] Sample of labels:\n{y_tabpfn.head()}")
    
    # Print class distribution
    n_ones = sum(binary_labels)
    n_zeros = len(binary_labels) - n_ones
    print(f"\nTraining set class distribution:")
    print(f"Class 0 (no event): {n_zeros} ({n_zeros/len(binary_labels)*100:.1f}%)")
    print(f"Class 1 (event):    {n_ones} ({n_ones/len(binary_labels)*100:.1f}%)")
    print(f"Total samples:      {len(binary_labels)}")
    
    return X_tabpfn, y_tabpfn

def construct_tabpfn_binary_testset(x_test_imputed, eval_times):
    """Construct TabPFN test dataset using actual evaluation times for binary classification."""
    n_test = x_test_imputed.shape[0]
    
    test_rows = []
    patient_ids = []
    
    for i in range(n_test):
        x_i = x_test_imputed.iloc[i].values
        for t in eval_times:  
            row = np.concatenate([x_i, [t]])
            test_rows.append(row)
            patient_ids.append(i)
    
    feature_cols = list(x_test_imputed.columns) + ["eval_time"]
    X_tabpfn_test = pd.DataFrame(test_rows, columns=feature_cols)
    return X_tabpfn_test, np.array(patient_ids)

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
        
        # --------------- 70-15-15 Split (same as baseline.py) ---------------
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x, y, test_size=0.15, stratify=y["event"], random_state=SEED
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=0.1765, stratify=y_trainval["event"], random_state=SEED
        )
        # 0.1765 * 0.85 ≈ 0.15, so test/val are both 15%

        # One-hot encode using get_dummies (fit on TRAIN only!)
        x_train_ohe = pd.get_dummies(x_train, drop_first=True)
        x_val_ohe = pd.get_dummies(x_val, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)

        # Align columns so train/val/test match
        x_train_ohe, x_val_ohe = x_train_ohe.align(x_val_ohe, join="left", axis=1, fill_value=0)
        x_train_ohe, x_test_ohe = x_train_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)

        covariates_ohe = x_train_ohe.columns  # all columns are covariates after OHE

        # Impute missing values in covariates (fit on train only, same as baseline.py)
        imputer = SimpleImputer().fit(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = imputer.transform(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = pd.DataFrame(x_train_imputed, columns=covariates_ohe, index=x_train.index)
        x_val_imputed = imputer.transform(x_val_ohe.loc[:, covariates_ohe.tolist()])
        x_val_imputed = pd.DataFrame(x_val_imputed, columns=covariates_ohe, index=x_val.index)
        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # Evaluation time points (use train+val, same as baseline.py)
        eval_times = np.percentile(y_trainval["time"], np.arange(10, 100, 10))
        eval_times = np.unique(eval_times)
        max_trainval_time = y_trainval["time"].max()
        eval_times = eval_times[eval_times < max_trainval_time]
        test_mask = y_test["time"] < max_trainval_time
        y_test_filtered = y_test[test_mask]
        x_test_filtered = x_test_imputed[test_mask]
        eval_times = eval_times[
            (eval_times > y_test_filtered["time"].min()) & 
            (eval_times < y_test_filtered["time"].max())
        ]
        print(f"Evaluation time points: {eval_times}")

        # ========================= Train and evaluate using eval_times =========================
        print("Training binary TabPFN model using evaluation time points...")
        
        # Train binary model using eval_times
        X_tabpfn_train, y_tabpfn_train = construct_tabpfn_binary_trainset(
            x_train_imputed, y_train, cuts=eval_times
        )
        
        # Evaluate on validation set using eval_times
        X_tabpfn_val, val_patient_ids = construct_tabpfn_binary_testset(
            x_val_imputed, eval_times
        )

        # First training section (replace XGBoost)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=SEED,
            n_jobs=-1  # Use all available cores
        )
        rf_model.fit(X_tabpfn_train, y_tabpfn_train)

        probs = rf_model.predict_proba(X_tabpfn_val)
        
        # Process predictions directly at evaluation times
        event_probs = probs[:, 1]
        n_patients = len(x_val_imputed)
        F = np.zeros((n_patients, len(eval_times)))
        
        for i, t in enumerate(eval_times):
            mask = np.isclose(X_tabpfn_val["eval_time"], t)
            F[val_patient_ids[mask], i] = event_probs[mask]
        
        S = 1 - F
        S = np.clip(S, 1e-15, 1.0)  # Clip to prevent log(0)
        S = np.minimum.accumulate(S, axis=1)
        # Calculate time-dependent risk scores using cumulative hazard
        H = -np.log(S)
        # Use last time point for validation ranking
        risk_scores = H[:, -1]

        # Validation C-index
        val_c_index, *_ = concordance_index_censored(
            y_val["event"], y_val["time"], risk_scores
        )
        
        print(f"Validation C-index: {val_c_index:.4f}")
        
        # ========================= Final evaluation on test set =========================
        print("Evaluating on test set...")

        # ========== Retrain on train+val, evaluate on test (same as baseline.py) ==========
        # OHE and impute trainval/test
        x_trainval_ohe = pd.get_dummies(x_trainval, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)
        x_trainval_ohe, x_test_ohe = x_trainval_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)
        covariates_ohe = x_trainval_ohe.columns
        imputer = SimpleImputer().fit(x_trainval_ohe.loc[:, covariates_ohe.tolist()])
        x_trainval_imputed = imputer.transform(x_trainval_ohe.loc[:, covariates_ohe.tolist()])
        x_trainval_imputed = pd.DataFrame(x_trainval_imputed, columns=covariates_ohe, index=x_trainval.index)
        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)
        test_mask = y_test["time"] < max_trainval_time
        y_test_filtered = y_test[test_mask]
        x_test_filtered = x_test_imputed[test_mask]
        eval_times = eval_times[
            (eval_times > y_test_filtered["time"].min()) & 
            (eval_times < y_test_filtered["time"].max())
        ]
        
        # Train final binary model using eval_times
        X_tabpfn_train, y_tabpfn_train = construct_tabpfn_binary_trainset(
            x_trainval_imputed, y_trainval, cuts=eval_times
        )

        X_tabpfn_test, test_patient_ids = construct_tabpfn_binary_testset(
            x_test_filtered, eval_times
        )

        # First training section (replace XGBoost)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=SEED,
            n_jobs=-1  # Use all available cores
        )
        rf_model.fit(X_tabpfn_train, y_tabpfn_train)

        # Final evaluation using evaluation times

        probs = rf_model.predict_proba(X_tabpfn_test)

        # Process predictions directly at evaluation times
        event_probs = probs[:, 1]
        n_patients = len(x_test_filtered)
        F = np.zeros((n_patients, len(eval_times)))

        for i, t in enumerate(eval_times):
            mask = np.isclose(X_tabpfn_test["eval_time"], t)
            F[test_patient_ids[mask], i] = event_probs[mask]

        S = 1 - F
        S = np.clip(S, 1e-15, 1.0)  # Clip to prevent log(0)
        S = np.minimum.accumulate(S, axis=1)

        # Calculate time-dependent risk scores using cumulative hazard
        H = -np.log(S)
        
        print(f"[DEBUG] Survival probabilities shape: {S.shape}")
        print(f"[DEBUG] Sample of survival probabilities:\n{S[:5, :]}")

        print(f"[DEBUG] Cumulative hazard shape: {H.shape}")
        print(f"[DEBUG] Sample of cumulative hazard:\n{H[:5, :]}")

        # For C-index, use the last time point for ranking
        risk_scores_ranking = H[:, -1]

        # Final metrics on test set
        c_index, *_ = concordance_index_censored(
            y_test_filtered["event"], 
            y_test_filtered["time"], 
            risk_scores_ranking
        )

        ibs = integrated_brier_score(y_trainval, y_test_filtered, S, eval_times)
        
        _, mean_auc = cumulative_dynamic_auc(
            y_trainval, y_test_filtered, H, eval_times
        )

        best_row = {
            "dataset": dataset_name,
            "n_eval_times": len(eval_times),  # Use number of evaluation time points
            "score": round(float(val_c_index), 4),
            "c_index": round(float(c_index), 4),
            "ibs": round(float(ibs), 4),
            "mean_auc": round(float(mean_auc), 4),
        }
        
        print("="*50)
        print(f"Final test results for dataset {dataset_name}:")
        print(f"Number of eval time points: {best_row['n_eval_times']}")
        print(f"Validation C-index: {best_row['score']:.4f}")
        print(f"Test C-index: {best_row['c_index']:.4f}")
        print(f"Test interval Brier Score (IBS): {best_row['ibs']:.4f}")
        print(f"Test mean AUC: {best_row['mean_auc']:.4f}")

        # ========================= write results =========================
        if best_row is not None:
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["dataset", "n_eval_times", "score", "c_index", "ibs", "mean_auc"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(best_row)

    except Exception as e:
        print(f"Error: {e}")
        continue
