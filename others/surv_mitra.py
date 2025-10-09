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
from sksurv.nonparametric import SurvivalFunctionEstimator 
import tempfile, shutil
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
csv_path = "mitra_evaluation.csv"

def construct_tabpfn_trainset(x_train_imputed, y_train, cuts):
    """
    Construct TabPFN training dataset using KM quantile cuts.
    Include patient's own horizon in addition to KM cuts.
    """
    # Extract event/censoring times and status
    T_train = y_train["time"]
    delta_train = y_train["event"]
    n_train = len(T_train)

    dataset_rows = []
    class_labels = []

    # Create examples for each patient
    for i in range(n_train):
        x_i = x_train_imputed.iloc[i].values
        T_i = T_train[i]
        delta_i = delta_train[i]

        # Add subject's own time to KM cuts
        horizons = np.append(cuts.copy(), T_i)
        horizons.sort()

        # Create examples at each timepoint
        for t_j in horizons:
            # Assign class label based on original time and current timepoint
            if T_i < t_j:
                label = "A"  # Event/censoring happened before this timepoint
            elif T_i > t_j:
                label = "B"  # Event/censoring will happen after this timepoint
            elif T_i == t_j and delta_i == 0:
                label = "C"  # Censored at this timepoint
            else:  # T_i == t_j and delta_i == 1
                label = "D"  # Event at this timepoint

            # Feature = original + eval_time
            row = np.concatenate([x_i, [t_j]])
            dataset_rows.append(row)
            class_labels.append(label)

    feature_cols = list(x_train_imputed.columns) + ["eval_time"]
    X_tabpfn_train = pd.DataFrame(dataset_rows, columns=feature_cols)
    y_tabpfn_train = pd.Series(class_labels)
    
    # Print class distribution
    class_dist = y_tabpfn_train.value_counts()
    total = len(y_tabpfn_train)
    print("\nTraining set class distribution:")
    for label in ["A", "B", "C", "D"]:
        count = class_dist.get(label, 0)
        print(f"Class {label}: {count} ({count/total*100:.1f}%)")
    print(f"Total samples: {total}")
    
    return X_tabpfn_train, y_tabpfn_train

def construct_tabpfn_testset(x_test_imputed, eval_times):
    """
    Construct TabPFN test dataset using actual evaluation times.
    No interpolation needed.
    """
    n_test = x_test_imputed.shape[0]
    
    test_rows = []
    patient_ids = []
    
    for i in range(n_test):
        x_i = x_test_imputed.iloc[i].values
        for t in eval_times:  # Use actual evaluation times
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
        x_val_ohe   = pd.get_dummies(x_val, drop_first=True)
        x_test_ohe  = pd.get_dummies(x_test, drop_first=True)

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
        eval_times = eval_times[(eval_times > y_test_filtered["time"].min()) & (eval_times < y_test_filtered["time"].max())]
        print(f"Evaluation time points: {eval_times}")

        # Keep the masked test index for alignment inside the n_bins loop
        masked_test_index = x_test_filtered.index

        # Keep originals for metrics (explicit names for clarity)
        y_train_orig = y_train
        y_test_orig = y_test_filtered

        # ========================= Train on fixed evaluation times =========================
        print("Training TabPFN with fixed evaluation times...")

        # ========== First train on train set and validate ==========
        # Use evaluation times directly as cuts for discretization
        cuts = eval_times.copy()
        print(f"Using evaluation times as cuts: {cuts}")

        # Train using KM cuts
        X_tabpfn_train, y_tabpfn_train = construct_tabpfn_trainset(
            x_train_imputed, y_train, cuts
        )

        # Sample 10000 rows if dataset is larger
        if len(X_tabpfn_train) > 10000:
            print(f"\nSampling 10000 rows from {len(X_tabpfn_train)} total rows...")
            sample_idx = np.random.choice(len(X_tabpfn_train), size=10000, replace=False)
            X_tabpfn_train = X_tabpfn_train.iloc[sample_idx].reset_index(drop=True)
            y_tabpfn_train = y_tabpfn_train.iloc[sample_idx].reset_index(drop=True)
    
        # Convert to TabularDataset
        train_data = pd.DataFrame(X_tabpfn_train)
        train_data['target'] = y_tabpfn_train
        train_data = TabularDataset(train_data)
        
        # Initialize and train Mitra
        print("Training Mitra classifier...")
        tmp_dir = tempfile.mkdtemp()
        mitra_predictor = TabularPredictor(label='target', path=tmp_dir)
        mitra_predictor.fit(
            train_data,
            hyperparameters={
                'MITRA': {'fine_tune': False}
            }
        )

        # Validate using actual evaluation times
        X_tabpfn_val, val_patient_ids = construct_tabpfn_testset(
            x_val_imputed, eval_times  # Use evaluation times
        )
        val_data = TabularDataset(pd.DataFrame(X_tabpfn_val))
        probs = mitra_predictor.predict_proba(val_data)
        probs = probs.to_numpy()

        shutil.rmtree(tmp_dir)
        
        # Process predictions (no interpolation needed)
        n_patients = len(x_val_imputed)
        F = np.zeros((n_patients, len(eval_times)))

        for i, t in enumerate(eval_times):
            mask = np.isclose(X_tabpfn_val["eval_time"], t)
            for idx in np.where(mask)[0]:
                pid = val_patient_ids[idx]
                pA, pB, pC, pD = probs[idx]
                denom = pB + pC + pD + 1e-12  # at-risk at t
                F[pid, i] = pD / denom
        
        # Calculate survival probabilities
        S = np.clip(np.cumprod(1.0 - F, axis=1), 0.0, 1.0)
        S = np.minimum.accumulate(S, axis=1)

        # Calculate risk scores
        risk_scores_ranking = 1.0 - S[:, -1]

        # Validation C-index
        val_c_index, *_ = concordance_index_censored(y_val["event"], y_val["time"], risk_scores_ranking)

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
        eval_times = eval_times[(eval_times > y_test_filtered["time"].min()) & (eval_times < y_test_filtered["time"].max())]

        # Use evaluation times directly as cuts for discretization
        cuts = eval_times.copy()
        print(f"Using evaluation times as cuts: {cuts}")
        
        # Train TabPFN on trainval data
        X_tabpfn_train, y_tabpfn_train = construct_tabpfn_trainset(
            x_trainval_imputed, y_trainval, cuts
        )

       # Sample 10000 rows if dataset is larger
        if len(X_tabpfn_train) > 10000:
            print(f"\nSampling 10000 rows from {len(X_tabpfn_train)} total rows...")
            sample_idx = np.random.choice(len(X_tabpfn_train), size=10000, replace=False)
            X_tabpfn_train = X_tabpfn_train.iloc[sample_idx].reset_index(drop=True)
            y_tabpfn_train = y_tabpfn_train.iloc[sample_idx].reset_index(drop=True)

        # Convert to TabularDataset with proper label column
        train_data = pd.DataFrame(X_tabpfn_train)
        train_data['target'] = y_tabpfn_train
        train_data = TabularDataset(train_data)

        print("Training Mitra classifier on classification dataset...")
        tmp_dir = tempfile.mkdtemp()
        mitra_predictor = TabularPredictor(label='target', path=tmp_dir)
        mitra_predictor.fit(
            train_data,
            hyperparameters={
                'MITRA': {'fine_tune': False}
            }
        )

        # Predict on unique bins only
        X_tabpfn_test, test_patient_ids = construct_tabpfn_testset(x_test_filtered, eval_times)
        # Convert test data to TabularDataset and get predictions
        test_data = TabularDataset(pd.DataFrame(X_tabpfn_test))
        probs = mitra_predictor.predict_proba(test_data)
        probs = probs.to_numpy()

        shutil.rmtree(tmp_dir)

        # Process predictions directly at evaluation times
        n_patients = len(x_test_filtered)
        F = np.zeros((n_patients, len(eval_times)))

        for i, t in enumerate(eval_times):
            mask = np.isclose(X_tabpfn_test["eval_time"], t)
            for idx in np.where(mask)[0]:
                pid = test_patient_ids[idx]
                pA, pB, pC, pD = probs[idx]
                denom = pB + pC + pD + 1e-12  # at-risk at t
                F[pid, i] = pD / denom
        
        # Calculate survival probabilities
        S = np.clip(np.cumprod(1.0 - F, axis=1), 0.0, 1.0)
        S = np.minimum.accumulate(S, axis=1)

        # Risk scores from final survival probabilities
        risk_scores_ranking = 1.0 - S[:, -1]

        # Calculate time-dependent risk scores using cumulative hazard
        H = -np.log(S)

        # Final metrics on test set
        c_index, *_ = concordance_index_censored(
            y_test_orig["event"], 
            y_test_orig["time"], 
            risk_scores_ranking)

        ibs = integrated_brier_score(y_trainval, y_test_orig, S, eval_times)

        _, mean_auc = cumulative_dynamic_auc(
            y_trainval, 
            y_test_orig, 
            H, 
            eval_times)

        best_row = {
            "dataset": dataset_name,
            "n_eval_times": len(eval_times),
            "score": round(float(val_c_index), 4),
            "c_index": round(float(c_index), 4),
            "ibs": round(float(ibs), 4),
            "mean_auc": round(float(mean_auc), 4),
        }
        
        print("="*50)
        print(f"Final test results for dataset {dataset_name}:")
        print(f"Number of eval time points: {best_row['n_eval_times']}")
        print(f"Validation C-index (Score): {best_row['score']:.4f}")
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