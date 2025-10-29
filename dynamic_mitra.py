import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from autogluon.tabular import TabularDataset, TabularPredictor
import tempfile, shutil
from sksurv.metrics import concordance_index_censored
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
csv_path = "dynamic_mitra_evaluation.csv"

TAU = 10

def construct_mitra_binary_trainset(x_train, y_train, times):
    """
    Construct MITRA training dataset using binary classification approach.
    Include patient's own horizon in addition to provided time points.
    """
    T = y_train["time"]
    delta = y_train["event"]
    n_train = len(T)
    
    dataset_rows = []
    binary_labels = []
    
    for i in range(n_train):
        T_i = T[i]
        delta_i = delta[i]
        x_i = x_train.iloc[i].values
        
        horizons = np.append(times.copy(), T_i)
        horizons.sort()
        
        if delta_i == 0:  # censored
            horizons = horizons[horizons <= T_i]

        for t in horizons:
            row = np.concatenate([x_i, [t]])
            label = int(delta_i and T_i <= t)
            dataset_rows.append(row)
            binary_labels.append(label)
    
    feature_cols = list(x_train.columns) + ["eval_time"]
    X_mitra = pd.DataFrame(dataset_rows, columns=feature_cols)
    y_mitra = pd.Series(binary_labels)
    
    n_ones = sum(binary_labels)
    n_zeros = len(binary_labels) - n_ones
    print(f"\nTraining set class distribution:")
    print(f"Class 0 (no event): {n_zeros} ({n_zeros/len(binary_labels)*100:.1f}%)")
    print(f"Class 1 (event):    {n_ones} ({n_ones/len(binary_labels)*100:.1f}%)")
    print(f"Total samples:      {len(binary_labels)}")

    return X_mitra, y_mitra


def build_patient_outcome(df):
    outcome = (
        df.sort_values("time2")
        .groupby("pid")
        .agg(
            t_event=("time2", "max"),
            status_event=("event", "last"),
        )
    )
    return outcome


def apply_locf_landmark(df, landmark_time, covariates, patient_outcome, tau):
    landmark_data = []

    for pid in df['pid'].unique():
        patient_data = df[df['pid'] == pid].sort_values('time2')

        t_event = patient_outcome.loc[pid, 't_event']
        status_event = patient_outcome.loc[pid, 'status_event']

        if t_event <= landmark_time:
            continue

        valid_obs = patient_data[patient_data['time2'] <= landmark_time]
        if len(valid_obs) == 0:
            continue

        latest_obs = valid_obs.iloc[-1]

        time_after_landmark = t_event - landmark_time
        remaining_time = min(time_after_landmark, tau)

        if (status_event == 1) and (time_after_landmark <= tau):
            final_event = 1
        else:
            final_event = 0
        
        new_row = {'pid': pid, 'time': remaining_time, 'event': final_event}

        for cov in covariates:
            new_row[cov] = latest_obs[cov]

        landmark_data.append(new_row)

    return pd.DataFrame(landmark_data)


for file_name in dataset_files:
    try:
        dataset_name = file_name.replace(".csv", "")
        file_path = os.path.join(data_dir, file_name)
        print("="*50)
        print(f"\nProcessing dataset: {dataset_name}")

        # Load datasets
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")

        # Define columns
        time_col = "time"
        time2_col = "time2"
        event_col = "event"

        covariates = df.columns.difference([time_col, time2_col, event_col, 'pid'])

        # Get event times for landmark calculation
        event_times = df[df['event'] == 1]['time2'].values
        if len(event_times) == 0:
            print(f"Skipping {dataset_name}: No events observed")
            continue

        # Pick three landmark times (25, 50, 75 quantiles of event times)
        landmark_times = np.quantile(event_times, [0.25, 0.5, 0.75])
        print(f"Landmark times: {landmark_times}")

        all_pids = df['pid'].unique()
        pids_trainval, pids_test = train_test_split(
            all_pids, test_size=0.15, random_state=SEED
        )
        pids_train, pids_val = train_test_split(
            pids_trainval, test_size=0.1765, random_state=SEED
        )

        patient_outcome = build_patient_outcome(df)
        
        results = []
        
        for i, landmark_time in enumerate(landmark_times):
            print(f"\nProcessing landmark time {i+1}: {landmark_time:.2f}")
            
            # Apply LOCF only for val and test
            landmark_df = apply_locf_landmark(
                df, landmark_time, covariates, patient_outcome, TAU
            )

            print("Example patient data after landmarking with tau:")
            print(landmark_df.head())
            print(f"Events: {landmark_df['event'].sum()}, Censored: {len(landmark_df) - landmark_df['event'].sum()}")

            # Build val/test from landmarked dataset (keep landmark-style evaluation)
            val_df = landmark_df[landmark_df["pid"].isin(pids_val)].copy()
            test_df = landmark_df[landmark_df["pid"].isin(pids_test)].copy()

            # Build training set from original time-varying rows (treat each row as independent)
            tv_rows = df[df['pid'].isin(pids_train)].copy()
            tv_rows = tv_rows.reset_index(drop=True).merge(
                patient_outcome.reset_index(), how='left', on='pid'
            )
            tv_rows['remaining_time'] = tv_rows['t_event'] - tv_rows['time2']
            
            # keep only rows observed before the final event/censoring
            tv_rows = tv_rows[tv_rows['remaining_time'] > 0].copy()
            if tv_rows.shape[0] == 0:
                print(f"Skipping landmark {landmark_time:.2f}: no valid training rows from time-varying data")
                continue

            # Check minimal sizes for val/test and train (post expansion later)
            if len(val_df) < 5 or len(test_df) < 5:
                print(f"Skipping landmark {landmark_time:.2f}: insufficient val/test data")
                continue

            # Prepare feature tables: training uses tv_rows; val/test use landmarked rows
            x_train_tv = tv_rows[covariates].copy()
            x_val = val_df[covariates].copy()
            x_test = test_df[covariates].copy()

            # One-hot encode and align columns across tv-train, val, test
            x_train_ohe = pd.get_dummies(x_train_tv, drop_first=True)
            x_val_ohe = pd.get_dummies(x_val, drop_first=True)
            x_test_ohe = pd.get_dummies(x_test, drop_first=True)

            all_columns = x_train_ohe.columns.union(x_val_ohe.columns).union(x_test_ohe.columns)
            for col in all_columns:
                if col not in x_train_ohe.columns:
                    x_train_ohe[col] = 0
                if col not in x_val_ohe.columns:
                    x_val_ohe[col] = 0
                if col not in x_test_ohe.columns:
                    x_test_ohe[col] = 0

            x_train_ohe = x_train_ohe.reindex(columns=sorted(all_columns))
            x_val_ohe = x_val_ohe.reindex(columns=sorted(all_columns))
            x_test_ohe = x_test_ohe.reindex(columns=sorted(all_columns))

            # Impute using training (tv) data imputer to keep pipeline consistent
            imputer = SimpleImputer(strategy='median')
            x_train_imputed = pd.DataFrame(
                imputer.fit_transform(x_train_ohe),
                columns=x_train_ohe.columns,
                index=x_train_ohe.index
            )
            x_val_imputed = pd.DataFrame(
                imputer.transform(x_val_ohe),
                columns=x_val_ohe.columns,
                index=x_val_ohe.index
            )
            x_test_imputed = pd.DataFrame(
                imputer.transform(x_test_ohe),
                columns=x_test_ohe.columns,
                index=x_test_ohe.index
            )

            # Remove constant columns based on training (tv) features
            constant_cols = x_train_imputed.columns[x_train_imputed.std() == 0]
            if len(constant_cols) > 0:
                print(f"Removing {len(constant_cols)} constant columns")
                x_train_imputed = x_train_imputed.drop(columns=constant_cols)
                x_val_imputed = x_val_imputed.drop(columns=constant_cols)
                x_test_imputed = x_test_imputed.drop(columns=constant_cols)

            # Build pseudo-survival y for each tv-row: time = remaining_time, event = status_event
            tv_y_df = pd.DataFrame({
                'time': tv_rows['remaining_time'].values,
                'event': tv_rows['status_event'].astype(int).values
            })
            y_tv = Surv.from_dataframe('event', 'time', tv_y_df)

            # Expand training set using construct function (treat each observation as independent "subject")
            X_mitra_train, y_mitra_train = construct_mitra_binary_trainset(x_train_imputed.reset_index(drop=True), y_tv, landmark_times)

            # rename 'eval_time' to 'time' so val/test pipeline (time column) matches training feature name
            if 'eval_time' in X_mitra_train.columns:
                X_mitra_train = X_mitra_train.rename(columns={'eval_time': 'time'})

            # attach label column for autogluon
            train_df_mitra = X_mitra_train.copy()
            train_df_mitra['event'] = y_mitra_train.astype(int).values

            # Build val/test frames: keep landmark-style 'time' (remaining to event up to tau) and event label
            val_df_mitra = x_val_imputed.reset_index(drop=True).copy()
            val_df_mitra['time'] = val_df['time'].reset_index(drop=True)
            val_df_mitra['event'] = val_df['event'].reset_index(drop=True).astype(int)

            test_df_mitra = x_test_imputed.reset_index(drop=True).copy()
            test_df_mitra['time'] = test_df['time'].reset_index(drop=True)
            test_df_mitra['event'] = test_df['event'].reset_index(drop=True).astype(int)

            # Train TabularPredictor (MITRA) on constructed training set
            tmp_dir = tempfile.mkdtemp()
            try:
                train_tab = TabularDataset(train_df_mitra)
                predictor = TabularPredictor(label='event', path=tmp_dir)
                predictor.fit(
                    train_tab,
                    hyperparameters={'MITRA': {'fine_tune': False}},
                    verbosity=0
                )

                # Predict event probabilities on validation and test
                val_tab = TabularDataset(val_df_mitra.drop(columns=['event']))
                probs_val = predictor.predict_proba(val_tab)
                # robustly get positive-class probability column
                if isinstance(probs_val, pd.Series):
                    event_prob_val = probs_val.to_numpy().astype(float)
                else:
                    if 1 in probs_val.columns:
                        col_pos = 1
                    elif '1' in probs_val.columns:
                        col_pos = '1'
                    else:
                        col_pos = probs_val.columns[-1]
                    event_prob_val = probs_val[col_pos].to_numpy().astype(float)

                test_tab = TabularDataset(test_df_mitra.drop(columns=['event']))
                probs_test = predictor.predict_proba(test_tab)
                if isinstance(probs_test, pd.Series):
                    event_prob_test = probs_test.to_numpy().astype(float)
                else:
                    if 1 in probs_test.columns:
                        col_pos = 1
                    elif '1' in probs_test.columns:
                        col_pos = '1'
                    else:
                        col_pos = probs_test.columns[-1]
                    event_prob_test = probs_test[col_pos].to_numpy().astype(float)

            finally:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass

            # Convert predicted event probabilities to survival probabilities and risk scores
            surv_val = np.clip(1.0 - event_prob_val, 1e-15, 1.0)
            surv_test = np.clip(1.0 - event_prob_test, 1e-15, 1.0)

            val_risk_scores = -np.log(surv_val)
            test_risk_scores = -np.log(surv_test)

            # Prepare survival objects for val/test (landmarked)
            y_val = Surv.from_dataframe('event', 'time', val_df)
            y_test = Surv.from_dataframe('event', 'time', test_df)

            # Compute C-index using same API as before
            val_cindex, *_ = concordance_index_censored(y_val["event"], y_val["time"], val_risk_scores)
            test_cindex, *_ = concordance_index_censored(y_test["event"], y_test["time"], test_risk_scores)

            result = {
                'dataset': dataset_name,
                'landmark_time': landmark_time,
                'tau': TAU,
                'n_patients': len(landmark_df),
                'n_events': int(landmark_df['event'].sum()),
                'val_cindex': float(val_cindex) if not np.isnan(val_cindex) else np.nan,
                'test_cindex': float(test_cindex) if not np.isnan(test_cindex) else np.nan,
            }
                    
            results.append(result)
                    
            print(f"Validation C-index: {result['val_cindex']:.4f}")
            print(f"Test C-index: {result['test_cindex']:.4f}")

        # Save results for this dataset
        if results:
            best_result = max(results, key=lambda x: x['val_cindex'] if not np.isnan(x['val_cindex']) else -np.inf)
            print(f"\nBest landmark time: {best_result['landmark_time']:.2f}")
            print(f"Best validation C-index: {best_result['val_cindex']:.4f}")
            print(f"Corresponding test C-index: {best_result['test_cindex']:.4f}")

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
