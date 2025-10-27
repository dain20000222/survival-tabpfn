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
csv_path = "landmark_mitra_evaluation.csv"

TAU = 10

def build_patient_outcome(df):
    outcome = (
        df.sort_values("time2")
        .groupby("pid")
        .agg(
            t_event = ("time2", "max"),
            status_event = ("event", "last")
        )

    )
    return outcome

def apply_locf_landmark(df, landmark_time, covariates, patient_outcome, tau):
    landmark_data = []

    for pid in df['pid'].unique():
        patient_data = df[df['pid'] == pid].sort_values('time2')

        # Check if patient has event/censoring time > landmark_time 
        t_event = patient_outcome.loc[pid, 't_event']
        status_event = patient_outcome.loc[pid, 'status_event']

        if t_event <= landmark_time:
            continue  # Remove patients with event/censoring <= landmark_time

        # Apply LOCF - find latest observation <= landmark_time
        valid_obs = patient_data[patient_data['time2'] <= landmark_time]
        if len(valid_obs) == 0:
            continue  # drop patient if we didn't actually observe them by the landmark

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
        shape = df.shape
        print(f"Dataset shape: {shape}")

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
            
            # Apply LOCF for this landmark
            landmark_df = apply_locf_landmark(
                df, landmark_time, covariates, patient_outcome, TAU
            )

            print("Example patient data after landmarking with tau:")
            print(landmark_df.head())
            print(f"Events: {landmark_df['event'].sum()}, Censored: {len(landmark_df) - landmark_df['event'].sum()}")

            # Split data based on patient IDs
            train_df = landmark_df[landmark_df["pid"].isin(pids_train)].copy()
            val_df = landmark_df[landmark_df["pid"].isin(pids_val)].copy()
            test_df = landmark_df[landmark_df["pid"].isin(pids_test)].copy()

            if len(train_df) < 10 or train_df['event'].sum() < 5:
                print(f"Skipping landmark {landmark_time:.2f}: insufficient train data")
                continue

            if len(val_df) < 5 or len(test_df) < 5:
                print(f"Skipping landmark {landmark_time:.2f}: insufficient val/test data")
                continue
            
            # Extract covariates
            x_train = train_df[covariates].copy()
            x_val = val_df[covariates].copy()
            x_test = test_df[covariates].copy()
            
            # One-hot encode categorical variables
            x_train_ohe = pd.get_dummies(x_train, drop_first=True)
            x_val_ohe = pd.get_dummies(x_val, drop_first=True)
            x_test_ohe = pd.get_dummies(x_test, drop_first=True)
            
            # Align columns between train, val, and test
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
            
            # Impute missing values
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
            
            # Remove constant columns to avoid singularity
            constant_cols = x_train_imputed.columns[x_train_imputed.std() == 0]
            if len(constant_cols) > 0:
                print(f"Removing {len(constant_cols)} constant columns")
                x_train_imputed = x_train_imputed.drop(columns=constant_cols)
                x_val_imputed = x_val_imputed.drop(columns=constant_cols)
                x_test_imputed = x_test_imputed.drop(columns=constant_cols)
                
            # Prepare survival data
            y_train = Surv.from_dataframe('event', 'time', train_df)
            y_val = Surv.from_dataframe('event', 'time', val_df)
            y_test = Surv.from_dataframe('event', 'time', test_df)
                
            # Train MITRA binary classifier to predict event within tau (use 'time' as covariate)
            # Build training dataframe: use imputed covariates and include the 'time' column as feature
            train_df_mitra = x_train_imputed.reset_index(drop=True).copy()
            train_df_mitra['time'] = train_df['time'].reset_index(drop=True)
            train_df_mitra['event'] = train_df['event'].reset_index(drop=True).astype(int)

            val_df_mitra = x_val_imputed.reset_index(drop=True).copy()
            val_df_mitra['time'] = val_df['time'].reset_index(drop=True)
            val_df_mitra['event'] = val_df['event'].reset_index(drop=True).astype(int)

            test_df_mitra = x_test_imputed.reset_index(drop=True).copy()
            test_df_mitra['time'] = test_df['time'].reset_index(drop=True)
            test_df_mitra['event'] = test_df['event'].reset_index(drop=True).astype(int)


            # Train TabularPredictor (MITRA) on training set
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

                # Get event probability column 
                event_prob_val = probs_val.iloc[:, -1].to_numpy().astype(float)

                test_tab = TabularDataset(test_df_mitra.drop(columns=['event']))
                probs_test = predictor.predict_proba(test_tab)
                event_prob_test = probs_test.iloc[:, -1].to_numpy().astype(float)
                
            finally:
                # cleanup autogluon tmp dir
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass

            # Convert predicted event probabilities to survival probabilities and risk scores
            # survival prob = 1 - P(event within tau)
            surv_val = np.clip(1.0 - event_prob_val, 1e-15, 1.0)
            surv_test = np.clip(1.0 - event_prob_test, 1e-15, 1.0)

            # S matrix (n_patients x 1) and monotonicity not needed for single-horizon
            # risk score: -log(S)
            val_risk_scores = -np.log(surv_val)
            test_risk_scores = -np.log(surv_test)

            # Compute C-index using same API as before (use last/time-independent risk score)
            val_cindex, *_ = concordance_index_censored(y_val["event"], y_val["time"], val_risk_scores)
            test_cindex, *_ = concordance_index_censored(y_test["event"], y_test["time"], test_risk_scores)

            # Save result if we have valid metrics
            result = {
                'dataset': dataset_name,
                'landmark_time': landmark_time,
                'tau': TAU,
                'n_patients': len(landmark_df),
                'n_events': landmark_df['event'].sum(),
                'val_cindex': val_cindex,
                'test_cindex': test_cindex,
            }
                    
            results.append(result)
                    
            print(f"Validation C-index: {val_cindex:.4f}")
            print(f"Test C-index: {test_cindex:.4f}")

        # Save results for this dataset
        if results:
            # Find best model based on validation C-index
            best_result = max(results, key=lambda x: x['val_cindex'] if not np.isnan(x['val_cindex']) else 0)
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
