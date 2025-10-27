import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv
import warnings
import random
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Track runtime
import time
start_time = time.time()

# Directory containing the datasets
data_dir = os.path.join("data")

# List all CSV files in the directory
dataset_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# CSV path
csv_path = "landmark_cox_evaluation.csv"

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
        shape = df.shape
        print(f"Dataset shape: {shape}")

        # Define columns
        time_col = "time"
        time2_col = "time2"
        event_col = "event"
        censored = (df["event"] == 0).sum()
        censored_percent = (censored)/len(df)*100
        print(f"Percentage of censored data: {censored_percent}%")
        covariates = df.columns.difference([time_col, time2_col, event_col, 'pid'])

        # Get event times for landmark calculation
        event_times = df[df['event'] == 1]['time2'].values
        if len(event_times) == 0:
            print(f"Skipping {dataset_name}: No events observed")
            continue
            
        # Step 1: Pick three landmark times (25, 50, 75 quantiles of event times)
        landmark_times = np.quantile(event_times, [0.25, 0.5, 0.75])
        print(f"Landmark times: {landmark_times}")
        
        def apply_locf_landmark(df, landmark_time, covariates):
            """
            Apply LOCF approach for a given landmark time
            """
            landmark_data = []
            
            for pid in df['pid'].unique():
                patient_data = df[df['pid'] == pid].sort_values('time2')
                
                # Step 2: Check if patient has event/censoring time > landmark_time
                max_time = patient_data['time2'].max()
                if max_time <= landmark_time:
                    continue  # Remove patients with event/censoring <= landmark_time
                
                # Step 3: Apply LOCF - find latest observation <= landmark_time
                valid_obs = patient_data[patient_data['time2'] <= landmark_time]
                if len(valid_obs) == 0:
                    # If no observation before landmark, use the first observation
                    # This handles cases where first observation is after landmark
                    latest_obs = patient_data.iloc[0]
                else:
                    latest_obs = valid_obs.iloc[-1]  # Latest observation
                
                # Step 4: Define new remaining time and event indicator
                # The remaining time should be the time from landmark to the actual event/censoring
                final_obs = patient_data.iloc[-1]  # Last observation (contains the event info)
                remaining_time = final_obs['time2'] - landmark_time
                
                # Event indicator: 1 if original event occurred after landmark, 0 otherwise
                final_event = final_obs['event']
                
                # Create new row with LOCF covariates
                new_row = {'pid': pid, 'time': remaining_time, 'event': final_event}
                for cov in covariates:
                    new_row[cov] = latest_obs[cov]
                
                landmark_data.append(new_row)
            
            return pd.DataFrame(landmark_data)
        
        # Process each landmark time
        results = []
        
        for i, landmark_time in enumerate(landmark_times):
            print(f"\nProcessing landmark time {i+1}: {landmark_time:.2f}")
            
            # Apply LOCF for this landmark
            landmark_df = apply_locf_landmark(df, landmark_time, covariates)

            print(f"Example patient data after landmarking:")
            print(landmark_df.head())
            print(f"Events: {landmark_df['event'].sum()}, Censored: {len(landmark_df) - landmark_df['event'].sum()}")
            
            # Split into train/val/test (70%/15%/15%) based on patient IDs
            unique_pids = landmark_df["pid"].unique()
            pids_trainval, pids_test = train_test_split(
                unique_pids, test_size=0.15, random_state=SEED
            )
            
            pids_train, pids_val = train_test_split(
                pids_trainval, test_size=0.1765, random_state=SEED
            )

            # Split data based on patient IDs
            train_df = landmark_df[landmark_df["pid"].isin(pids_train)].copy()
            val_df = landmark_df[landmark_df["pid"].isin(pids_val)].copy()
            test_df = landmark_df[landmark_df["pid"].isin(pids_test)].copy()
            
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
            
            # Skip if too few samples or events
            if len(train_df) < 10 or train_df['event'].sum() < 5:
                print(f"Skipping landmark {landmark_time:.2f}: insufficient data")
                continue
                
            # Fit Cox model
            cox_model = CoxPHSurvivalAnalysis()
            cox_model.fit(x_train_imputed, y_train)
            
            # Evaluate on validation set
            val_risk_scores = cox_model.predict(x_val_imputed)
            val_cindex, *_ = concordance_index_censored(y_val["event"], y_val["time"], val_risk_scores)

            # Evaluate on test set
            test_risk_scores = cox_model.predict(x_test_imputed)
            test_cindex,  *_ = concordance_index_censored(y_test["event"], y_test["time"], test_risk_scores)


            # Save result if we have valid metrics
            result = {
                'dataset': dataset_name,
                'landmark_time': landmark_time,
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
                fieldnames = ['dataset', 'landmark_time', 'n_patients', 'n_events', 
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

# Print total runtime
end_time = time.time()
total_runtime = end_time - start_time
hours = int(total_runtime // 3600)
minutes = int((total_runtime % 3600) // 60)
seconds = int(total_runtime % 60)
print(f"Total runtime: {hours}h {minutes}m {seconds}s")

