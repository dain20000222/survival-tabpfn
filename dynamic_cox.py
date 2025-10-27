import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from lifelines import CoxTimeVaryingFitter
from sksurv.metrics import concordance_index_ipcw
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
csv_path = "cox_timevarying_evaluation.csv"

def c_index(Prediction, Time_survival, Death, Time):
    '''
    Cause-specific c(t)-index calculation
    '''
    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
        
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1

    Num = np.sum(((A)*N_t)*Q)
    Den = np.sum((A)*N_t)

    if Den == 0:
        result = 0
    else:
        result = float(Num/Den)

    return result

def brier_score(Prediction, Time_survival, Death, Time):
    '''
    Time-dependent Brier score
    '''
    N = len(Prediction)
    y_true = ((Time_survival <= Time) * Death).astype(float)
    return np.mean((Prediction - y_true)**2)

def calculate_time_dependent_metrics(scores, time, event, eval_times):
    '''
    Calculate time-dependent metrics at each evaluation time
    '''
    metrics = []
    for t in eval_times:
        # Get predictions at this time
        pred_t = scores
        
        # Calculate metrics
        c_idx = c_index(pred_t, time, event, t)
        bs = brier_score(pred_t, time, event, t)
        
        metrics.append({
            'time': t,
            'c_index': c_idx if c_idx != -1 else np.nan,
            'brier_score': bs
        })
    
    return pd.DataFrame(metrics)

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
        time2_col = "time2"  # Add time2 column
        event_col = "event"
        censored = (df["event"] == 0).sum()
        censored_percent = (censored)/len(df)*100
        print(f"Percentage of censored data: {censored_percent}%")
        covariates = df.columns.difference([time_col, time2_col, event_col, 'pid'])

        # Debug: Initial data check
        print(f"\n[DEBUG] Initial data check:")
        print(f"Time column range: {df[time_col].min():.2f} to {df[time_col].max():.2f}")
        print(f"Time2 column range: {df[time2_col].min():.2f} to {df[time2_col].max():.2f}")
        print(f"Number of unique patients: {df['pid'].nunique()}")

        # Define covariates 
        x = df[covariates].copy()
        
        # Split into train/val/test (70%/15%/15%) based on patient IDs
        unique_pids = df["pid"].unique()
        pids_trainval, pids_test = train_test_split(
            unique_pids, test_size=0.15, random_state=SEED
        )
        
        pids_train, pids_val = train_test_split(
            pids_trainval, test_size=0.1765, random_state=SEED
        )

        # Split data based on patient IDs
        x_train = x[df["pid"].isin(pids_train)].copy()
        x_val = x[df["pid"].isin(pids_val)].copy()
        x_test = x[df["pid"].isin(pids_test)].copy()
        
        # Get corresponding time information
        df_train = df[df["pid"].isin(pids_train)].copy()
        df_val = df[df["pid"].isin(pids_val)].copy()
        df_test = df[df["pid"].isin(pids_test)].copy()

        # One-hot encode using get_dummies (fit on train only)
        x_train_ohe = pd.get_dummies(x_train, drop_first=True)
        x_val_ohe = pd.get_dummies(x_val, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)

        # Align columns of val/test to train (add missing columns with 0s)
        x_train_ohe, x_val_ohe = x_train_ohe.align(x_val_ohe, join="left", axis=1, fill_value=0)
        x_train_ohe, x_test_ohe = x_train_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)

        covariates_ohe = x_train_ohe.columns  # all columns are covariates after OHE

        # Impute missing values in covariates (fit on train only)
        imputer = SimpleImputer().fit(x_train_ohe.loc[:, covariates_ohe.tolist()])

        x_train_imputed = imputer.transform(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = pd.DataFrame(x_train_imputed, columns=covariates_ohe, index=x_train.index)

        # Add back the required columns for time-varying Cox
        x_train_final = pd.concat([
            df_train[['pid', event_col, time_col, time2_col]],
            x_train_imputed
        ], axis=1)

        x_val_imputed = imputer.transform(x_val_ohe.loc[:, covariates_ohe.tolist()])
        x_val_imputed = pd.DataFrame(x_val_imputed, columns=covariates_ohe, index=x_val.index)

        # Add back the required columns for time-varying Cox
        x_val_final = pd.concat([
            df_val[['pid', event_col, time_col, time2_col]], 
            x_val_imputed
        ], axis=1)

        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # Add back the required columns for time-varying Cox
        x_test_final = pd.concat([
            df_test[['pid', event_col, time_col, time2_col]], 
            x_test_imputed
        ], axis=1)

        # Determine evaluation times using time2 column since it's the end time for intervals
        eval_times = np.percentile(
            np.concatenate([df_train["time2"], df_val["time2"]]),
            np.arange(10, 100, 10)
        )
        # eval_times = np.unique(eval_times)

        # Get maximum time from training and validation data
        max_trainval_time = max(df_train["time2"].max(), df_val["time2"].max())

        # Filter test set to only include intervals within training range
        test_mask = df_test["time2"] <= max_trainval_time
        x_test_final_filtered = x_test_final[test_mask].copy()

        # Debug: Data split sizes
        print(f"\n[DEBUG] Data split sizes:")
        print(f"Train patients: {len(pids_train)}")
        print(f"Val patients: {len(pids_val)}")
        print(f"Test patients: {len(pids_test)}")

        # Train Cox model on training data
        print("Training Cox model on training data...")
        ctv = CoxTimeVaryingFitter(penalizer=0.1)
        ctv.fit(
            x_train_final, 
            id_col="pid",
            event_col="event",
            start_col="time",
            stop_col="time2"
        )

        # Get predictions for validation set
        val_scores = ctv.predict_partial_hazard(x_val_final).values

        # Validation metrics
        print("Calculating validation metrics...")
        val_metrics = calculate_time_dependent_metrics(
            scores=val_scores,
            time=x_val_final['time2'].values,
            event=x_val_final['event'].values,
            eval_times=eval_times
        )
        print("\nValidation metrics summary:")
        print(val_metrics.mean())

        # Retrain on combined train+val data
        x_trainval = x[df["pid"].isin(pids_trainval)].copy()
        df_trainval = df[df["pid"].isin(pids_trainval)].copy()

        # One-hot encode using get_dummies
        x_trainval_ohe = pd.get_dummies(x_trainval, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)

        # Align columns
        x_trainval_ohe, x_test_ohe = x_trainval_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)
        covariates_ohe = x_trainval_ohe.columns

        # Impute missing values
        imputer = SimpleImputer().fit(x_trainval_ohe.loc[:, covariates_ohe.tolist()])

        x_trainval_imputed = imputer.transform(x_trainval_ohe.loc[:, covariates_ohe.tolist()])
        x_trainval_imputed = pd.DataFrame(x_trainval_imputed, columns=covariates_ohe, index=x_trainval.index)

        # Create final trainval dataset with required columns
        x_trainval_final = pd.concat([
            df_trainval[['pid', event_col, time_col, time2_col]],
            x_trainval_imputed
        ], axis=1)

        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # Create final test dataset with required columns
        x_test_final = pd.concat([
            df_test[['pid', event_col, time_col, time2_col]],
            x_test_imputed
        ], axis=1)

        # Filter test set to only include intervals within training range
        test_mask = x_test_final["time2"] <= max_trainval_time
        x_test_final_filtered = x_test_final[test_mask].copy()

        # Train final model
        print("Training final model on train+val data...")
        ctv_final = CoxTimeVaryingFitter(penalizer=0.1)
        ctv_final.fit(
            x_trainval_final,
            id_col="pid",
            event_col="event",
            start_col="time",
            stop_col="time2"
        )

        # Get predictions for filtered test set
        test_scores = ctv_final.predict_partial_hazard(x_test_final_filtered).values

        # Test metrics
        print("\nCalculating test metrics...")
        test_metrics = calculate_time_dependent_metrics(
            scores=test_scores,
            time=x_test_final_filtered['time2'].values,
            event=x_test_final_filtered['event'].values,
            eval_times=eval_times
        )
        print("\nTest metrics summary:")
        print(test_metrics.mean())

        # Create lists of metrics at each time point
        val_times = val_metrics['time'].tolist()
        val_cindices = val_metrics['c_index'].tolist()
        val_brier = val_metrics['brier_score'].tolist()
        
        test_times = test_metrics['time'].tolist()
        test_cindices = test_metrics['c_index'].tolist()
        test_brier = test_metrics['brier_score'].tolist()
        
        # Create results dictionary with lists
        best_row = {
            "dataset": dataset_name,
            "eval_times": val_times,  # Same for both val and test
            "val_cindex": val_cindices,
            "val_brier": val_brier,
            "test_cindex": test_cindices,
            "test_brier": test_brier
        }

        # Write results
        if best_row is not None:
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                fieldnames = [
                    "dataset",
                    "eval_times",
                    "val_cindex",
                    "val_brier",
                    "test_cindex",
                    "test_brier"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                # Convert lists to strings for CSV writing
                row_to_write = {
                    "dataset": dataset_name,
                    "eval_times": ','.join(map(str, val_times)),
                    "val_cindex": ','.join(map(str, val_cindices)),
                    "val_brier": ','.join(map(str, val_brier)),
                    "test_cindex": ','.join(map(str, test_cindices)),
                    "test_brier": ','.join(map(str, test_brier))
                }
                writer.writerow(row_to_write)

    except Exception as e:
        print(f"Error: {e}")
        continue
def plot_time_dependent_metrics(csv_path, metric_type='c_index'):
    """
    Plot time-dependent metrics for all datasets using quantiles (10th–90th),
    extending shorter sequences, and annotating values.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Read CSV
    results_df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 6))

    # Quantile labels for x-axis
    quantile_labels = [f"{i}th" for i in range(10, 100, 10)]  # ['10th','20th',...,'90th']

    # Process each dataset
    for _, row in results_df.iterrows():
        if metric_type == 'c_index':
            values = np.array([float(x) for x in row['test_cindex'].split(',')])
        else:
            values = np.array([float(x) for x in row['test_brier'].split(',')])
        times = np.array([float(x) for x in row['eval_times'].split(',')])

        # Extend shorter sequences to 9 by repeating last value
        if len(values) < 9:
            values = np.pad(values, (0, 9 - len(values)), 'edge')
  
        indices = range(9)
        
        # Plot dataset line
        plt.plot(indices, values, marker='o', label=f"{row['dataset']}")
        
        # Annotate each point with its metric value
        for x, y in zip(indices, values):
            plt.text(x, y + 0.005, f"{y:.3f}", ha='center', va='bottom', fontsize=8, color='black')

    # Titles and labels
    if metric_type == 'c_index':
        title = 'Time-dependent C-index'
        ylabel = 'C-index'
    else:
        title = 'Time-dependent Brier Score'
        ylabel = 'Brier Score'
    
    plt.title(title)
    plt.xlabel('Evaluation Time Quantile')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    # Remove duplicate legends (handles multiple runs per dataset)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add vertical lines
    for i in range(9):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.2)
    
    # Use quantile labels on x-axis
    plt.xticks(range(9), quantile_labels)
    
    plt.tight_layout()
    plt.savefig(f'time_dependent_{metric_type}.png', bbox_inches='tight', dpi=300)
    plt.close()

# Create plots for both metrics
plot_time_dependent_metrics(csv_path, 'c_index')
plot_time_dependent_metrics(csv_path, 'brier')

print("\nPlots have been saved as 'time_dependent_c_index.png' and 'time_dependent_brier.png'")