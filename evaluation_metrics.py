import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from scipy import integrate
import os
import warnings
warnings.filterwarnings('ignore')

def load_original_survival_data(dataset_name, data_dir="data"):
    """Load original survival data for a specific dataset"""
    file_path = os.path.join(data_dir, f"{dataset_name}.csv")
    original_df = pd.read_csv(file_path)

    # Get the last observation for each patient (final survival outcome)
    time_col = 'time2' if 'time2' in original_df.columns else 'time'

    # Group by patient and get the last observation
    patient_data = original_df.groupby('pid').agg({
        time_col: 'max',  # Last observed time
        'event': 'max'    # Event indicator (1 if event occurred, 0 if censored)
    }).reset_index()

    return patient_data['pid'].values, patient_data[time_col].values, patient_data['event'].values

def fit_censoring_distribution(observed_times, observed_events):
    """Fit Kaplan-Meier for censoring distribution G(t)"""
    kmf_censoring = KaplanMeierFitter()
    # For censoring distribution, we reverse events (treat censoring as event)
    kmf_censoring.fit(observed_times, 1 - observed_events)
    return kmf_censoring

def get_censoring_prob(kmf_censoring, t):
    """Get censoring survival probability G(t)"""
    if t <= 0:
        return 1.0
    if t >= kmf_censoring.timeline[-1]:
        return kmf_censoring.survival_function_at_times(kmf_censoring.timeline[-1]).iloc[0]
    return kmf_censoring.survival_function_at_times(t).iloc[0]

def dynamic_c_index(predictions_df, pid_to_time, pid_to_event, tau_k, dataset=None):
    """
    Calculate dynamic C-index at time tau_k
    
    Formula: C(tau_k) = sum_{i≠j} δ_i * I(tau_k < T_i < T_j) * I(r_i > r_j) / 
                       sum_{i≠j} δ_i * I(tau_k < T_i < T_j)
    """
    # Filter predictions at tau_k
    if dataset:
        pred_at_tau = predictions_df[
            (predictions_df['eval_time'] == tau_k) & 
            (predictions_df['dataset'] == dataset)
        ]
    else:
        pred_at_tau = predictions_df[predictions_df['eval_time'] == tau_k]
    
    if len(pred_at_tau) == 0:
        return np.nan
    
    numerator = 0
    denominator = 0
    
    for _, row_i in pred_at_tau.iterrows():
        pid_i = row_i['pid']
        T_i = pid_to_time[pid_i]
        delta_i = pid_to_event[pid_i]
        r_i = row_i['risk']
        
        if delta_i == 0 or T_i <= tau_k:  # Skip if censored or event before tau_k
            continue
            
        for _, row_j in pred_at_tau.iterrows():
            pid_j = row_j['pid']
            if pid_i == pid_j:
                continue
                
            T_j = pid_to_time[pid_j]
            r_j = row_j['risk']
            
            if tau_k < T_i < T_j:  # Condition: tau_k < T_i < T_j
                denominator += 1
                if r_i > r_j:  # Concordant pair
                    numerator += 1
    
    return numerator / denominator if denominator > 0 else np.nan

def dynamic_auc(predictions_df, pid_to_time, pid_to_event, tau_k, tau_k_minus_1, dataset=None):
    """
    Calculate dynamic AUC between tau_{k-1} and tau_k
    
    Formula: AUC(tau_k, tau_{k-1}) = 
            sum_{i≠j} δ_i * I(T_i ∈ (tau_{k-1}, tau_k], T_j > tau_k) * I(r_{i,tau_{k-1}} > r_{j,tau_{k-1}}) /
            sum_{i≠j} δ_i * I(T_i ∈ (tau_{k-1}, tau_k], T_j > tau_k)
    """
    # Get predictions at tau_{k-1}
    if dataset:
        pred_at_tau = predictions_df[
            (predictions_df['eval_time'] == tau_k_minus_1) & 
            (predictions_df['dataset'] == dataset)
        ]
    else:
        pred_at_tau = predictions_df[predictions_df['eval_time'] == tau_k_minus_1]
    
    if len(pred_at_tau) == 0:
        return np.nan
    
    numerator = 0
    denominator = 0
    
    for _, row_i in pred_at_tau.iterrows():
        pid_i = row_i['pid']
        T_i = pid_to_time[pid_i]
        delta_i = pid_to_event[pid_i]
        r_i = row_i['risk']
        
        # Check if T_i ∈ (tau_{k-1}, tau_k] and event occurred
        if delta_i == 0 or T_i <= tau_k_minus_1 or T_i > tau_k:
            continue
            
        for _, row_j in pred_at_tau.iterrows():
            pid_j = row_j['pid']
            if pid_i == pid_j:
                continue
                
            T_j = pid_to_time[pid_j]
            r_j = row_j['risk']
            
            if T_j > tau_k:  # T_j > tau_k
                denominator += 1
                if r_i > r_j:  # Concordant pair
                    numerator += 1
    
    return numerator / denominator if denominator > 0 else np.nan


def dynamic_brier_score(predictions_df, pid_to_time, pid_to_event, kmf_censoring, tau_k, tau_k_minus_1, dataset=None):
    """
    Calculate dynamic Brier Score between tau_{k-1} and tau_k
    
    BS(tau_k, tau_{k-1}) = 1/n * sum_i { [0 - S_{i,tau_{k-1}}(tau_k)]^2 * I(tau_{k-1} < T_i ≤ tau_k, δ_i = 1) / G(T_i) +
                                        [1 - S_{i,tau_{k-1}}(tau_k)]^2 * I(T_i > tau_k) / G(tau_k) }
    """
    # Get predictions at tau_{k-1} (we use survival probabilities from tau_{k-1} to predict tau_k)
    if dataset:
        pred_at_tau_minus_1 = predictions_df[
            (predictions_df['eval_time'] == tau_k_minus_1) & 
            (predictions_df['dataset'] == dataset)
        ]
    else:
        pred_at_tau_minus_1 = predictions_df[predictions_df['eval_time'] == tau_k_minus_1]
    
    if len(pred_at_tau_minus_1) == 0:
        return np.nan
    
    bs_sum = 0
    n = 0
    
    G_tau_k = get_censoring_prob(kmf_censoring, tau_k)
    
    for _, row in pred_at_tau_minus_1.iterrows():
        pid = row['pid']
        if pid not in pid_to_time:
            continue
            
        T_i = pid_to_time[pid]
        delta_i = pid_to_event[pid]
        
        # S_{i,tau_{k-1}}(tau_k) - survival probability from tau_{k-1} to tau_k
        # This should be the survival probability at tau_k given information up to tau_{k-1}
        S_i_tau_k = row['surv_prob']  # This represents survival up to eval_time (tau_{k-1})
        
        # We need to extrapolate/estimate survival to tau_k
        # For simplicity, assuming the surv_prob represents survival to the next evaluation time
        # In practice, you might need to interpolate or use a different approach
        
        if tau_k_minus_1 < T_i <= tau_k and delta_i == 1:
            # Patient had event in (tau_{k-1}, tau_k]
            G_Ti = get_censoring_prob(kmf_censoring, T_i)
            if G_Ti > 0:
                bs_sum += (0 - S_i_tau_k)**2 / G_Ti
        elif T_i > tau_k:
            # Patient survived beyond tau_k
            if G_tau_k > 0:
                bs_sum += (1 - S_i_tau_k)**2 / G_tau_k
        
        n += 1
    
    return bs_sum / n if n > 0 else np.nan

def dynamic_d_calibration(predictions_df, pid_to_event, tau_k, a, b, dataset=None):
    """
    Calculate dynamic D-calibration for interval (a, b]
    
    DCal_{(a,b]}(tau_k) = 1/n * sum_i δ_i * I(S_{i,tau_k} ∈ (a,b])
    """
    # Get predictions at tau_k
    if dataset:
        pred_at_tau = predictions_df[
            (predictions_df['eval_time'] == tau_k) & 
            (predictions_df['dataset'] == dataset)
        ]
    else:
        pred_at_tau = predictions_df[predictions_df['eval_time'] == tau_k]
    
    if len(pred_at_tau) == 0:
        return np.nan
    
    count = 0
    total = 0
    
    for _, row in pred_at_tau.iterrows():
        pid = row['pid']
        delta_i = pid_to_event[pid]
        S_i = row['surv_prob']
        
        total += 1
        if delta_i == 1 and a < S_i <= b:
            count += 1
    
    return count / total if total > 0 else np.nan

def create_pid_mappings(predictions_df, pids, observed_times, observed_events):
    """Create mappings from pid to observed data, filtering to only include patients in predictions"""
    # Get unique patient IDs that actually appear in predictions
    prediction_pids = set(predictions_df['pid'].unique())
    
    # Filter to only include patients that appear in both datasets
    pid_to_time = {}
    pid_to_event = {}
    
    for i, pid in enumerate(pids):
        if pid in prediction_pids:
            pid_to_time[pid] = observed_times[i]
            pid_to_event[pid] = observed_events[i]
    
    print(f"Mapped {len(pid_to_time)} patients (intersection of predictions and original data)")
    return pid_to_time, pid_to_event

def main():
    # Load the predictions CSV file
    df = pd.read_csv('dynamic_tabpfn_risks.csv')
    
    # Calculate metrics for each dataset
    datasets = df['dataset'].unique()
    results = []
    
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        
        # Load actual observed survival data for this dataset
        try:
            pids, observed_times, observed_events = load_original_survival_data(dataset)
            print(f"Loaded {len(pids)} patients from original {dataset}")
            
            # Get patients that appear in predictions for this dataset
            dataset_predictions = df[df['dataset'] == dataset]
            prediction_pids = dataset_predictions['pid'].unique()
            print(f"Found {len(prediction_pids)} patients in predictions for {dataset}")
            
            # Create mappings (only for patients that appear in both datasets)
            pid_to_time, pid_to_event = create_pid_mappings(dataset_predictions, pids, observed_times, observed_events)
            
            # Check if we have any valid mappings
            if len(pid_to_time) == 0:
                print(f"No matching patients found between predictions and original data for {dataset}")
                continue
            
            # Fit censoring distribution for this dataset
            valid_pids = list(pid_to_time.keys())
            observed_events_valid = np.array([pid_to_event[pid] for pid in valid_pids])
            observed_times_valid = np.array([pid_to_time[pid] for pid in valid_pids])
            kmf_censoring = fit_censoring_distribution(observed_times_valid, observed_events_valid)
            
            dataset_times = sorted(dataset_predictions['eval_time'].unique())
            
            for i, tau_k in enumerate(dataset_times):
                result = {'dataset': dataset, 'tau_k': tau_k}
                
                # Dynamic C-index
                c_index = dynamic_c_index(df, pid_to_time, pid_to_event, tau_k, dataset)
                result['c_index'] = c_index
                
                # Dynamic AUC and Brier Score (if not the first time point)
                if i > 0:
                    tau_k_minus_1 = dataset_times[i-1]
                    auc = dynamic_auc(df, pid_to_time, pid_to_event, tau_k, tau_k_minus_1, dataset)
                    result['auc'] = auc
                    
                    # Dynamic Brier Score between tau_{k-1} and tau_k
                    bs = dynamic_brier_score(df, pid_to_time, pid_to_event, kmf_censoring, tau_k, tau_k_minus_1, dataset)
                    result['brier_score'] = bs
                else:
                    result['auc'] = np.nan
                    result['brier_score'] = np.nan
                
                # Dynamic D-calibration
                dcal = dynamic_d_calibration(df, pid_to_event, tau_k, 0.3, 0.7, dataset)
                result['dcal_0.3_0.7'] = dcal
                
                results.append(result)
                
                print(f"  τ_k = {tau_k:.2f}: C-index = {c_index:.4f}, AUC = {result['auc']:.4f}, "
                      f"BS = {result['brier_score']:.4f}, D-cal(0.3,0.7] = {dcal:.4f}")
        
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('dynamic_tabpfn_results.csv', index=False)
    print(f"\nResults saved to 'dynamic_tabpfn_results.csv'")

    return results_df

if __name__ == "__main__":
    results = main()