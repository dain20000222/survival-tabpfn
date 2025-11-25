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

def create_pid_mappings(predictions_df, observed_times, observed_events):
    """Create mappings from pid to observed data"""
    pids = predictions_df['pid'].unique()
    pid_to_time = dict(zip(pids, observed_times))
    pid_to_event = dict(zip(pids, observed_events))
    return pid_to_time, pid_to_event

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

def brier_score_at_time(predictions_df, pid_to_time, pid_to_event, kmf_censoring, t, tau_k, dataset=None):
    """
    Calculate Brier Score at time t given tau_k
    
    BS(t|tau_k) = 1/n * sum_i { [0 - S_{i,tau_k}(t)]^2 * I(tau_k < T_i ≤ t, δ_i = 1) / G(T_i) +
                               [1 - S_{i,tau_k}(t)]^2 * I(T_i > t) / G(t) }
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
    
    bs_sum = 0
    n = 0
    
    for _, row in pred_at_tau.iterrows():
        pid = row['pid']
        T_i = pid_to_time[pid]
        delta_i = pid_to_event[pid]
        
        # Get survival probability at time t (assuming it's available or can be interpolated)
        # For simplicity, using surv_prob from the row (this assumes eval_time == t)
        S_it = row['surv_prob']
        
        G_Ti = get_censoring_prob(kmf_censoring, T_i)
        G_t = get_censoring_prob(kmf_censoring, t)
        
        if tau_k < T_i <= t and delta_i == 1:
            # Patient had event in (tau_k, t]
            if G_Ti > 0:
                bs_sum += (0 - S_it)**2 / G_Ti
        elif T_i > t:
            # Patient survived beyond t
            if G_t > 0:
                bs_sum += (1 - S_it)**2 / G_t
        
        n += 1
    
    return bs_sum / n if n > 0 else np.nan

def dynamic_ibs(predictions_df, pid_to_time, pid_to_event, observed_times, tau_k, tau_m=None, dataset=None):
    """
    Calculate dynamic Integrated Brier Score
    
    IBS(tau_k) = ∫_{tau_k}^{tau_m} BS(t|tau_k) dw(t)
    
    Using uniform weight w(t) = 1/(tau_m - tau_k)
    """
    if tau_m is None:
        tau_m = np.max(observed_times)
    
    if tau_k >= tau_m:
        return np.nan
    
    # Fit censoring distribution
    observed_events = np.array([pid_to_event[pid] for pid in predictions_df['pid'].unique()])
    kmf_censoring = fit_censoring_distribution(observed_times, observed_events)
    
    # Get unique time points for integration
    time_points = np.linspace(tau_k, tau_m, num=50)  # Adjust num for precision
    
    def bs_function(t):
        return brier_score_at_time(predictions_df, pid_to_time, pid_to_event, kmf_censoring, t, tau_k, dataset)
    
    # Numerical integration
    try:
        ibs_value, _ = integrate.quad(bs_function, tau_k, tau_m)
        return ibs_value / (tau_m - tau_k)  # Normalize by interval length
    except:
        # Fallback to trapezoidal rule
        bs_values = [brier_score_at_time(predictions_df, pid_to_time, pid_to_event, kmf_censoring, t, tau_k, dataset) 
                    for t in time_points]
        bs_values = [bs for bs in bs_values if not np.isnan(bs)]
        if len(bs_values) > 1:
            return np.trapz(bs_values, time_points[:len(bs_values)]) / (tau_m - tau_k)
        else:
            return np.nan

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
            print(f"Loaded {len(pids)} patients from {dataset}")
            
            # Create mappings
            pid_to_time, pid_to_event = create_pid_mappings(df, observed_times, observed_events)
            
            dataset_times = sorted(df[df['dataset'] == dataset]['eval_time'].unique())
            
            for i, tau_k in enumerate(dataset_times):
                result = {'dataset': dataset, 'tau_k': tau_k}
                
                # Dynamic C-index
                c_index = dynamic_c_index(df, pid_to_time, pid_to_event, tau_k, dataset)
                result['c_index'] = c_index
                
                # Dynamic AUC (if not the first time point)
                if i > 0:
                    tau_k_minus_1 = dataset_times[i-1]
                    auc = dynamic_auc(df, pid_to_time, pid_to_event, tau_k, tau_k_minus_1, dataset)
                    result['auc'] = auc
                else:
                    result['auc'] = np.nan
                
                # Dynamic IBS
                ibs = dynamic_ibs(df, pid_to_time, pid_to_event, observed_times, tau_k, dataset=dataset)
                result['ibs'] = ibs
                
                # Dynamic D-calibration (example: for survival prob in (0.3, 0.7])
                dcal = dynamic_d_calibration(df, pid_to_event, tau_k, 0.3, 0.7, dataset)
                result['dcal_0.3_0.7'] = dcal
                
                results.append(result)
                
                print(f"  τ_k = {tau_k:.2f}: C-index = {c_index:.4f}, AUC = {result['auc']:.4f}, "
                      f"IBS = {ibs:.4f}, D-cal(0.3,0.7] = {dcal:.4f}")
        
        except Exception as e:
            print(f"Error loading dataset {dataset}: {e}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('dynamic_metrics_results.csv', index=False)
    print(f"\nResults saved to 'dynamic_metrics_results.csv'")
    
    return results_df

if __name__ == "__main__":
    results = main()