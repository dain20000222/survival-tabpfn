"""
TabPFN n_bins Analysis: Effect of Diswarnings.filterwarnings("ignore")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create figures directoryGranularity on Performance

This script investigates how the number of discretization bins (n_bins) affects TabPFN's 
performance across calibration and discrimination metrics on the top 5 worst IBS datasets.

Metrics analyzed:
1. Time-specific Brier Scores (calibration)
2. C-index (discrimination) 
3. Mean AUC (discrimination)

Hypothesis: Finer discretization (more bins) should improve calibration (lower Brier scores)
but may not significantly impact discrimination metrics (C-index, AUC) which are based on
relative ordering rather than absolute probability values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import random
import torch

# Set matplotlib font sizes
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from sklearn.pipeline import make_pipeline
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import integrated_brier_score, brier_score, concordance_index_censored, cumulative_dynamic_auc
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
from pycox.models import DeepHitSingle, CoxPH
from tabpfn import TabPFNClassifier
from sksurv.nonparametric import SurvivalFunctionEstimator
from sklearn.calibration import calibration_curve
from scipy.stats import binom
import torchtuples as tt
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

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

# Create figures directory
os.makedirs("figures/nbins_analysis", exist_ok=True)

# TabPFN utility functions (same as analysis.py)
def _is_monotonic_increasing(x: np.ndarray) -> bool:
    return (x[1:] >= x[:-1]).all()

def bin_numerical(x: np.ndarray, right_cuts: np.ndarray, error_on_larger: bool = False) -> np.ndarray:
    assert _is_monotonic_increasing(right_cuts), "Need `right_cuts` to be sorted."
    bins = np.searchsorted(right_cuts, x, side="left")
    if error_on_larger and bins.max() == right_cuts.size:
        raise ValueError("x contains larger values than right_cuts.")
    return bins

def discretize(x: np.ndarray, cuts: np.ndarray, side: str = "right", error_on_larger: bool = False) -> np.ndarray:
    if side not in ["right", "left"]:
        raise ValueError("side must be 'right' or 'left'.")
    bins = bin_numerical(x, cuts, error_on_larger=error_on_larger)
    if side == "right":
        cuts_ext = np.concatenate([cuts, np.array([np.inf])])
        return cuts_ext[bins]
    bins_cut = bins.copy()
    bins_cut[bins_cut == cuts.size] = -1
    exact = cuts[bins_cut] == x
    left_bins = bins - 1 + exact
    vals = cuts[left_bins]
    vals[left_bins == -1] = -np.inf
    return vals

def km_quantile_cuts(durations: np.ndarray, events: np.ndarray, num: int, min_=0., dtype="float64") -> np.ndarray:
    sfe = SurvivalFunctionEstimator().fit(
        Surv.from_arrays(event=events.astype(bool), time=durations.astype(float))
    )
    t_sorted = np.sort(np.unique(durations.astype(float)))
    if t_sorted.size < 2:
        return np.array([t_sorted.min(), t_sorted.max()], dtype=dtype)

    S_hat = sfe.predict_proba(t_sorted)
    s_cuts = np.linspace(S_hat.min(), S_hat.max(), num)
    cuts_idx = np.searchsorted(S_hat[::-1], s_cuts)[::-1]
    cuts = t_sorted[::-1][cuts_idx]
    cuts = np.unique(cuts)
    if cuts.size != num:
        warnings.warn(f"cuts are not unique, continue with {cuts.size} cuts instead of {num}")
    cuts = cuts.astype(dtype)
    cuts[0] = durations.min() if min_ is None else min_
    cuts[-1] = durations.max()
    return cuts

def discretize_unknown_c(duration: np.ndarray, event: np.ndarray, cuts: np.ndarray,
                         right_censor: bool = True, censor_side: str = "left"):
    dtype_event = event.dtype
    event = event.astype(bool).copy()
    duration = duration.astype(float).copy()

    if right_censor:
        censor_mask = duration > cuts.max()
        if censor_mask.any():
            duration[censor_mask] = cuts.max()
            event[censor_mask] = False

    if duration.max() > cuts.max():
        raise ValueError("`duration` contains larger values than cuts. Set right_censor=True to cap these.")

    td = np.zeros_like(duration, dtype=float)
    cens_mask = ~event
    td[event] = discretize(duration[event], cuts, side="right", error_on_larger=True)
    if cens_mask.any():
        td[cens_mask] = discretize(duration[cens_mask], cuts, side=censor_side, error_on_larger=True)
    return td, event.astype(dtype_event)

def duration_to_index_map(cuts: np.ndarray):
    uniq = np.unique(cuts)
    order = np.argsort(uniq)
    idx_map = {float(uniq[o]): int(o) for o in order}
    return np.vectorize(idx_map.get, otypes=[int])

def cdi_interpolate(eval_times, grid_times, S_grid):
    """
    Constant Density Interpolation (CDI) for survival functions.
    """
    n_patients, n_grid = S_grid.shape
    n_eval = len(eval_times)
    S_eval = np.zeros((n_patients, n_eval))
    
    for i in range(n_patients):
        S_i = S_grid[i]
        
        for j, t in enumerate(eval_times):
            # Find interval: t ∈ (τ_{k-1}, τ_k]
            k = np.searchsorted(grid_times, t, side='right')
            
            if k == 0:
                # t <= τ_0, extrapolate as constant
                S_eval[i, j] = 1.0  # or S_i[0]
            elif k >= n_grid:
                # t > τ_{max}, extrapolate as constant
                S_eval[i, j] = S_i[-1]
            else:
                # t ∈ (τ_{k-1}, τ_k], apply CDI formula
                tau_prev = grid_times[k-1] if k > 0 else 0.0
                tau_curr = grid_times[k]
                S_prev = S_i[k-1] if k > 0 else 1.0
                S_curr = S_i[k]
                
                # CDI coefficients: S(t) = α_j - β_j * t
                delta_tau = tau_curr - tau_prev
                if delta_tau > 0:
                    alpha_j = (S_prev * tau_curr - S_curr * tau_prev) / delta_tau
                    beta_j = (S_prev - S_curr) / delta_tau
                    S_eval[i, j] = alpha_j - beta_j * t
                else:
                    # Degenerate interval, use endpoint value
                    S_eval[i, j] = S_curr
    
    # Ensure monotonicity and bounds
    S_eval = np.clip(S_eval, 0.0, 1.0)
    return S_eval

def construct_tabpfn_trainset(x_train_imputed, y_train, cuts):
    T_train = y_train["time"]
    delta_train = y_train["event"]
    n_train = len(T_train)

    selected_times = cuts.copy()
    dataset_rows = []
    class_labels = []

    for i in range(n_train):
        x_i = x_train_imputed.iloc[i].values
        T_i_idx = T_train[i]
        delta_i = delta_train[i]

        for j, t_j in enumerate(selected_times):
            if T_i_idx < j:
                label = "A"
            elif T_i_idx > j:
                label = "B"
            elif T_i_idx == j and delta_i == 0:
                label = "C"
            elif T_i_idx == j and delta_i == 1:
                label = "D"
            else:
                if T_i_idx < j:
                    label = "A"
                else:
                    label = "B"

            row = np.concatenate([x_i, [t_j]])
            dataset_rows.append(row)
            class_labels.append(label)

    dataset_rows = np.array(dataset_rows)
    class_labels = np.array(class_labels)
    
    feature_cols = list(x_train_imputed.columns) + ["eval_time"]
    X_tabpfn_train = pd.DataFrame(dataset_rows, columns=feature_cols)
    y_tabpfn_train = pd.Series(class_labels)
    
    return X_tabpfn_train, y_tabpfn_train

def construct_tabpfn_testset(x_test_imputed, times):
    n_test = x_test_imputed.shape[0]
    test_rows = []
    patient_ids = []

    for i in range(n_test):
        x_i = x_test_imputed.iloc[i].values
        for t in times:
            row = np.concatenate([x_i, [t]])
            test_rows.append(row)
            patient_ids.append(i)

    feature_cols = list(x_test_imputed.columns) + ["eval_time"]
    X_tabpfn_test = pd.DataFrame(test_rows, columns=feature_cols)
    return X_tabpfn_test, np.array(patient_ids)

def get_worst_datasets():
    """Get the top 5 datasets where TabPFN performs worst on IBS."""
    tabpfn_df = pd.read_csv("tabpfn_evaluation.csv")
    baseline_df = pd.read_csv("baseline_evaluation.csv")
    
    # Merge datasets
    merged = pd.merge(tabpfn_df, baseline_df, left_on='dataset', right_on='dataset_name', how='inner')
    
    # Find best baseline IBS for each dataset
    baseline_ibs_cols = ['rsf_ibs', 'cph_ibs', 'dh_ibs', 'ds_ibs']
    merged['best_baseline_ibs'] = merged[baseline_ibs_cols].min(axis=1)
    merged['ibs_difference'] = merged['ibs'] - merged['best_baseline_ibs']
    
    # Sort by largest IBS difference (worst relative performance)
    worst_datasets = merged.sort_values('ibs_difference', ascending=False).head(5)
    
    print("Top 5 datasets where TabPFN has worst IBS relative to best baseline:")
    print(worst_datasets[['dataset', 'ibs', 'best_baseline_ibs', 'ibs_difference']].round(4))
    
    return worst_datasets['dataset'].tolist()

def train_tabpfn_with_nbins(x_trainval_imputed, y_trainval, x_test_filtered, y_test_filtered, 
                           df, time_col, event_col, times, n_bins):
    """Train TabPFN with specified n_bins and return survival predictions and metrics."""
    
    # Discretization
    cuts = km_quantile_cuts(
        durations=df.loc[x_trainval_imputed.index, time_col].to_numpy(),
        events=df.loc[x_trainval_imputed.index, event_col].to_numpy(),
        num=n_bins,
        min_=df[time_col].min(),
        dtype="float64",
    )
    
    # Discretize times
    t_trainval = df.loc[x_trainval_imputed.index, time_col].to_numpy()
    e_trainval = df.loc[x_trainval_imputed.index, event_col].astype(bool).to_numpy()
    td_trainval, e_trainval_adj = discretize_unknown_c(t_trainval, e_trainval, cuts, right_censor=True, censor_side="left")
    
    t_te_filtered = df.loc[x_test_filtered.index, time_col].to_numpy()
    e_te_filtered = df.loc[x_test_filtered.index, event_col].astype(bool).to_numpy()
    td_te_filtered, e_te_adj_filtered = discretize_unknown_c(t_te_filtered, e_te_filtered, cuts, right_censor=True, censor_side="left")
    
    # Map to indices
    to_idx = duration_to_index_map(cuts)
    t_trainval_bin = to_idx(td_trainval)
    t_test_bin_filtered = to_idx(td_te_filtered)
    
    # Build Surv objects
    y_trainval_model = Surv.from_arrays(event=e_trainval_adj, time=t_trainval_bin)
    y_test_model = Surv.from_arrays(event=e_te_adj_filtered, time=t_test_bin_filtered)
    
    # Map evaluation times
    times_eval = bin_numerical(times, cuts, error_on_larger=False)
    times_eval = np.clip(times_eval, 0, len(cuts) - 1).astype(int)
    
    # Train TabPFN
    X_tabpfn_train, y_tabpfn_train = construct_tabpfn_trainset(x_trainval_imputed, y_trainval_model, cuts)
    
    tabpfn_model = TabPFNClassifier(
        device= "cuda" if torch.cuda.is_available() else "cpu",
        ignore_pretraining_limits=True
    )
    tabpfn_model.fit(X_tabpfn_train, y_tabpfn_train)
    
    # Predict with TabPFN
    unique_bins, inv = np.unique(times_eval, return_inverse=True)
    X_tabpfn_test, test_patient_ids = construct_tabpfn_testset(x_test_filtered, cuts[unique_bins])
    probs = tabpfn_model.predict_proba(X_tabpfn_test)
    
    # Convert to hazards and survival
    def hazards_from_probs(X_test, pids, probs, unique_cut_values, eps=1e-12):
        n_patients = pids.max() + 1
        m = len(unique_cut_values)
        h = np.zeros((n_patients, m), dtype=float)
        for j, t in enumerate(unique_cut_values):
            idx_at_t = np.where(np.isclose(X_test["eval_time"], t))[0]
            for idx in idx_at_t:
                pid = pids[idx]
                pA, pB, pC, pD = probs[idx]
                denom = pB + pC + pD + eps   # at-risk at t
                h[pid, j] = pD / denom
        return np.clip(h, 0.0, 1.0)
    
    h_unique = hazards_from_probs(X_tabpfn_test, test_patient_ids, probs, cuts[unique_bins])
    S_unique = np.clip(np.cumprod(1.0 - h_unique, axis=1), 0.0, 1.0)
    
    # Interpolate to evaluation times using CDI
    bin_t_grid = cuts[unique_bins]
    S_tabpfn = cdi_interpolate(times, bin_t_grid, S_unique)
    S_tabpfn = np.minimum.accumulate(S_tabpfn, axis=1)
    
    # Compute metrics
    metrics = {}
    
    # 1. Integrated Brier Score
    ibs = integrated_brier_score(y_trainval, y_test_filtered, S_tabpfn, times)
    metrics['ibs'] = ibs
    
    # 2. Time-specific Brier Scores
    brier_scores = []
    for i, t in enumerate(times):
        try:
            bs = brier_score(y_trainval, y_test_filtered, S_tabpfn[:, i], t)[1]
            brier_scores.append(bs)
        except:
            brier_scores.append(np.nan)
    metrics['brier_scores'] = brier_scores
    
    # 3. C-index
    try:
        # Use risk scores: probability of event by horizon (same as surv_tabpfn.py)
        risk_scores = 1.0 - S_tabpfn[:, -1]
        c_index = concordance_index_censored(y_test_filtered['event'], y_test_filtered['time'], risk_scores)[0]
        metrics['c_index'] = c_index
    except:
        metrics['c_index'] = np.nan
    
    # 4. Mean AUC
    try:
        # Compute time-dependent mean AUC (same as surv_tabpfn.py)
        risk_scores = 1.0 - S_tabpfn[:, -1]
        _, mean_auc = cumulative_dynamic_auc(y_trainval, y_test_filtered, risk_scores, times=times)
        metrics['mean_auc'] = mean_auc
    except:
        metrics['mean_auc'] = np.nan
    
    return S_tabpfn, metrics

def analyze_nbins_effect(dataset_name, n_bins_range=[3, 5, 10, 15, 20, 25, 30]):
    """Analyze the effect of different n_bins values on a single dataset."""
    print(f"\nAnalyzing n_bins effect on dataset: {dataset_name}")
    
    # Load dataset
    file_path = os.path.join("test", f"{dataset_name}.csv")
    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    df.drop(columns=['pid'], inplace=True, errors='ignore')
    
    # Prepare data
    time_col = "time"
    event_col = "event"
    covariates = df.columns.difference([time_col, event_col])
    
    x = df[covariates].copy()
    y = Surv.from_arrays(event=df[event_col].astype(bool), time=df[time_col])
    
    # Train-test split
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.15, stratify=y["event"], random_state=SEED
    )
    
    # One-hot encoding and imputation
    x_trainval_ohe = pd.get_dummies(x_trainval, drop_first=True)
    x_test_ohe = pd.get_dummies(x_test, drop_first=True)
    x_trainval_ohe, x_test_ohe = x_trainval_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)
    
    covariates_ohe = x_trainval_ohe.columns
    imputer = SimpleImputer().fit(x_trainval_ohe)
    x_trainval_imputed = pd.DataFrame(
        imputer.transform(x_trainval_ohe), 
        columns=covariates_ohe, 
        index=x_trainval.index
    )
    x_test_imputed = pd.DataFrame(
        imputer.transform(x_test_ohe), 
        columns=covariates_ohe, 
        index=x_test.index
    )
    
    # Evaluation times
    times = np.percentile(y_trainval["time"], np.arange(10, 100, 10))
    times = np.unique(times)
    max_trainval_time = y_trainval["time"].max()
    times = times[times < max_trainval_time]
    test_mask = y_test["time"] < max_trainval_time
    y_test_filtered = y_test[test_mask]
    x_test_filtered = x_test_imputed[test_mask]
    times = times[(times > y_test_filtered["time"].min()) & (times < y_test_filtered["time"].max())]
    
    print(f"Evaluation times: {times}")
    print(f"Testing n_bins: {n_bins_range}")
    
    # Store results for each n_bins
    results = {}
    
    for n_bins in n_bins_range:
        print(f"  Training with n_bins={n_bins}...")
        try:
            S_tabpfn, metrics = train_tabpfn_with_nbins(
                x_trainval_imputed, y_trainval, x_test_filtered, y_test_filtered,
                df, time_col, event_col, times, n_bins
            )
            results[n_bins] = metrics
            print(f"    IBS: {metrics['ibs']:.4f}, C-index: {metrics['c_index']:.4f}, Mean AUC: {metrics['mean_auc']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    
    if not results:
        print(f"No successful runs for {dataset_name}")
        return None
    
    # Plot results
    plot_nbins_analysis(dataset_name, results, times, n_bins_range)
    
    return results

def plot_nbins_analysis(dataset_name, results, times, n_bins_range):
    """Plot the effect of n_bins on various metrics."""
    
    # Prepare data for plotting
    n_bins_values = list(results.keys())
    n_bins_values.sort()
    
    # Extract metrics
    ibs_values = [results[n]['ibs'] for n in n_bins_values]
    c_index_values = [results[n]['c_index'] for n in n_bins_values]
    mean_auc_values = [results[n]['mean_auc'] for n in n_bins_values]
    
    # Create a separate figure with only the 3 main metrics (IBS, C-index, Mean AUC)
    fig_metrics, axes_metrics = plt.subplots(1, 3, figsize=(18, 6))
    
    # IBS plot
    axes_metrics[0].plot(n_bins_values, ibs_values, 'o-', linewidth=2, markersize=8, color='red')
    axes_metrics[0].set_xlabel('Number of Bins (m)', fontsize=14)
    axes_metrics[0].set_ylabel('Integrated Brier Score (IBS)', fontsize=14)
    axes_metrics[0].set_title('IBS vs m', fontsize=16)
    axes_metrics[0].grid(True, alpha=0.3)
    
    # C-index plot
    axes_metrics[1].plot(n_bins_values, c_index_values, 'o-', linewidth=2, markersize=8, color='blue')
    axes_metrics[1].set_xlabel('Number of Bins (m)', fontsize=14)
    axes_metrics[1].set_ylabel('C-index', fontsize=14)
    axes_metrics[1].set_title('C-index vs m', fontsize=16)
    axes_metrics[1].grid(True, alpha=0.3)
    
    # Mean AUC plot
    axes_metrics[2].plot(n_bins_values, mean_auc_values, 'o-', linewidth=2, markersize=8, color='green')
    axes_metrics[2].set_xlabel('Number of Bins (m)', fontsize=14)
    axes_metrics[2].set_ylabel('Mean AUC', fontsize=14)
    axes_metrics[2].set_title('Mean AUC vs m', fontsize=16)
    axes_metrics[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figures/nbins_analysis/{dataset_name}_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed time-specific Brier score plot
    plt.figure(figsize=(10,6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_bins_values)))
    for i, n_bins in enumerate(n_bins_values):
        brier_scores = results[n_bins]['brier_scores']
        plt.plot(times, brier_scores, 'o-', label=f'm={n_bins}', 
                color=colors[i], linewidth=2, markersize=6)
    
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Brier Score', fontsize=14)
    plt.title(f'{dataset_name}: Time-specific Brier Scores for Different m', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/nbins_analysis/{dataset_name}_time_specific_brier_nbins.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_plots(all_results, dataset_names, n_bins_range):
    """Create summary plots across all datasets."""
    
    # Summary metrics vs n_bins across all datasets
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))
    
    for i, dataset_name in enumerate(dataset_names):
        if dataset_name not in all_results:
            continue
            
        results = all_results[dataset_name]
        n_bins_values = list(results.keys())
        n_bins_values.sort()
        
        ibs_values = [results[n]['ibs'] for n in n_bins_values]
        c_index_values = [results[n]['c_index'] for n in n_bins_values]
        mean_auc_values = [results[n]['mean_auc'] for n in n_bins_values]
        
        # IBS
        axes[0].plot(n_bins_values, ibs_values, 'o-', label=dataset_name, 
                    color=colors[i], linewidth=2, markersize=6)
        
        # C-index
        axes[1].plot(n_bins_values, c_index_values, 'o-', label=dataset_name, 
                    color=colors[i], linewidth=2, markersize=6)
        
        # Mean AUC
        axes[2].plot(n_bins_values, mean_auc_values, 'o-', label=dataset_name, 
                    color=colors[i], linewidth=2, markersize=6)
    
    axes[0].set_xlabel('Number of Bins (m)', fontsize=14)
    axes[0].set_ylabel('Integrated Brier Score (IBS)', fontsize=14)
    axes[0].set_title('IBS vs m (All Datasets)', fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Number of Bins (m)', fontsize=14)
    axes[1].set_ylabel('C-index', fontsize=14)
    axes[1].set_title('C-index vs m (All Datasets)', fontsize=16)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Number of Bins (m)', fontsize=14)
    axes[2].set_ylabel('Mean AUC', fontsize=14)
    axes[2].set_title('Mean AUC vs m (All Datasets)', fontsize=16)
    axes[2].legend(fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figures/nbins_analysis/summary_nbins_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Get top 5 worst IBS datasets
    worst_datasets = get_worst_datasets()
    
    # Range of n_bins to test
    n_bins_range = [3, 5, 10, 15, 20, 25, 30]
    
    print(f"\n{'='*60}")
    print("n_bins Analysis on Top 5 Worst IBS Datasets")
    print(f"{'='*60}")
    print(f"Testing n_bins: {n_bins_range}")
    print(f"Datasets: {worst_datasets}")
    
    # Analyze each dataset
    all_results = {}
    
    for dataset_name in worst_datasets:
        print(f"\n{'='*60}")
        print(f"Analyzing: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            results = analyze_nbins_effect(dataset_name, n_bins_range)
            if results:
                all_results[dataset_name] = results
                print(f"✅ Analysis completed for {dataset_name}")
            else:
                print(f"❌ Analysis failed for {dataset_name}")
        except Exception as e:
            print(f"❌ Analysis failed for {dataset_name}: {e}")
            continue
    
    # Create summary plots
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Creating Summary Plots")
        print(f"{'='*60}")
        create_summary_plots(all_results, list(all_results.keys()), n_bins_range)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY: n_bins Effect Analysis")
    print(f"{'='*60}")
    
    if not all_results:
        print("No successful analyses completed.")
    else:
        for dataset_name, results in all_results.items():
            print(f"\n{dataset_name}:")
            n_bins_values = list(results.keys())
            n_bins_values.sort()
            
            print("  n_bins | IBS    | C-index | Mean AUC")
            print("  -------|--------|---------|----------")
            for n_bins in n_bins_values:
                metrics = results[n_bins]
                print(f"  {n_bins:6d} | {metrics['ibs']:.4f} | {metrics['c_index']:.4f}  | {metrics['mean_auc']:.4f}")
            
            # Find best n_bins for each metric
            best_ibs_nbins = min(n_bins_values, key=lambda n: results[n]['ibs'])
            best_cindex_nbins = max(n_bins_values, key=lambda n: results[n]['c_index'])
            best_auc_nbins = max(n_bins_values, key=lambda n: results[n]['mean_auc'])
            
            print(f"  Best IBS: n_bins={best_ibs_nbins} ({results[best_ibs_nbins]['ibs']:.4f})")
            print(f"  Best C-index: n_bins={best_cindex_nbins} ({results[best_cindex_nbins]['c_index']:.4f})")
            print(f"  Best Mean AUC: n_bins={best_auc_nbins} ({results[best_auc_nbins]['mean_auc']:.4f})")
    
    print(f"\n{'='*60}")
    print("Generated figures:")
    for dataset_name in all_results.keys():
        print(f"  figures/nbins_analysis/{dataset_name}_metrics_comparison.png (3 main metrics)")
        print(f"  figures/nbins_analysis/{dataset_name}_time_specific_brier_nbins.png (time-specific Brier)")
    if len(all_results) > 1:
        print(f"  figures/nbins_analysis/summary_nbins_analysis.png")
    
    print(f"\n{'='*60}")
    print("Key Insights:")
    print("- Calibration (IBS, Brier scores): Should improve with more bins (finer discretization)")
    print("- Discrimination (C-index, AUC): Should remain relatively stable across n_bins")
    print("- Optimal n_bins may differ between calibration and discrimination metrics")
    print("- Trade-off between calibration improvement and computational complexity")
    print(f"{'='*60}")