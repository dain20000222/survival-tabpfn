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
import warnings
import matplotlib.pyplot as plt
import random
from sksurv.nonparametric import SurvivalFunctionEstimator 
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
os.makedirs("figures/figures/confusion_matrix_analysis", exist_ok=True)
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

bin_sizes = [3, 5, 7, 10, 15, 20, 25, 30]

def construct_tabpfn_trainset(x_train_imputed, y_train, cuts):
    """
    Construct TabPFN training dataset by creating examples for each patient at cut timepoints:
    - Examples at timepoints where patient was observed (class C or D)
    - Examples at timepoints before/after patient time (class A or B)

    Parameters:
        x_train_imputed (pd.DataFrame): Imputed covariates (n_patients × features)
        y_train (structured array): Survival data with fields "event" and "time" (discretized indices)
        cuts (np.ndarray): The actual cut values used for discretization

    Returns:
        X_tabpfn_train (pd.DataFrame): Feature matrix with timepoints
        y_tabpfn_train (pd.Series): Corresponding class labels ("A", "B", "C", "D")
    """
    # Extract event/censoring times and status
    T_train = y_train["time"]  # These are discretized indices
    delta_train = y_train["event"]
    n_train = len(T_train)

    # Use the actual cut values as evaluation timepoints
    selected_times = cuts.copy()

    dataset_rows = []
    class_labels = []

    # Create examples for each patient at each cut timepoint
    for i in range(n_train):
        x_i = x_train_imputed.iloc[i].values
        T_i_idx = T_train[i]  # Discretized time index
        delta_i = delta_train[i]

        # Create examples at each cut timepoint
        for j, t_j in enumerate(selected_times):
            # Assign class label based on T_i_idx and current timepoint index j
            if T_i_idx < j:
                label = "A"  # Event/censoring happened before this timepoint
            elif T_i_idx > j:
                label = "B"  # Event/censoring will happen after this timepoint
            elif T_i_idx == j and delta_i == 0:
                label = "C"  # Censored at this timepoint
            elif T_i_idx == j and delta_i == 1:
                label = "D"  # Event at this timepoint
            else:
                # Fallback (shouldn't happen with proper discretization)
                if T_i_idx < j:
                    label = "A"
                else:
                    label = "B"

            # Feature = original + eval_time (use actual cut value)
            row = np.concatenate([x_i, [t_j]])
            dataset_rows.append(row)
            class_labels.append(label)

    # Convert to arrays for easier manipulation
    dataset_rows = np.array(dataset_rows)
    class_labels = np.array(class_labels)
    
    # Create DataFrame and Series for return
    feature_cols = list(x_train_imputed.columns) + ["eval_time"]
    X_tabpfn_train = pd.DataFrame(dataset_rows, columns=feature_cols)
    y_tabpfn_train = pd.Series(class_labels)
    
    return X_tabpfn_train, y_tabpfn_train

def construct_tabpfn_testset(x_test_imputed, times):
    n_test = x_test_imputed.shape[0]
    d = x_test_imputed.shape[1]

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

def _is_monotonic_increasing(x: np.ndarray) -> bool:
    return (x[1:] >= x[:-1]).all()

def bin_numerical(x: np.ndarray, right_cuts: np.ndarray, error_on_larger: bool = False) -> np.ndarray:
    """
    pycox.utils.bin_numerical: index of bins defined by right_cuts (sorted).
    If right_cuts = [c0, c1, ... c_{m-1}], bins are (-inf, c0], (c0, c1], ... (c_{m-2}, c_{m-1}], (c_{m-1}, inf)
    """
    assert _is_monotonic_increasing(right_cuts), "Need `right_cuts` to be sorted."
    bins = np.searchsorted(right_cuts, x, side="left")
    if error_on_larger and bins.max() == right_cuts.size:
        raise ValueError("x contains larger values than right_cuts.")
    return bins

def discretize(x: np.ndarray, cuts: np.ndarray, side: str = "right", error_on_larger: bool = False) -> np.ndarray:
    """
    pycox.preprocessing.discretize: map times to cut values (not indices).
    side='right'  -> round up to right cut (events)
    side='left'   -> round down to left cut (censorings)
    """
    if side not in ["right", "left"]:
        raise ValueError("side must be 'right' or 'left'.")
    bins = bin_numerical(x, cuts, error_on_larger=error_on_larger)
    if side == "right":
        # append +inf so the last open interval maps to inf (pycox does the same then caps elsewhere)
        cuts_ext = np.concatenate([cuts, np.array([np.inf])])
        return cuts_ext[bins]
    # side == 'left'
    bins_cut = bins.copy()
    bins_cut[bins_cut == cuts.size] = -1
    exact = cuts[bins_cut] == x
    left_bins = bins - 1 + exact
    vals = cuts[left_bins]
    vals[left_bins == -1] = -np.inf
    return vals

def km_quantile_cuts(durations: np.ndarray, events: np.ndarray, num: int, min_=0., dtype="float64") -> np.ndarray:
    """
    pycox.preprocessing.cuts_quantiles replicated with sksurv:
    Build `num` right-cuts (sorted), with equal drop in KM survival between min and max observed time.
    Ensures cuts[0] = min_ (or durations.min() if min_=None) and cuts[-1] = durations.max().
    """
    sfe = SurvivalFunctionEstimator().fit(
        Surv.from_arrays(event=events.astype(bool), time=durations.astype(float))
    )
    t_sorted = np.sort(np.unique(durations.astype(float)))
    if t_sorted.size < 2:
        return np.array([t_sorted.min(), t_sorted.max()], dtype=dtype)

    S_hat = sfe.predict_proba(t_sorted)
    # pycox line: s_cuts = linspace(km.values.min(), km.values.max(), num)
    s_cuts = np.linspace(S_hat.min(), S_hat.max(), num)

    # pycox logic on reversed arrays
    cuts_idx = np.searchsorted(S_hat[::-1], s_cuts)[::-1]
    cuts = t_sorted[::-1][cuts_idx]
    cuts = np.unique(cuts)
    if cuts.size != num:
        warnings.warn(f"cuts are not unique, continue with {cuts.size} cuts instead of {num}")
    # first cut: durations.min() if min_ is None else min_
    cuts = cuts.astype(dtype)
    cuts[0] = durations.min() if min_ is None else min_
    # pycox asserts last cut equals durations.max()
    cuts[-1] = durations.max()
    return cuts

def discretize_unknown_c(duration: np.ndarray,
                         event: np.ndarray,
                         cuts: np.ndarray,
                         right_censor: bool = True,
                         censor_side: str = "left"):
    """
    pycox.preprocessing.DiscretizeUnknownC.transform:
    - Optionally right-censor values > cuts.max() and set event=False there
    - Events -> side='right'
    - Censorings -> side=censor_side (default 'left')
    Returns (td, event_adj) where `td` are cut-values (right/left edge), not indices.
    """
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
    """Map each cut value to its index (0..len(cuts)-1)."""
    uniq = np.unique(cuts)
    order = np.argsort(uniq)
    idx_map = {float(uniq[o]): int(o) for o in order}
    return np.vectorize(idx_map.get, otypes=[int])


def cdi_interpolate(eval_times, grid_times, S_grid):
    """
    Constant Density Interpolation (CDI) for survival functions.
    
    Assumes uniform distribution of event times within each interval,
    leading to S(t) = α_j - β_j * t within interval (τ_{j-1}, τ_j].
    
    Parameters:
        eval_times: array of times to evaluate S(t) at
        grid_times: array of grid points τ_j (sorted)
        S_grid: survival probabilities at grid points (n_patients × n_grid)
    
    Returns:
        S_eval: survival probabilities at eval_times (n_patients × n_eval)
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


def create_confusion_matrix(tabpfn_model, x_test_imputed, y_test, cuts, dataset_name, n_bins):
    """Create confusion matrix using discretized time values."""
    
    # First discretize test times using same process as training
    t_test = y_test["time"]
    e_test = y_test["event"].astype(bool)
    td_test, e_test_adj = discretize_unknown_c(t_test, e_test, cuts, right_censor=True, censor_side="left")
    
    # Map to indices
    to_idx = duration_to_index_map(cuts)
    t_test_bin = to_idx(td_test)
    
    # Get test instances using existing function
    X_test, patient_ids = construct_tabpfn_testset(x_test_imputed, cuts)
    
    # Get true labels using discretized times
    y_true = []
    for i, pid in enumerate(patient_ids):
        T_i_idx = t_test_bin[pid]  # Use discretized time index
        delta_i = e_test_adj[pid]  # Use adjusted event indicator
        t_j = X_test.iloc[i]["eval_time"]
        t_j_idx = to_idx(t_j)  # Convert evaluation time to index
        
        if T_i_idx < t_j_idx:
            label = 0  # A
        elif T_i_idx > t_j_idx:
            label = 1  # B
        elif T_i_idx == t_j_idx and not delta_i:
            label = 2  # C
        else:  # T_i_idx == t_j_idx and delta_i
            label = 3  # D
            
        y_true.append(label)
    
    y_true = np.array(y_true)
    
    # Get predictions from TabPFN
    probs = tabpfn_model.predict_proba(X_test)
    y_pred = np.argmax(probs, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Calculate precision and recall for each class
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3], average=None, zero_division=0)
    accuracy = np.mean(y_true == y_pred)
    
    # Create figure with two equal-sized subplots
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot: Confusion matrix with both counts and percentages
    im1 = ax_left.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax_left.set_title(f'Confusion Matrix\nm={n_bins}, Acc: {accuracy:.3f}', fontsize=18)
    
    class_names = ["A", "B", "C", "D"]
    # Add text annotations to confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i,j]}\n({cm_norm[i,j]:.2f})"
            ax_left.text(j, i, text, ha="center", va="center", fontsize=14,
                        color="white" if cm_norm[i,j] > 0.5 else "black")
    
    ax_left.set_xticks(range(4))
    ax_left.set_yticks(range(4))
    ax_left.set_xticklabels(class_names, fontsize=16)
    ax_left.set_yticklabels(class_names, fontsize=16)
    ax_left.set_xlabel('Predicted', fontsize=16)
    ax_left.set_ylabel('True', fontsize=16)
    
    # Right subplot: Average probabilities by predicted class
    avg_probs_by_pred = []
    for pred_class in range(4):
        mask = y_pred == pred_class
        if np.sum(mask) > 0:
            avg_probs = np.mean(probs[mask], axis=0)
        else:
            avg_probs = np.zeros(4)
        avg_probs_by_pred.append(avg_probs)
    
    avg_probs_by_pred = np.array(avg_probs_by_pred)
    
    im2 = ax_right.imshow(avg_probs_by_pred, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax_right.set_title('Average Probabilities\nby Predicted Class', fontsize=18)
    
    # Add text annotations to probability matrix
    for i in range(4):
        for j in range(4):
            text = f"{avg_probs_by_pred[i,j]:.3f}"
            ax_right.text(j, i, text, ha="center", va="center", fontsize=14,
                         color="white" if avg_probs_by_pred[i,j] > 0.5 else "black")
    
    ax_right.set_xticks(range(4))
    ax_right.set_yticks(range(4))
    ax_right.set_xticklabels(class_names, fontsize=16)
    ax_right.set_yticklabels(class_names, fontsize=16)
    ax_right.set_xlabel('Probability for Class', fontsize=16)
    ax_right.set_ylabel('Predicted Class', fontsize=16)
    
    plt.colorbar(im2, ax=ax_right)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'figures/confusion_matrix_analysis/{dataset_name}_nbins_{n_bins}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return {
        'confusion_matrix': cm,
        'normalized_confusion_matrix': cm_norm,
        'average_probabilities': avg_probs_by_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


def plot_accuracy_vs_nbins(dataset_name, accuracies_dict):
    """Plot classification accuracy vs number of bins."""
    n_bins_values = sorted(accuracies_dict.keys())
    accuracies = [accuracies_dict[n] for n in n_bins_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_bins_values, accuracies, 'o-', linewidth=2, markersize=8, color='purple')
    plt.xlabel('Number of Grid Points (m)', fontsize=18)
    plt.ylabel('Classification Accuracy', fontsize=18)
    plt.title(f'{dataset_name}: TabPFN Classification Accuracy vs m', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figures/confusion_matrix_analysis/{dataset_name}_accuracy_vs_m.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_vs_nbins(dataset_name, precision_dict):
    """Plot per-class precision vs number of bins."""
    n_bins_values = sorted(precision_dict.keys())
    class_names = ['A', 'B', 'C', 'D']
    class_colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    plt.figure(figsize=(12, 6))
    
    for i, (class_name, color) in enumerate(zip(class_names, class_colors)):
        precisions = [precision_dict[n][i] for n in n_bins_values]
        plt.plot(n_bins_values, precisions, 'o-', linewidth=2, markersize=8, 
                label=f'Class {class_name}', color=color)
    
    plt.xlabel('Number of Grid Points (m)', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title(f'{dataset_name}: TabPFN Per-Class Precision vs m', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'figures/confusion_matrix_analysis/{dataset_name}_precision_vs_m.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_recall_vs_nbins(dataset_name, recall_dict):
    """Plot per-class recall vs number of bins."""
    n_bins_values = sorted(recall_dict.keys())
    class_names = ['A', 'B', 'C', 'D']
    class_colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    plt.figure(figsize=(12, 6))
    
    for i, (class_name, color) in enumerate(zip(class_names, class_colors)):
        recalls = [recall_dict[n][i] for n in n_bins_values]
        plt.plot(n_bins_values, recalls, 'o-', linewidth=2, markersize=8, 
                label=f'Class {class_name}', color=color)
    
    plt.xlabel('Number of Grid Points (m)', fontsize=18)
    plt.ylabel('Recall', fontsize=18)
    plt.title(f'{dataset_name}: TabPFN Per-Class Recall vs m', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'figures/confusion_matrix_analysis/{dataset_name}_recall_vs_m.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_macro_precision_recall_vs_nbins(dataset_name, precision_dict, recall_dict):
    """Plot macro-averaged precision and recall vs number of bins."""
    n_bins_values = sorted(precision_dict.keys())
    
    # Calculate macro averages
    macro_precisions = [np.mean(precision_dict[n]) for n in n_bins_values]
    macro_recalls = [np.mean(recall_dict[n]) for n in n_bins_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_bins_values, macro_precisions, 'o-', linewidth=2, markersize=8, 
             color='blue', label='Macro Precision')
    plt.plot(n_bins_values, macro_recalls, 's-', linewidth=2, markersize=8, 
             color='red', label='Macro Recall')
    
    plt.xlabel('Number of Grid Points (m)', fontsize=18)
    plt.ylabel('Score', fontsize=18)
    plt.title(f'{dataset_name}: TabPFN Macro Precision & Recall vs m', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'figures/confusion_matrix_analysis/{dataset_name}_macro_precision_recall_vs_m.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution(dataset_name, class_distributions):
    """Plot training set class distribution for different n_bins values."""
    n_bins_values = sorted(class_distributions.keys())
    
    # Prepare data for plotting
    class_names = ['A', 'B', 'C', 'D']
    class_colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    # Get percentages for each class
    percentages = {}
    for n in n_bins_values:
        dist = class_distributions[n]
        total = sum(dist.values())
        percentages[n] = {cls: (dist[cls]/total)*100 for cls in class_names}
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(n_bins_values))
    width = 0.2

    # Plot bars for each class
    for i, (cls, color) in enumerate(zip(class_names, class_colors)):
        values = [percentages[n][cls] for n in n_bins_values]
        plt.bar(x + i*width - width*1.5, values, width, label=f'Class {cls}', color=color)

    plt.xlabel('Number of Grid Points (m)', fontsize=18)
    plt.ylabel('Percentage (%)', fontsize=16)
    plt.title(f'{dataset_name}: Training Set Class Distribution vs m', fontsize=18)
    plt.xticks(x, n_bins_values, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'figures/confusion_matrix_analysis/{dataset_name}_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

for file_name in dataset_files:
    try:
        dataset_name = file_name.replace(".csv", "")
        file_path = os.path.join(data_dir, file_name)
        print("="*50)
        print(f"\nProcessing dataset: {dataset_name}")

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
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.15, stratify=y["event"], random_state=SEED
        )
        # x_train, x_val, y_train, y_val = train_test_split(
        #     x_trainval, y_trainval, test_size=0.1765, stratify=y_trainval["event"], random_state=SEED
        # )
        # 0.1765 * 0.85 ≈ 0.15, so test/val are both 15%

        # One-hot encode using get_dummies (fit on TRAIN only!)
        x_train_ohe = pd.get_dummies(x_train, drop_first=True)
        x_test_ohe  = pd.get_dummies(x_test, drop_first=True)

        # Align columns so train/val/test match
        x_train_ohe, x_test_ohe = x_train_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)

        covariates_ohe = x_train_ohe.columns  # all columns are covariates after OHE

        # Impute missing values in covariates (fit on train only, same as baseline.py)
        imputer = SimpleImputer().fit(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = imputer.transform(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = pd.DataFrame(x_train_imputed, columns=covariates_ohe, index=x_train.index)
        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # Evaluation time points (use train+val, same as baseline.py)
        times = np.percentile(y_train["time"], np.arange(10, 100, 10))
        times = np.unique(times)
        max_trainval_time = y_train["time"].max()
        times = times[times < max_trainval_time]
        test_mask = y_test["time"] < max_trainval_time
        y_test_filtered = y_test[test_mask]
        x_test_filtered = x_test_imputed[test_mask]
        times = times[(times > y_test_filtered["time"].min()) & (times < y_test_filtered["time"].max())]
        print(f"Evaluation time points: {times}")

        # Keep the masked test index for alignment inside the n_bins loop
        masked_test_index = x_test_filtered.index

        # Keep originals for metrics (explicit names for clarity)
        y_train_orig = y_train
        y_test_orig = y_test_filtered

        accuracies = {}
        precisions = {}
        recalls = {}
        class_distributions = {}

        for n_bins in bin_sizes:
            try:
                print(f"\nUsing KM-based discretization with {n_bins} cuts.")
                
                # Build KM-quantile right-cuts using training data
                cuts = km_quantile_cuts(
                    durations=df.loc[x_train.index, time_col].to_numpy(),
                    events=df.loc[x_train.index, event_col].to_numpy(),
                    num=n_bins,
                    min_=df[time_col].min(),
                    dtype="float64"
                )

                # Discretize TRAIN
                t_train = df.loc[x_train.index, time_col].to_numpy()
                e_train = df.loc[x_train.index, event_col].astype(bool).to_numpy()
                td_train, e_train_adj = discretize_unknown_c(t_train, e_train, cuts, right_censor=True, censor_side="left")

                # Map discrete cut-values -> integer indices
                to_idx = duration_to_index_map(cuts)
                t_train_bin = to_idx(td_train)

                # Build Surv objects
                y_train_model = Surv.from_arrays(event=e_train_adj, time=t_train_bin)
                
                # Train TabPFN
                X_tabpfn_train, y_tabpfn_train = construct_tabpfn_trainset(
                    x_train_imputed, y_train_model, cuts
                )

                # Add subsampling for large datasets when n_bins >= 15
                if n_bins >= 15 and len(X_tabpfn_train) > 30000:
                    print(f"Subsampled training set to 30000 rows (from {len(X_tabpfn_train)})")
                    subsample_idx = np.random.choice(len(X_tabpfn_train), size=30000, replace=False)
                    X_tabpfn_train = X_tabpfn_train.iloc[subsample_idx]
                    y_tabpfn_train = y_tabpfn_train.iloc[subsample_idx]

                # Store class distribution after subsampling
                class_dist = y_tabpfn_train.value_counts().to_dict()
                class_distributions[n_bins] = class_dist

                # Train TabPFN model
                tabpfn_model = TabPFNClassifier(
                    device='cuda',
                    ignore_pretraining_limits=True
                )
                tabpfn_model.fit(X_tabpfn_train, y_tabpfn_train)

                # Create and save confusion matrix
                confusion_metrics = create_confusion_matrix(
                    tabpfn_model, x_test_filtered, y_test_filtered,
                    cuts, dataset_name, n_bins
                )

                # Store accuracy, precision, and recall
                accuracies[n_bins] = confusion_metrics['accuracy']
                precisions[n_bins] = confusion_metrics['precision']
                recalls[n_bins] = confusion_metrics['recall']

            except Exception as e:
                print(f"Error with n_bins={n_bins}: {str(e)}")
                continue

        # Create accuracy plot after all n_bins are processed
        plot_accuracy_vs_nbins(dataset_name, accuracies)
        plot_precision_vs_nbins(dataset_name, precisions)
        plot_recall_vs_nbins(dataset_name, recalls)
        plot_macro_precision_recall_vs_nbins(dataset_name, precisions, recalls)
        plot_class_distribution(dataset_name, class_distributions)

    except Exception as e:
        print(f"Error: {e}")
        continue