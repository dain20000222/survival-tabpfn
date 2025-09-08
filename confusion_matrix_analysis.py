"""
TabPFN Confusion Matrix Analysis: Effect of n_bins on Classification Performance

This script investigates how the number of discretization bins (n_bins) affects TabPFN's 
classification performance using confusion matrix analysis on the top 5 worst IBS datasets.

The analysis focuses on TabPFN's ability to classify survival scenarios at ground truth times T_i:
- Class A (Early): T_i < times[0] (event before first evaluation time)
- Class B (Late): T_i > times[-1] (event after last evaluation time)  
- Class C (Censored): T_i ≈ actual_time and delta_i == 0 (censored at evaluation time)
- Class D (Event): T_i ≈ actual_time and delta_i == 1 (event at evaluation time)

This provides insight into TabPFN's intrinsic understanding of survival patterns and how
discretization granularity affects its fundamental classification ability.
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import random
import torch

# Set CUDA_LAUNCH_BLOCKING for better CUDA error debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from tabpfn import TabPFNClassifier
from sksurv.nonparametric import SurvivalFunctionEstimator
import traceback

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
os.makedirs("figures/confusion_matrix_analysis", exist_ok=True)

def print_gpu_memory_info():
    """Print current GPU memory usage for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"      GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def check_and_subsample_if_needed(X_tabpfn_train, y_tabpfn_train, max_samples=50000):
    """
    Check if TabPFN training set is too large and subsample if needed.
    TabPFN can struggle with very large datasets, especially on GPU.
    """
    n_samples = X_tabpfn_train.shape[0]
    
    if n_samples > max_samples:
        print(f"      Dataset too large ({n_samples} samples), subsampling to {max_samples}")
        # Stratified subsampling to maintain class distribution
        from sklearn.model_selection import train_test_split
        X_sub, _, y_sub, _ = train_test_split(
            X_tabpfn_train, y_tabpfn_train, 
            train_size=max_samples, 
            stratify=y_tabpfn_train,
            random_state=SEED
        )
        return X_sub, y_sub
    
    return X_tabpfn_train, y_tabpfn_train

# TabPFN utility functions
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

def get_analysis_datasets():
    """Get the specified datasets for confusion matrix analysis."""
    datasets = ['pharmacoSmoking', 'grace', 'rhc']
    
    print("Selected datasets for confusion matrix analysis:")
    print(f"  - {datasets}")
    
    # Verify datasets exist (optional check)
    missing_datasets = []
    for dataset in datasets:
        file_path = os.path.join("test", f"{dataset}.csv")
        if not os.path.exists(file_path):
            missing_datasets.append(dataset)
    
    if missing_datasets:
        print(f"Warning: The following datasets were not found: {missing_datasets}")
        available_datasets = [d for d in datasets if d not in missing_datasets]
        print(f"Will proceed with available datasets: {available_datasets}")
        return available_datasets
    
    return datasets

def train_tabpfn_for_confusion_analysis(x_trainval_imputed, y_trainval, df, time_col, event_col, n_bins):
    """Train TabPFN with specified n_bins and return the trained model, cuts, and training class distribution."""
    
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
    
    # Map to indices
    to_idx = duration_to_index_map(cuts)
    t_trainval_bin = to_idx(td_trainval)
    
    # Build Surv objects
    y_trainval_model = Surv.from_arrays(event=e_trainval_adj, time=t_trainval_bin)
    
    # Train TabPFN
    X_tabpfn_train, y_tabpfn_train = construct_tabpfn_trainset(x_trainval_imputed, y_trainval_model, cuts)
    
    print(f"      TabPFN training set size: {X_tabpfn_train.shape[0]} samples, {X_tabpfn_train.shape[1]} features")
    
    # Check and subsample if dataset is too large
    X_tabpfn_train, y_tabpfn_train = check_and_subsample_if_needed(X_tabpfn_train, y_tabpfn_train)
    
    # Calculate training class distribution
    train_class_counts = y_tabpfn_train.value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0).values
    
    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory_info()
    
    # Try CUDA first, stop if CUDA fails (don't fallback to CPU as it won't be effective)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. TabPFN requires GPU acceleration for effective training.")
    
    try:
        tabpfn_model = TabPFNClassifier(
            device=device,
            ignore_pretraining_limits=True
        )
        tabpfn_model.fit(X_tabpfn_train, y_tabpfn_train)
        print(f"      Successfully trained on {device.upper()}")
        if torch.cuda.is_available():
            print_gpu_memory_info()
    except Exception as cuda_error:
        print(f"      CUDA training failed: {str(cuda_error)}")
        print("      Cannot fallback to CPU as TabPFN requires GPU acceleration.")
        # Clear CUDA cache and raise the error to stop this n_bins iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print_gpu_memory_info()
        raise cuda_error
    
    return tabpfn_model, cuts, train_class_counts



def confusion_matrix_analysis(tabpfn_model, x_test_imputed, y_test, times, cuts, dataset_name, n_bins):
    """
    Perform confusion matrix analysis at ground truth T_i for each patient.
    
    Class definitions:
    - A: T_i < times[0] (event before first evaluation time)
    - B: T_i > times[-1] (event after last evaluation time)  
    - C: T_i == actual_time and delta_i == 0 (censored at evaluation time)
    - D: T_i == actual_time and delta_i == 1 (event at evaluation time)
    """
    
    print(f"\n{'='*50}")
    print(f"Confusion Matrix Analysis at Ground Truth T_i (n_bins={n_bins})")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")
    
    # Create (x, T_i) test instances
    X_gt = []
    y_true = []
    
    for i in range(len(x_test_imputed)):
        x_i = x_test_imputed.iloc[i].values
        T_i = y_test["time"][i]
        delta_i = y_test["event"][i]
        
        # Label logic based on ground truth time T_i
        if T_i < times[0]:
            label = 0  # A: event before first evaluation time
        elif T_i > times[-1]:
            label = 1  # B: event after last evaluation time
        else:
            # Find the closest time point
            closest_idx = np.argmin(np.abs(times - T_i))
            closest_time = times[closest_idx]
            
            if np.isclose(T_i, closest_time, rtol=1e-3) and delta_i == 0:
                label = 2  # C: censored at evaluation time
            elif np.isclose(T_i, closest_time, rtol=1e-3) and delta_i == 1:
                label = 3  # D: event at evaluation time
            else:
                # Use temporal ordering as fallback
                if T_i < closest_time:
                    label = 0  # A
                else:
                    label = 1  # B
        
        y_true.append(label)
        row = np.concatenate([x_i, [T_i]])
        X_gt.append(row)
    
    X_gt = pd.DataFrame(X_gt, columns=list(x_test_imputed.columns) + ["eval_time"])
    y_true = np.array(y_true)
    
    # Clear CUDA cache before prediction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory_info()
    
    # Predict using TabPFN
    try:
        probs_gt = tabpfn_model.predict_proba(X_gt)
        if torch.cuda.is_available():
            print_gpu_memory_info()
    except Exception as pred_error:
        print(f"      Prediction failed: {str(pred_error)}")
        if torch.cuda.is_available():
            print("      Clearing CUDA cache and retrying prediction...")
            torch.cuda.empty_cache()
            print_gpu_memory_info()
        probs_gt = tabpfn_model.predict_proba(X_gt)
    
    y_pred = np.argmax(probs_gt, axis=1)
    
    print("Ground truth class counts:", np.bincount(y_true, minlength=4))
    print("Predicted class counts:", np.bincount(y_pred, minlength=4))
    
    # Detailed Probability Analysis
    print(f"\n{'='*60}")
    print("DETAILED PROBABILITY ANALYSIS")
    print(f"{'='*60}")
    
    class_names = ["A", "B", "C", "D"]
    
    # Overall probability statistics for each class
    print("\nOverall Probability Statistics by Class:")
    print("Class | Mean±Std    | Median | Min   | Max   | Q25   | Q75")
    print("------|-------------|--------|-------|-------|-------|-------")
    for i, name in enumerate(class_names):
        probs_class = probs_gt[:, i]
        mean_prob = np.mean(probs_class)
        std_prob = np.std(probs_class)
        median_prob = np.median(probs_class)
        min_prob = np.min(probs_class)
        max_prob = np.max(probs_class)
        q25_prob = np.percentile(probs_class, 25)
        q75_prob = np.percentile(probs_class, 75)
        
        print(f"  {name}   | {mean_prob:.3f}±{std_prob:.3f} | {median_prob:.3f}  | {min_prob:.3f} | {max_prob:.3f} | {q25_prob:.3f} | {q75_prob:.3f}")
    
    # Prediction confidence analysis
    max_probs = np.max(probs_gt, axis=1)
    print(f"\nPrediction Confidence Analysis:")
    print(f"  Mean maximum probability: {np.mean(max_probs):.3f}±{np.std(max_probs):.3f}")
    print(f"  Median maximum probability: {np.median(max_probs):.3f}")
    print(f"  Min/Max maximum probability: {np.min(max_probs):.3f}/{np.max(max_probs):.3f}")
    
    # Confidence by prediction correctness
    correct_mask = (y_true == y_pred)
    print(f"  Correct predictions confidence: {np.mean(max_probs[correct_mask]):.3f}±{np.std(max_probs[correct_mask]):.3f}")
    print(f"  Incorrect predictions confidence: {np.mean(max_probs[~correct_mask]):.3f}±{np.std(max_probs[~correct_mask]):.3f}")
    
    # High/Low confidence thresholds
    high_conf_threshold = 0.8
    low_conf_threshold = 0.4
    high_conf_mask = max_probs >= high_conf_threshold
    low_conf_mask = max_probs <= low_conf_threshold
    
    print(f"\nConfidence Categories:")
    print(f"  High confidence (≥{high_conf_threshold}): {np.sum(high_conf_mask)} samples ({np.sum(high_conf_mask)/len(max_probs)*100:.1f}%)")
    print(f"  Low confidence (≤{low_conf_threshold}): {np.sum(low_conf_mask)} samples ({np.sum(low_conf_mask)/len(max_probs)*100:.1f}%)")
    
    if np.sum(high_conf_mask) > 0:
        print(f"    High confidence accuracy: {np.mean(correct_mask[high_conf_mask]):.3f}")
    if np.sum(low_conf_mask) > 0:
        print(f"    Low confidence accuracy: {np.mean(correct_mask[low_conf_mask]):.3f}")
    
    # Probability distribution by true class
    print(f"\nProbability Distribution by True Class:")
    for true_class in range(4):
        if np.sum(y_true == true_class) == 0:
            continue
        mask = y_true == true_class
        print(f"\n  True Class {class_names[true_class]} ({np.sum(mask)} samples):")
        print("    Predicted | Mean Prob | Std  | Samples | Percentage")
        print("    ----------|-----------|------|---------|----------")
        for pred_class in range(4):
            class_probs = probs_gt[mask, pred_class]
            mean_prob = np.mean(class_probs)
            std_prob = np.std(class_probs)
            assigned_samples = np.sum((y_true == true_class) & (y_pred == pred_class))
            pct = assigned_samples / np.sum(mask) * 100
            print(f"    {class_names[pred_class]:8s}  | {mean_prob:.3f}     | {std_prob:.3f} | {assigned_samples:7d} | {pct:8.1f}%")
    
    print(f"\n{'='*60}")
    
    # Print class distribution percentages
    total_samples = len(y_true)
    class_names = ["A", "B", "C", "D"]
    print(f"\nGround truth class distribution:")
    for i, name in enumerate(class_names):
        count = np.sum(y_true == i)
        pct = count / total_samples * 100
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    print(f"\nPredicted class distribution:")
    for i, name in enumerate(class_names):
        count = np.sum(y_pred == i)
        pct = count / total_samples * 100
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    # Calculate accuracy and per-class metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    try:
        # Get unique labels present in both y_true and y_pred
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        target_names_subset = [class_names[i] for i in unique_labels]
        
        print(classification_report(y_true, y_pred, labels=unique_labels, 
                                  target_names=target_names_subset, zero_division=0))
    except Exception as e:
        print(f"Classification report failed: {e}")
        # Fallback: just print basic metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=[0,1,2,3]
        )
        print("Per-class metrics:")
        for i, name in enumerate(class_names):
            print(f"  {name}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")
    
    # Return metrics for analysis
    max_probs = np.max(probs_gt, axis=1)
    correct_mask = (y_true == y_pred)
    
    # Calculate entropy
    epsilon = 1e-15
    probs_safe = np.clip(probs_gt, epsilon, 1-epsilon)
    entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1)
    
    # Calculate probability differences (confidence margin)
    sorted_probs = np.sort(probs_gt, axis=1)
    prob_diff = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    confusion_metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'class_distribution_true': np.bincount(y_true, minlength=4),
        'class_distribution_pred': np.bincount(y_pred, minlength=4),
        'probs': probs_gt,
        'max_probs': max_probs,
        'mean_confidence': np.mean(max_probs),
        'std_confidence': np.std(max_probs),
        'mean_confidence_correct': np.mean(max_probs[correct_mask]) if np.sum(correct_mask) > 0 else 0,
        'mean_confidence_incorrect': np.mean(max_probs[~correct_mask]) if np.sum(~correct_mask) > 0 else 0,
        'mean_entropy': np.mean(entropy),
        'std_entropy': np.std(entropy),
        'mean_prob_diff': np.mean(prob_diff),
        'std_prob_diff': np.std(prob_diff),
        'class_prob_means': np.mean(probs_gt, axis=0),
        'class_prob_stds': np.std(probs_gt, axis=0)
    }
    
    return confusion_metrics

def analyze_confusion_matrix_nbins(dataset_name, n_bins_range=[3, 5, 10, 15, 20, 25, 30]):
    """Analyze confusion matrix for different n_bins values on a single dataset."""
    print(f"\nAnalyzing confusion matrix for n_bins effect on dataset: {dataset_name}")
    
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
    print(f"Test set size after filtering: {len(y_test_filtered)}")
    
    # Check if test set is too small
    if len(y_test_filtered) < 10:
        print(f"Warning: Test set is very small ({len(y_test_filtered)} samples). Results may be unreliable.")
    
    # Store results for each n_bins
    confusion_results = {}
    
    for n_bins in n_bins_range:
        print(f"  Training with n_bins={n_bins}...")
        
        # Clear CUDA cache before each iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Train TabPFN model
            tabpfn_model, cuts, train_class_counts = train_tabpfn_for_confusion_analysis(
                x_trainval_imputed, y_trainval, df, time_col, event_col, n_bins
            )
            
            # Perform confusion matrix analysis
            print(f"  Performing confusion matrix analysis for n_bins={n_bins}...")
            confusion_metrics = confusion_matrix_analysis(
                tabpfn_model, x_test_filtered, y_test_filtered, times, cuts, dataset_name, n_bins
            )
            # Add training class distribution to metrics
            confusion_metrics['class_distribution_train'] = train_class_counts
            confusion_results[n_bins] = confusion_metrics
            
        except Exception as e:
            print(f"    Failed: {e}")
            # Clear CUDA cache after failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            traceback.print_exc()
            continue
    
    if not confusion_results:
        print(f"No successful runs for {dataset_name}")
        return None
    
    # Plot confusion matrix comparison
    plot_confusion_matrix_comparison(dataset_name, confusion_results, n_bins_range)
    
    # Plot training set class distribution
    plot_training_class_distribution(dataset_name, confusion_results, n_bins_range)
    
    return confusion_results

def plot_training_class_distribution(dataset_name, confusion_results, n_bins_range):
    """Plot training set class distribution for different n_bins."""
    
    if not confusion_results:
        print(f"No confusion matrix results to plot training distribution for {dataset_name}")
        return
    
    n_bins_values = list(confusion_results.keys())
    n_bins_values.sort()
    
    class_names = ["A", "B", "C", "D"]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    # Plot training class distributions
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(n_bins_values))
    width = 0.2
    
    for i, class_name in enumerate(class_names):
        class_pcts = []
        for n_bins in n_bins_values:
            # Use the actual training class distribution
            train_counts = confusion_results[n_bins]['class_distribution_train']
            train_pcts = train_counts / train_counts.sum() * 100
            class_pcts.append(train_pcts[i])
        
        ax.bar(x + i*width, class_pcts, width, label=f'Class {class_name}', color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Number of Bins (n_bins)', fontsize=14)
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_title(f'{dataset_name}: Training Set Class Distribution vs n_bins', fontsize=16)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(n_bins_values)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figures/confusion_matrix_analysis/{dataset_name}_training_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix_comparison(dataset_name, confusion_results, n_bins_range):
    """Plot comparison of confusion matrices across different n_bins."""
    
    if not confusion_results:
        print(f"No confusion matrix results to plot for {dataset_name}")
        return
    
    n_bins_values = list(confusion_results.keys())
    n_bins_values.sort()
    
    # Plot 1: Accuracy vs n_bins
    plt.figure(figsize=(10, 6))
    accuracies = [confusion_results[n]['accuracy'] for n in n_bins_values]
    plt.plot(n_bins_values, accuracies, 'o-', linewidth=2, markersize=8, color='purple')
    plt.xlabel('Number of Bins (n_bins)', fontsize=14)
    plt.ylabel('Classification Accuracy', fontsize=14)
    plt.title(f'{dataset_name}: TabPFN Classification Accuracy vs n_bins', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/confusion_matrix_analysis/{dataset_name}_accuracy_vs_nbins.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Individual confusion matrix + average probabilities for each n_bins
    class_names = ["A", "B", "C", "D"]
    
    for n_bins in n_bins_values:
        cm = confusion_results[n_bins]['confusion_matrix']
        confusion_data = confusion_results[n_bins]
        probs_gt = confusion_data['probs']
        y_pred = confusion_data['y_pred']
        
        # Create figure with two equal-sized subplots
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left subplot: Confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im1 = ax_left.imshow(cm_norm, interpolation='nearest', cmap='Blues')
        ax_left.set_title(f'Confusion Matrix\nn_bins={n_bins}, Acc: {confusion_results[n_bins]["accuracy"]:.3f}', fontsize=12)
        
        # Add text annotations to confusion matrix
        thresh = cm_norm.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax_left.text(k, j, f'{cm[j, k]}\n({cm_norm[j, k]:.2f})',
                           ha="center", va="center",
                           color="white" if cm_norm[j, k] > thresh else "black",
                           fontsize=9)
        
        ax_left.set_xticks(range(4))
        ax_left.set_yticks(range(4))
        ax_left.set_xticklabels(class_names, rotation=45)
        ax_left.set_yticklabels(class_names)
        ax_left.set_xlabel('Predicted')
        ax_left.set_ylabel('True')
        
        # Right subplot: Average probabilities by predicted class
        avg_probs_by_pred = []
        for pred_class in range(4):
            mask = y_pred == pred_class
            if np.sum(mask) > 0:
                avg_probs = np.mean(probs_gt[mask], axis=0)
            else:
                avg_probs = np.zeros(4)
            avg_probs_by_pred.append(avg_probs)
        
        avg_probs_by_pred = np.array(avg_probs_by_pred)
        
        im2 = ax_right.imshow(avg_probs_by_pred, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations to probability matrix
        for j in range(4):
            for k in range(4):
                text = ax_right.text(k, j, f'{avg_probs_by_pred[j, k]:.3f}',
                                   ha="center", va="center", 
                                   color="white" if avg_probs_by_pred[j, k] > 0.5 else "black",
                                   fontsize=9)
        
        ax_right.set_title(f'Average Probabilities\nby Predicted Class', fontsize=12)
        ax_right.set_xlabel('Probability for Class')
        ax_right.set_ylabel('Predicted Class')
        ax_right.set_xticks(range(4))
        ax_right.set_yticks(range(4))
        ax_right.set_xticklabels(class_names)
        ax_right.set_yticklabels(class_names)
        
        # Add colorbar for probability matrix
        plt.colorbar(im2, ax=ax_right, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'{dataset_name}: Confusion Matrix and Average Probabilities (n_bins={n_bins})', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'figures/confusion_matrix_analysis/{dataset_name}_confusion_and_probs_nbins{n_bins}.png', dpi=300, bbox_inches='tight')
        plt.show()



if __name__ == "__main__":
    # Get specified datasets for analysis
    analysis_datasets = get_analysis_datasets()
    
    # Range of n_bins to test
    n_bins_range = [3, 5, 10, 15, 20, 25, 30]
    
    print(f"\n{'='*60}")
    print("Confusion Matrix Analysis: n_bins Effect on TabPFN Classification")
    print(f"{'='*60}")
    print(f"Testing n_bins: {n_bins_range}")
    print(f"Datasets: {analysis_datasets}")
    
    # Analyze each dataset
    all_confusion_results = {}
    
    for dataset_name in analysis_datasets:
        print(f"\n{'='*60}")
        print(f"Analyzing: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            confusion_results = analyze_confusion_matrix_nbins(dataset_name, n_bins_range)
            if confusion_results:
                all_confusion_results[dataset_name] = confusion_results
                print(f"✅ Confusion matrix analysis completed for {dataset_name}")
            else:
                print(f"❌ Confusion matrix analysis failed for {dataset_name}")
        except Exception as e:
            print(f"❌ Confusion matrix analysis failed for {dataset_name}: {e}")
            continue
    
    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY: Confusion Matrix Analysis - n_bins Effect")
    print(f"{'='*60}")
    
    if not all_confusion_results:
        print("No successful analyses completed.")
    else:
        for dataset_name, confusion_results in all_confusion_results.items():
            print(f"\n{dataset_name}:")
            n_bins_values = list(confusion_results.keys())
            n_bins_values.sort()
            
            print("  n_bins | Accuracy | Confidence | Entropy | Margin  | Class A | Class B | Class C | Class D")
            print("  -------|----------|------------|---------|---------|---------|---------|---------|----------")
            for n_bins in n_bins_values:
                metrics = confusion_results[n_bins]
                class_dist = metrics['class_distribution_true']
                total = class_dist.sum()
                class_pcts = class_dist / total * 100
                print(f"  {n_bins:6d} | {metrics['accuracy']:.4f}   | {metrics['mean_confidence']:.4f}     | {metrics['mean_entropy']:.4f}  | {metrics['mean_prob_diff']:.4f}  | {class_pcts[0]:5.1f}%  | {class_pcts[1]:5.1f}%  | {class_pcts[2]:5.1f}%  | {class_pcts[3]:5.1f}%")
            
            # Find best n_bins for different metrics
            best_acc_nbins = max(n_bins_values, key=lambda n: confusion_results[n]['accuracy'])
            best_conf_nbins = max(n_bins_values, key=lambda n: confusion_results[n]['mean_confidence'])
            best_entropy_nbins = min(n_bins_values, key=lambda n: confusion_results[n]['mean_entropy'])
            best_margin_nbins = max(n_bins_values, key=lambda n: confusion_results[n]['mean_prob_diff'])
            
            print(f"  Best Accuracy: n_bins={best_acc_nbins} ({confusion_results[best_acc_nbins]['accuracy']:.4f})")
            print(f"  Best Confidence: n_bins={best_conf_nbins} ({confusion_results[best_conf_nbins]['mean_confidence']:.4f})")
            print(f"  Best Certainty (low entropy): n_bins={best_entropy_nbins} ({confusion_results[best_entropy_nbins]['mean_entropy']:.4f})")
            print(f"  Best Decision Margin: n_bins={best_margin_nbins} ({confusion_results[best_margin_nbins]['mean_prob_diff']:.4f})")
            
            # Probability distribution analysis
            print(f"\n  Class Probability Analysis:")
            print("    Class | Mean Prob | Std Prob | Best n_bins")
            print("    ------|-----------|----------|------------")
            class_names = ["A", "B", "C", "D"]
            for class_idx in range(4):
                # Find n_bins with highest mean probability for this class
                best_prob_nbins = max(n_bins_values, key=lambda n: confusion_results[n]['class_prob_means'][class_idx])
                best_mean_prob = confusion_results[best_prob_nbins]['class_prob_means'][class_idx]
                avg_mean_prob = np.mean([confusion_results[n]['class_prob_means'][class_idx] for n in n_bins_values])
                avg_std_prob = np.mean([confusion_results[n]['class_prob_stds'][class_idx] for n in n_bins_values])
                print(f"    {class_names[class_idx]:5s} | {avg_mean_prob:.4f}    | {avg_std_prob:.4f}   | {best_prob_nbins} ({best_mean_prob:.4f})")
    
    print(f"\n{'='*60}")
    print("Generated figures:")
    for dataset_name in all_confusion_results.keys():
        print(f"  figures/confusion_matrix_analysis/{dataset_name}_accuracy_vs_nbins.png")
        print(f"  figures/confusion_matrix_analysis/{dataset_name}_confusion_matrices_comparison.png (combined with avg probabilities)")
        print(f"  figures/confusion_matrix_analysis/individual/{dataset_name}_probability_analysis_nbins*.png")
        print(f"  figures/confusion_matrix_analysis/individual/{dataset_name}_confmat_Ti_nbins*.png")
    if len(all_confusion_results) > 1:
        print(f"  figures/confusion_matrix_analysis/summary_accuracy_vs_nbins.png")
        print(f"  figures/confusion_matrix_analysis/summary_class_distributions.png")
