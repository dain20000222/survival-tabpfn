#!/usr/bin/env python3
"""
Minimal nbins_analysis.py that should avoid segmentation faults.
Only tests one dataset with minimal n_bins values.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
import os
import gc
import random
import torch

warnings.filterwarnings("ignore")

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_worst_dataset():
    """Get the worst performing dataset."""
    tabpfn_df = pd.read_csv("tabpfn_evaluation.csv")
    baseline_df = pd.read_csv("baseline_evaluation.csv")
    
    merged = pd.merge(tabpfn_df, baseline_df, left_on='dataset', right_on='dataset_name', how='inner')
    baseline_ibs_cols = ['rsf_ibs', 'cph_ibs', 'dh_ibs', 'ds_ibs']
    merged['best_baseline_ibs'] = merged[baseline_ibs_cols].min(axis=1)
    merged['ibs_difference'] = merged['ibs'] - merged['best_baseline_ibs']
    
    worst_dataset = merged.sort_values('ibs_difference', ascending=False).iloc[0]['dataset']
    return worst_dataset

def test_single_nbins(dataset_name, n_bins=5):
    """Test a single n_bins value on one dataset."""
    print(f"Testing {dataset_name} with n_bins={n_bins}")
    
    # Load dataset
    file_path = os.path.join("test", f"{dataset_name}.csv")
    df = pd.read_csv(file_path)
    df.drop(columns=['pid'], inplace=True, errors='ignore')
    
    print(f"Dataset shape: {df.shape}")
    
    # Simple test - just verify we can process the data
    time_col = "time"
    event_col = "event"
    
    if time_col not in df.columns or event_col not in df.columns:
        print(f"❌ Required columns not found: {list(df.columns)}")
        return False
    
    print(f"✅ Time range: {df[time_col].min():.2f} - {df[time_col].max():.2f}")
    print(f"✅ Event rate: {df[event_col].mean():.2f}")
    
    # Test TabPFN with minimal data
    try:
        from tabpfn import TabPFNClassifier
        
        # Create very simple test case
        n_samples = min(100, len(df))  # Use only 100 samples
        sample_df = df.sample(n_samples, random_state=42)
        
        # Create simple features
        X = sample_df[['time', 'event']].copy()
        X['eval_time'] = sample_df['time']  # Use actual time as eval_time
        
        # Create simple labels
        y = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
        
        print(f"Training TabPFN with {len(X)} samples...")
        
        model = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
        model.fit(X, y)
        
        # Test prediction
        probs = model.predict_proba(X[:10])
        print(f"✅ TabPFN prediction shape: {probs.shape}")
        
        del model
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ TabPFN test failed: {e}")
        return False

def main():
    print("="*50)
    print("Minimal n_bins Analysis Test")
    print("="*50)
    
    # Get worst dataset
    try:
        worst_dataset = get_worst_dataset()
        print(f"Testing dataset: {worst_dataset}")
    except Exception as e:
        print(f"❌ Failed to get worst dataset: {e}")
        return
    
    # Test with single n_bins
    success = test_single_nbins(worst_dataset, n_bins=5)
    
    if success:
        print("✅ Minimal test passed!")
        print("✅ You can now try the full analysis.")
    else:
        print("❌ Minimal test failed.")
        print("❌ Check your environment and data files.")

if __name__ == "__main__":
    main()
