#!/usr/bin/env python3
"""
Simple test script to debug segmentation fault in nbins_analysis.py
This script tests basic functionality step by step.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
import os
import sys
import gc
import psutil

warnings.filterwarnings("ignore")

def check_system_resources():
    """Check available system resources."""
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / (1024**3):.2f} GB")
    print(f"Memory usage: {memory.percent}%")

def test_data_loading():
    """Test loading a single dataset."""
    print("Step 1: Testing data loading...")
    
    # Check if evaluation files exist
    if not os.path.exists("tabpfn_evaluation.csv"):
        print("❌ tabpfn_evaluation.csv not found")
        return False
    
    if not os.path.exists("baseline_evaluation.csv"):
        print("❌ baseline_evaluation.csv not found")
        return False
    
    # Load evaluation data
    tabpfn_df = pd.read_csv("tabpfn_evaluation.csv")
    baseline_df = pd.read_csv("baseline_evaluation.csv")
    print(f"✅ Loaded evaluation data: {len(tabpfn_df)} TabPFN results, {len(baseline_df)} baseline results")
    
    # Get worst dataset
    merged = pd.merge(tabpfn_df, baseline_df, left_on='dataset', right_on='dataset_name', how='inner')
    baseline_ibs_cols = ['rsf_ibs', 'cph_ibs', 'dh_ibs', 'ds_ibs']
    merged['best_baseline_ibs'] = merged[baseline_ibs_cols].min(axis=1)
    merged['ibs_difference'] = merged['ibs'] - merged['best_baseline_ibs']
    worst_dataset = merged.sort_values('ibs_difference', ascending=False).iloc[0]['dataset']
    print(f"✅ Worst dataset identified: {worst_dataset}")
    
    return worst_dataset

def test_dataset_loading(dataset_name):
    """Test loading a specific dataset."""
    print(f"Step 2: Testing dataset loading for {dataset_name}...")
    
    file_path = os.path.join("test", f"{dataset_name}.csv")
    if not os.path.exists(file_path):
        print(f"❌ Dataset file not found: {file_path}")
        return False
    
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    
    return df

def test_basic_imports():
    """Test importing key libraries."""
    print("Step 3: Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        from tabpfn import TabPFNClassifier
        print("✅ TabPFN imported successfully")
        
        import sksurv
        print("✅ scikit-survival imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_tabpfn_basic():
    """Test basic TabPFN functionality."""
    print("Step 4: Testing basic TabPFN...")
    
    try:
        from tabpfn import TabPFNClassifier
        import torch
        
        # Create simple test data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.choice(['A', 'B', 'C', 'D'], 100))
        
        # Try to create and fit TabPFN
        model = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
        print("✅ TabPFN model created")
        
        model.fit(X, y)
        print("✅ TabPFN model fitted")
        
        # Test prediction
        probs = model.predict_proba(X[:10])
        print(f"✅ TabPFN predictions: {probs.shape}")
        
        del model
        gc.collect()
        return True
        
    except Exception as e:
        print(f"❌ TabPFN test failed: {e}")
        return False

def main():
    print("="*60)
    print("TabPFN Debug Test Script")
    print("="*60)
    
    # Check system resources
    check_system_resources()
    print()
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("❌ Basic imports failed - stopping tests")
        return
    print()
    
    # Test 2: Data loading
    worst_dataset = test_data_loading()
    if not worst_dataset:
        print("❌ Data loading failed - stopping tests")
        return
    print()
    
    # Test 3: Dataset loading
    df = test_dataset_loading(worst_dataset)
    if df is False:
        print("❌ Dataset loading failed - stopping tests")
        return
    print()
    
    # Test 4: Basic TabPFN
    if not test_tabpfn_basic():
        print("❌ TabPFN basic test failed - stopping tests")
        return
    print()
    
    print("="*60)
    print("✅ All basic tests passed! The issue might be in the full analysis loop.")
    print("✅ You can now try running the main nbins_analysis.py with DEBUG_MODE=True")
    print("="*60)

if __name__ == "__main__":
    main()
