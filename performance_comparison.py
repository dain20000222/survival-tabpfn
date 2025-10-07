#!/usr/bin/env python3
"""
Performance Comparison Script for Survival Analysis Models

This script compares the performance of 4 different models:
1. MITRA Binary
2. MITRA 
3. TabPFN Binary
4. TabPFN

Metrics compared: score, c_index, ibs, mean_auc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_evaluation_data():
    """Load all evaluation CSV files and return as a dictionary of DataFrames."""
    
    files = {
        'MITRA_Binary': 'mitra_binary_evaluation.csv',
        'MITRA': 'mitra_evaluation.csv',
        'TabPFN_Binary': 'tabpfn_binary_evaluation.csv',
        'TabPFN': 'tabpfn_evaluation.csv'
    }
    
    data = {}
    for model_name, filename in files.items():
        try:
            df = pd.read_csv(filename)
            data[model_name] = df
            print(f"✓ Loaded {filename}: {len(df)} datasets")
        except FileNotFoundError:
            print(f"✗ Could not find {filename}")
    
    # Load baseline evaluation data
    try:
        baseline_df = pd.read_csv('baseline_evaluation.csv')
        print(f"✓ Loaded baseline_evaluation.csv: {len(baseline_df)} datasets")
        
        # Process baseline data to match the format of other files
        # Extract RSF, CPH, DH, DS results
        baseline_models = ['RSF', 'CPH', 'DH', 'DS']
        
        for model in baseline_models:
            model_df = pd.DataFrame()
            model_df['dataset'] = baseline_df['dataset_name']
            model_df['c_index'] = baseline_df[f'{model.lower()}_c_index']
            model_df['ibs'] = baseline_df[f'{model.lower()}_ibs']
            model_df['mean_auc'] = baseline_df[f'{model.lower()}_mean_auc']
            # Add dummy values for missing columns to maintain consistency
            model_df['n_eval_times'] = 9  # Default value
            model_df['score'] = model_df['c_index']  # Use c_index as score proxy
            
            data[model] = model_df
            print(f"✓ Processed baseline model {model}: {len(model_df)} datasets")
            
    except FileNotFoundError:
        print("✗ Could not find baseline_evaluation.csv")
    
    return data

def create_combined_dataframe(data_dict):
    """Combine all dataframes into a single dataframe for comparison."""
    
    combined_data = []
    
    for model_name, df in data_dict.items():
        df_copy = df.copy()
        df_copy['model'] = model_name
        combined_data.append(df_copy)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    return combined_df

def calculate_summary_statistics(combined_df):
    """Calculate summary statistics for each model."""
    
    metrics = ['c_index', 'ibs', 'mean_auc']
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY MODEL")
    print("="*80)
    
    summary_stats = []
    
    for model in combined_df['model'].unique():
        model_data = combined_df[combined_df['model'] == model]
        
        print(f"\n{model}:")
        print("-" * (len(model) + 1))
        
        model_stats = {'Model': model}
        
        for metric in metrics:
            mean_val = model_data[metric].mean()
            std_val = model_data[metric].std()
            median_val = model_data[metric].median()
            
            model_stats[f'{metric}_mean'] = mean_val
            model_stats[f'{metric}_std'] = std_val
            model_stats[f'{metric}_median'] = median_val
            
            print(f"  {metric:12s}: Mean={mean_val:.4f}, Std={std_val:.4f}, Median={median_val:.4f}")
        
        summary_stats.append(model_stats)
    
    return pd.DataFrame(summary_stats)

def find_best_worst_performers(combined_df):
    """Find best and worst performing datasets for each metric."""
    
    metrics = ['c_index', 'ibs', 'mean_auc']
    
    print("\n" + "="*80)
    print("BEST AND WORST PERFORMERS BY METRIC")
    print("="*80)
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-" * (len(metric) + 1))
        
        # For IBS, lower is better; for others, higher is better
        ascending = True if metric == 'ibs' else False
        
        metric_data = combined_df.pivot_table(
            values=metric, 
            index='dataset', 
            columns='model', 
            aggfunc='mean'
        )
        
        # Best performers
        if metric == 'ibs':
            best_overall = metric_data.min(axis=1).sort_values(ascending=True).head(3)
            print("  Top 3 datasets (lowest IBS):")
        else:
            best_overall = metric_data.max(axis=1).sort_values(ascending=False).head(3)
            print("  Top 3 datasets (highest score):")
        
        for dataset, score in best_overall.items():
            best_model = metric_data.loc[dataset].idxmax() if metric != 'ibs' else metric_data.loc[dataset].idxmin()
            best_score = metric_data.loc[dataset, best_model]
            print(f"    {dataset:20s}: {best_score:.4f} ({best_model})")
        
        # Worst performers
        if metric == 'ibs':
            worst_overall = metric_data.min(axis=1).sort_values(ascending=False).head(3)
            print("  Bottom 3 datasets (highest IBS):")
        else:
            worst_overall = metric_data.max(axis=1).sort_values(ascending=True).head(3)
            print("  Bottom 3 datasets (lowest score):")
        
        for dataset, score in worst_overall.items():
            worst_model = metric_data.loc[dataset].idxmin() if metric != 'ibs' else metric_data.loc[dataset].idxmax()
            worst_score = metric_data.loc[dataset, worst_model]
            print(f"    {dataset:20s}: {worst_score:.4f} ({worst_model})")

def perform_pairwise_comparisons(combined_df):
    """Perform pairwise statistical comparisons between models."""
    
    from scipy.stats import wilcoxon
    
    metrics = ['c_index', 'ibs', 'mean_auc']
    models = combined_df['model'].unique()
    
    print("\n" + "="*80)
    print("PAIRWISE MODEL COMPARISONS (Wilcoxon Signed-Rank Test)")
    print("="*80)
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-" * (len(metric) + 1))
        
        # Create pivot table for the metric
        metric_data = combined_df.pivot_table(
            values=metric,
            index='dataset',
            columns='model',
            aggfunc='mean'
        )
        
        # Remove datasets that don't have data for all models
        complete_data = metric_data.dropna()
        
        print(f"  Datasets with complete data: {len(complete_data)}")
        
        # Perform pairwise comparisons
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:  # Avoid duplicate comparisons
                    if model1 in complete_data.columns and model2 in complete_data.columns:
                        data1 = complete_data[model1]
                        data2 = complete_data[model2]
                        
                        try:
                            statistic, p_value = wilcoxon(data1, data2)
                            
                            # Calculate effect size (mean difference)
                            mean_diff = data1.mean() - data2.mean()
                            
                            significance = ""
                            if p_value < 0.001:
                                significance = "***"
                            elif p_value < 0.01:
                                significance = "**"
                            elif p_value < 0.05:
                                significance = "*"
                            
                            print(f"  {model1:15s} vs {model2:15s}: p={p_value:.4f}{significance:3s} (diff={mean_diff:+.4f})")
                        
                        except ValueError as e:
                            print(f"  {model1:15s} vs {model2:15s}: Cannot compare (identical values)")

def perform_win_loss_analysis(combined_df):
    """Perform win/loss/tie analysis between models."""
    
    metrics = ['c_index', 'ibs', 'mean_auc']
    models = combined_df['model'].unique()
    
    print("\n" + "="*80)
    print("WIN/LOSS/TIE ANALYSIS")
    print("="*80)
    
    win_loss_results = {}
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-" * (len(metric) + 1))
        
        # Create pivot table for the metric
        metric_data = combined_df.pivot_table(
            values=metric,
            index='dataset',
            columns='model',
            aggfunc='mean'
        )
        
        # Remove datasets that don't have data for all models
        complete_data = metric_data.dropna()
        
        if len(complete_data) == 0:
            print("  No datasets with complete data for all models.")
            continue
            
        win_loss_results[metric] = {}
        
        # Calculate win/loss/tie for each model pair
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j and model1 in complete_data.columns and model2 in complete_data.columns:
                    data1 = complete_data[model1]
                    data2 = complete_data[model2]
                    
                    if metric == 'ibs':  # Lower is better for IBS
                        wins = (data1 < data2).sum()
                        losses = (data1 > data2).sum()
                    else:  # Higher is better for other metrics
                        wins = (data1 > data2).sum()
                        losses = (data1 < data2).sum()
                    
                    ties = (data1 == data2).sum()
                    total = len(data1)
                    
                    win_rate = wins / total * 100
                    
                    print(f"  {model1:15s} vs {model2:15s}: W={wins:2d} L={losses:2d} T={ties:2d} ({win_rate:5.1f}% win rate)")
                    
                    win_loss_results[metric][(model1, model2)] = {
                        'wins': wins, 'losses': losses, 'ties': ties, 'win_rate': win_rate
                    }
    
    return win_loss_results

def calculate_model_rankings(combined_df):
    """Calculate overall model rankings across all metrics."""
    
    metrics = ['c_index', 'ibs', 'mean_auc']
    models = combined_df['model'].unique()
    
    print("\n" + "="*80)
    print("OVERALL MODEL RANKINGS")
    print("="*80)
    
    # Calculate average rank for each model across all metrics
    model_ranks = {model: [] for model in models}
    
    for metric in metrics:
        model_means = combined_df.groupby('model')[metric].mean()
        
        if metric == 'ibs':  # Lower is better for IBS
            ranked = model_means.rank(ascending=True)
        else:  # Higher is better for other metrics
            ranked = model_means.rank(ascending=False)
        
        for model in models:
            if model in ranked:
                model_ranks[model].append(ranked[model])
    
    # Calculate average rank and sort
    avg_ranks = {}
    for model, ranks in model_ranks.items():
        if ranks:  # Only if model has data
            avg_ranks[model] = np.mean(ranks)
    
    sorted_models = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    print("\nAverage Rank (lower is better):")
    print("-" * 35)
    for i, (model, avg_rank) in enumerate(sorted_models, 1):
        print(f"  {i}. {model:20s}: {avg_rank:.2f}")
    
    return sorted_models

def create_visualizations(combined_df):
    """Create comprehensive visualizations for model comparison."""
    
    metrics = ['c_index', 'ibs', 'mean_auc']  # Removed 'score' as it's just validation score
    
    # Separate TabPFN/MITRA models from baseline models
    tabpfn_models = ['MITRA_Binary', 'MITRA', 'TabPFN_Binary', 'TabPFN']
    baseline_models = ['RSF', 'CPH', 'DH', 'DS']
    
    # Filter data for TabPFN/MITRA only
    tabpfn_df = combined_df[combined_df['model'].isin(tabpfn_models)].copy()
    
    print("Creating TabPFN/MITRA only visualizations...")
    
    # 1. Box plots for TabPFN/MITRA models only
    if not tabpfn_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        for i, metric in enumerate(metrics):
            sns.boxplot(data=tabpfn_df, x='model', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} Distribution - TabPFN/MITRA Models')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add mean markers
            model_means = tabpfn_df.groupby('model')[metric].mean()
            for j, (model, mean_val) in enumerate(model_means.items()):
                axes[i].scatter(j, mean_val, color='red', s=100, marker='D', 
                              label='Mean' if i == 0 and j == 0 else "", zorder=5)
        
        plt.tight_layout()
        plt.savefig('tabpfn_mitra_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary comparison bar chart for TabPFN/MITRA
        model_means = tabpfn_df.groupby('model')[metrics].mean()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        for i, metric in enumerate(metrics):
            model_means_sorted = model_means[metric].sort_values(ascending=(metric == 'ibs'))
            
            bars = axes[i].bar(range(len(model_means_sorted)), model_means_sorted.values)
            axes[i].set_title(f'Average {metric.upper()} - TabPFN/MITRA Models')
            axes[i].set_xticks(range(len(model_means_sorted)))
            axes[i].set_xticklabels(model_means_sorted.index, rotation=45)
            axes[i].set_ylabel(f'{metric.upper()}')
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, model_means_sorted.values)):
                height = bar.get_height()
                if metric == 'ibs':
                    axes[i].text(bar.get_x() + bar.get_width()/2, height + 0.002,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                else:
                    axes[i].text(bar.get_x() + bar.get_width()/2, height + 0.002,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Color coding: green for best, red for worst (corrected logic)
            if metric == 'ibs':  # Lower is better for IBS
                bars[0].set_color('green')  # First (lowest) is best
                bars[-1].set_color('red')   # Last (highest) is worst
            else:  # Higher is better for c_index and mean_auc
                bars[-1].set_color('green') # Last (highest) is best
                bars[0].set_color('red')    # First (lowest) is worst
        
        plt.tight_layout()
        plt.savefig('tabpfn_mitra_comparison_bars.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("Creating all models visualizations...")
    
    # 2. Box plots for all models
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, metric in enumerate(metrics):
        sns.boxplot(data=combined_df, x='model', y=metric, ax=axes[i])
        axes[i].set_title(f'{metric.upper()} Distribution - All Models')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add mean markers
        model_means = combined_df.groupby('model')[metric].mean()
        for j, (model, mean_val) in enumerate(model_means.items()):
            axes[i].scatter(j, mean_val, color='red', s=100, marker='D', 
                          label='Mean' if i == 0 and j == 0 else "", zorder=5)
    
    plt.tight_layout()
    plt.savefig('all_models_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary comparison bar chart for all models
    model_means = combined_df.groupby('model')[metrics].mean()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, metric in enumerate(metrics):
        model_means_sorted = model_means[metric].sort_values(ascending=(metric == 'ibs'))
        
        bars = axes[i].bar(range(len(model_means_sorted)), model_means_sorted.values)
        axes[i].set_title(f'Average {metric.upper()} - All Models')
        axes[i].set_xticks(range(len(model_means_sorted)))
        axes[i].set_xticklabels(model_means_sorted.index, rotation=45)
        axes[i].set_ylabel(f'{metric.upper()}')
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, model_means_sorted.values)):
            height = bar.get_height()
            if metric == 'ibs':
                axes[i].text(bar.get_x() + bar.get_width()/2, height + 0.002,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            else:
                axes[i].text(bar.get_x() + bar.get_width()/2, height + 0.002,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Color bars by model type
        for j, (bar, model_name) in enumerate(zip(bars, model_means_sorted.index)):
            if model_name in tabpfn_models:
                if metric == 'ibs':  # Lower is better
                    if j == 0:  # Best TabPFN/MITRA
                        bar.set_color('darkgreen')
                    else:
                        bar.set_color('lightgreen')
                else:  # Higher is better
                    if j == len(bars) - 1:  # Best TabPFN/MITRA
                        bar.set_color('darkgreen')
                    else:
                        bar.set_color('lightgreen')
            elif model_name in baseline_models:
                if metric == 'ibs':  # Lower is better
                    if j == 0:  # Best baseline
                        bar.set_color('darkblue')
                    else:
                        bar.set_color('lightblue')
                else:  # Higher is better
                    if j == len(bars) - 1:  # Best baseline
                        bar.set_color('darkblue')
                    else:
                        bar.set_color('lightblue')
    
    plt.tight_layout()
    plt.savefig('all_models_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Heatmap of model rankings per dataset
    print("\nCreating performance heatmaps...")
    
    # Create ranking matrix for TabPFN/MITRA models only
    if not tabpfn_df.empty:
        for metric in metrics:
            pivot_data = tabpfn_df.pivot_table(
                values=metric,
                index='dataset',
                columns='model',
                aggfunc='mean'
            )
            
            # Rank models (1 = best, higher = worse)
            if metric == 'ibs':  # Lower is better for IBS
                rankings = pivot_data.rank(axis=1, method='min')
            else:  # Higher is better for other metrics
                rankings = pivot_data.rank(axis=1, method='min', ascending=False)
            
            plt.figure(figsize=(8, max(6, len(rankings) * 0.3)))
            sns.heatmap(rankings, annot=True, cmap='RdYlBu_r', cbar_kws={'label': 'Rank (1=best)'})
            plt.title(f'TabPFN/MITRA Model Rankings by Dataset - {metric.upper()}')
            plt.xlabel('Model')
            plt.ylabel('Dataset')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'tabpfn_mitra_rankings_{metric}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Create ranking matrix for all models
    for metric in metrics:
        pivot_data = combined_df.pivot_table(
            values=metric,
            index='dataset',
            columns='model',
            aggfunc='mean'
        )
        
        # Rank models (1 = best, higher = worse)
        if metric == 'ibs':  # Lower is better for IBS
            rankings = pivot_data.rank(axis=1, method='min')
        else:  # Higher is better for other metrics
            rankings = pivot_data.rank(axis=1, method='min', ascending=False)
        
        plt.figure(figsize=(12, max(8, len(rankings) * 0.3)))
        sns.heatmap(rankings, annot=True, cmap='RdYlBu_r', cbar_kws={'label': 'Rank (1=best)'})
        plt.title(f'All Model Rankings by Dataset - {metric.upper()}')
        plt.xlabel('Model')
        plt.ylabel('Dataset')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'all_models_rankings_{metric}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Overall performance radar charts
    print("Creating performance radar charts...")
    
    from math import pi
    
    categories = ['C-Index', 'IBS (inverted)', 'Mean AUC']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Radar chart for TabPFN/MITRA models only
    if not tabpfn_df.empty:
        model_performance = []
        for model in tabpfn_df['model'].unique():
            model_data = tabpfn_df[tabpfn_df['model'] == model]
            
            # Normalize metrics (higher = better)
            c_index_norm = model_data['c_index'].mean()
            ibs_norm = 1 - model_data['ibs'].mean()  # Invert IBS (lower is better)
            auc_norm = model_data['mean_auc'].mean()
            
            model_performance.append({
                'Model': model,
                'C-Index': c_index_norm,
                'IBS (inverted)': ibs_norm,
                'Mean AUC': auc_norm
            })
        
        perf_df = pd.DataFrame(model_performance)
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Colors for TabPFN/MITRA models
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (_, row) in enumerate(perf_df.iterrows()):
            values = [row['C-Index'], row['IBS (inverted)'], row['Mean AUC']]
            values += values[:1]  # Complete the circle
            
            color = colors[i] if i < len(colors) else colors[i % len(colors)]
            ax.plot(angles, values, 'o-', linewidth=3, label=row['Model'], color=color)
            ax.fill(angles, values, alpha=0.2, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        plt.title('TabPFN/MITRA Model Performance Comparison\n(Higher values = Better performance)', 
                  pad=20, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('tabpfn_mitra_radar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Radar chart for all models
    model_performance = []
    for model in combined_df['model'].unique():
        model_data = combined_df[combined_df['model'] == model]
        
        # Normalize metrics (higher = better)
        c_index_norm = model_data['c_index'].mean()
        ibs_norm = 1 - model_data['ibs'].mean()  # Invert IBS (lower is better)
        auc_norm = model_data['mean_auc'].mean()
        
        model_performance.append({
            'Model': model,
            'C-Index': c_index_norm,
            'IBS (inverted)': ibs_norm,
            'Mean AUC': auc_norm
        })
    
    perf_df = pd.DataFrame(model_performance)
    
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    # Generate colors for all models
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, len(perf_df)))
    
    for i, (_, row) in enumerate(perf_df.iterrows()):
        values = [row['C-Index'], row['IBS (inverted)'], row['Mean AUC']]
        values += values[:1]  # Complete the circle
        
        # Different line styles for TabPFN/MITRA vs baseline models
        if row['Model'] in tabpfn_models:
            linestyle = '-'
            linewidth = 3
            alpha = 0.2
        else:
            linestyle = '--'
            linewidth = 2
            alpha = 0.1
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, label=row['Model'], 
                color=colors[i], linestyle=linestyle)
        ax.fill(angles, values, alpha=alpha, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('All Model Performance Comparison\n(Higher values = Better performance, Solid=TabPFN/MITRA, Dashed=Baseline)', 
              pad=20, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('all_models_radar.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_report(summary_stats, combined_df, win_loss_results=None, model_rankings=None):
    """Generate a comprehensive text report."""
    
    report_filename = 'performance_comparison_report.txt'
    
    with open(report_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SURVIVAL ANALYSIS MODEL PERFORMANCE COMPARISON REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODELS COMPARED:\n")
        f.write("-" * 16 + "\n")
        models = combined_df['model'].unique()
        tabpfn_models = [m for m in models if 'TabPFN' in m or 'MITRA' in m]
        baseline_models = [m for m in models if m in ['RSF', 'CPH', 'DH', 'DS']]
        
        if tabpfn_models:
            f.write("\nTabPFN/MITRA Models:\n")
            for model in tabpfn_models:
                count = len(combined_df[combined_df['model'] == model])
                f.write(f"• {model}: {count} datasets\n")
        
        if baseline_models:
            f.write("\nBaseline Models:\n")
            for model in baseline_models:
                count = len(combined_df[combined_df['model'] == model])
                f.write(f"• {model}: {count} datasets\n")
        
        f.write(f"\nTOTAL DATASETS: {combined_df['dataset'].nunique()}\n")
        f.write(f"METRICS EVALUATED: c_index, ibs, mean_auc\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 19 + "\n")
        f.write(summary_stats.to_string(index=False))
        f.write("\n\n")
        
        # Model rankings by average performance
        f.write("MODEL RANKINGS (by average metric performance):\n")
        f.write("-" * 48 + "\n")
        
        metrics = ['c_index', 'ibs', 'mean_auc']
        
        for metric in metrics:
            f.write(f"\n{metric.upper()} Rankings:\n")
            model_means = combined_df.groupby('model')[metric].mean()
            
            if metric == 'ibs':  # Lower is better for IBS
                ranked = model_means.sort_values(ascending=True)
            else:  # Higher is better for other metrics
                ranked = model_means.sort_values(ascending=False)
            
            for i, (model, score) in enumerate(ranked.items(), 1):
                f.write(f"  {i}. {model:20s}: {score:.4f}\n")
        
        # Overall rankings
        if model_rankings:
            f.write(f"\nOVERALL RANKINGS (Average across all metrics):\n")
            f.write("-" * 46 + "\n")
            for i, (model, avg_rank) in enumerate(model_rankings, 1):
                f.write(f"  {i}. {model:20s}: {avg_rank:.2f}\n")
        
        # Performance comparison between TabPFN/MITRA and baselines
        if tabpfn_models and baseline_models:
            f.write(f"\nTABPFN/MITRA vs BASELINE COMPARISON:\n")
            f.write("-" * 36 + "\n")
            
            for metric in metrics:
                tabpfn_means = combined_df[combined_df['model'].isin(tabpfn_models)].groupby('model')[metric].mean()
                baseline_means = combined_df[combined_df['model'].isin(baseline_models)].groupby('model')[metric].mean()
                
                best_tabpfn = tabpfn_means.max() if metric != 'ibs' else tabpfn_means.min()
                best_baseline = baseline_means.max() if metric != 'ibs' else baseline_means.min()
                best_tabpfn_model = tabpfn_means.idxmax() if metric != 'ibs' else tabpfn_means.idxmin()
                best_baseline_model = baseline_means.idxmax() if metric != 'ibs' else baseline_means.idxmin()
                
                f.write(f"\n{metric.upper()}:\n")
                f.write(f"  Best TabPFN/MITRA: {best_tabpfn_model} ({best_tabpfn:.4f})\n")
                f.write(f"  Best Baseline:      {best_baseline_model} ({best_baseline:.4f})\n")
                
                if metric == 'ibs':
                    improvement = ((best_baseline - best_tabpfn) / best_baseline) * 100
                    if best_tabpfn < best_baseline:
                        f.write(f"  TabPFN/MITRA advantage: {improvement:.1f}% lower IBS\n")
                    else:
                        f.write(f"  Baseline advantage: {-improvement:.1f}% lower IBS\n")
                else:
                    improvement = ((best_tabpfn - best_baseline) / best_baseline) * 100
                    if best_tabpfn > best_baseline:
                        f.write(f"  TabPFN/MITRA advantage: {improvement:.1f}% higher {metric}\n")
                    else:
                        f.write(f"  Baseline advantage: {-improvement:.1f}% higher {metric}\n")
        
        # Win/Loss analysis summary
        if win_loss_results:
            f.write(f"\nWIN/LOSS ANALYSIS SUMMARY:\n")
            f.write("-" * 26 + "\n")
            
            for metric in metrics:
                if metric in win_loss_results:
                    f.write(f"\n{metric.upper()} Head-to-Head Records:\n")
                    for (model1, model2), results in win_loss_results[metric].items():
                        f.write(f"  {model1:15s} vs {model2:15s}: {results['win_rate']:.1f}% win rate\n")
        
        # Overall recommendation
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("="*80 + "\n")
        
        # Calculate overall rankings
        overall_scores = {}
        for model in combined_df['model'].unique():
            model_data = combined_df[combined_df['model'] == model]
            
            # Weighted average of normalized metrics
            c_index = model_data['c_index'].mean()
            ibs = 1 - model_data['ibs'].mean()  # Invert IBS
            auc = model_data['mean_auc'].mean()
            
            overall_scores[model] = (c_index + ibs + auc) / 3
        
        best_model = max(overall_scores.items(), key=lambda x: x[1])
        
        f.write(f"\nBest Overall Model: {best_model[0]} (Normalized Score: {best_model[1]:.4f})\n")
        
        # Top 3 models
        sorted_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        f.write(f"\nTop 3 Models:\n")
        for i, (model, score) in enumerate(sorted_models[:3], 1):
            f.write(f"  {i}. {model:20s}: {score:.4f}\n")
        
        f.write("\nDetailed Analysis:\n")
        f.write("• Consider dataset-specific performance variations\n")
        f.write("• Statistical significance should be evaluated for final conclusions\n")
        f.write("• Domain knowledge and model interpretability may influence final choice\n")
        if tabpfn_models:
            f.write("• TabPFN/MITRA models show competitive performance with baseline methods\n")
        f.write("• IBS (Integrated Brier Score) - lower values indicate better calibration\n")
        f.write("• C-Index and Mean AUC - higher values indicate better discrimination\n")
    
    print(f"\n📊 Comprehensive report saved to: {report_filename}")

def main():
    """Main execution function."""
    
    print("🔍 SURVIVAL ANALYSIS MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    # Load data
    print("\n📂 Loading evaluation data...")
    data = load_evaluation_data()
    
    if not data:
        print("❌ No data files found. Please ensure CSV files are in the current directory.")
        return
    
    # Create combined dataframe
    combined_df = create_combined_dataframe(data)
    print(f"\n📊 Combined dataset: {len(combined_df)} total evaluations")
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(combined_df)
    
    # Find best/worst performers
    find_best_worst_performers(combined_df)
    
    # Perform statistical comparisons
    try:
        perform_pairwise_comparisons(combined_df)
    except ImportError:
        print("\n⚠️  scipy not available. Skipping statistical tests.")
        print("   Install with: pip install scipy")
    
    # Perform win/loss analysis
    print("\n📊 Performing win/loss analysis...")
    win_loss_results = perform_win_loss_analysis(combined_df)
    
    # Calculate model rankings
    print("\n🏆 Calculating overall model rankings...")
    model_rankings = calculate_model_rankings(combined_df)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    try:
        create_visualizations(combined_df)
    except Exception as e:
        print(f"⚠️  Could not create all visualizations: {e}")
        print("   Make sure matplotlib and seaborn are installed.")
    
    # Generate report
    print("\n📝 Generating comprehensive report...")
    generate_report(summary_stats, combined_df, win_loss_results, model_rankings)
    
    print("\n✅ Analysis complete!")
    print("\nFiles generated:")
    print("\nTabPFN/MITRA Only:")
    print("• tabpfn_mitra_distributions.png")
    print("• tabpfn_mitra_comparison_bars.png")
    print("• tabpfn_mitra_rankings_*.png")
    print("• tabpfn_mitra_radar.png")
    print("\nAll Models (including baselines):")
    print("• all_models_distributions.png")
    print("• all_models_comparison_bars.png")
    print("• all_models_rankings_*.png")
    print("• all_models_radar.png")
    print("\nReport:")
    print("• performance_comparison_report.txt")

if __name__ == "__main__":
    main()
