import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

def load_and_prepare_data():
    """Load baseline and TabPFN results and prepare for comparison."""
    # Load baseline results
    baseline_df = pd.read_csv('baseline_evaluation.csv')
    
    # Load TabPFN results
    tabpfn_df = pd.read_csv('tabpfn_evaluation.csv')
    
    # Rename columns to be consistent
    tabpfn_df = tabpfn_df.rename(columns={
        'dataset': 'dataset_name',
        'c_index': 'tabpfn_c_index',
        'ibs': 'tabpfn_ibs',
        'mean_auc': 'tabpfn_mean_auc'
    })
    
    # Merge datasets on dataset name
    merged_df = pd.merge(baseline_df, tabpfn_df[['dataset_name', 'tabpfn_c_index', 'tabpfn_ibs', 'tabpfn_mean_auc']], 
                        on='dataset_name', how='inner')
    
    print(f"Successfully merged {len(merged_df)} datasets")
    return merged_df

def create_evaluation_metrics_table(df):
    """Create comprehensive evaluation metrics table for all models per dataset."""
    # Select relevant columns and rename for clarity
    eval_columns = ['dataset_name']
    
    # Add baseline model columns with proper names
    model_mapping = {
        'rsf': 'RSF',
        'cph': 'CoxPH', 
        'dh': 'DeepHit',
        'ds': 'DeepSurv'
    }
    
    metrics = ['c_index', 'ibs', 'mean_auc']
    
    # Reorganize data for better readability
    results = []
    for idx, row in df.iterrows():
        dataset = row['dataset_name']
        
        # Create base row
        base_row = {'dataset': dataset}
        
        # Add baseline models
        for model_key, model_name in model_mapping.items():
            for metric in metrics:
                col_name = f"{model_key}_{metric}"
                if col_name in df.columns:
                    base_row[f"{model_name}_{metric}"] = row[col_name]
        
        # Add TabPFN
        for metric in metrics:
            tabpfn_col = f"tabpfn_{metric}"
            if tabpfn_col in df.columns:
                base_row[f"TabPFN_{metric}"] = row[tabpfn_col]
        
        results.append(base_row)
    
    return pd.DataFrame(results)

def rank_models_per_dataset(df):
    """Rank models for each dataset and metric."""
    results = []
    
    # Define metrics and their optimization direction (True = higher is better, False = lower is better)
    metrics = {
        'c_index': True,     # Higher is better
        'ibs': False,        # Lower is better  
        'mean_auc': True     # Higher is better
    }
    
    # Define baseline models with proper names
    baseline_models = {
        'rsf': 'RSF',
        'cph': 'CoxPH', 
        'dh': 'DeepHit',
        'ds': 'DeepSurv'
    }
    
    for idx, row in df.iterrows():
        dataset_name = row['dataset_name']
        
        for metric, higher_is_better in metrics.items():
            # Collect values for all models
            model_values = {}
            
            # Baseline models
            for model_key, model_name in baseline_models.items():
                col_name = f"{model_key}_{metric}"
                if col_name in df.columns:
                    model_values[model_name] = row[col_name]
            
            # TabPFN model
            tabpfn_col = f"tabpfn_{metric}"
            if tabpfn_col in df.columns:
                model_values['TabPFN'] = row[tabpfn_col]
            
            # Rank models (1 = best, 5 = worst for 5 models)
            if higher_is_better:
                # Sort descending (highest value gets rank 1)
                sorted_models = sorted(model_values.items(), key=lambda x: x[1], reverse=True)
            else:
                # Sort ascending (lowest value gets rank 1)
                sorted_models = sorted(model_values.items(), key=lambda x: x[1])
            
            # Assign ranks
            for rank, (model, value) in enumerate(sorted_models, 1):
                results.append({
                    'dataset': dataset_name,
                    'metric': metric,
                    'model': model,
                    'value': value,
                    'rank': rank
                })
    
    return pd.DataFrame(results)

def create_ranking_table_per_dataset(rankings_df):
    """Create ranking table showing ranks for all models per dataset."""
    # Pivot the rankings to show models as columns
    ranking_results = []
    
    datasets = rankings_df['dataset'].unique()
    metrics = rankings_df['metric'].unique()
    models = rankings_df['model'].unique()
    
    for dataset in datasets:
        for metric in metrics:
            metric_data = rankings_df[(rankings_df['dataset'] == dataset) & 
                                    (rankings_df['metric'] == metric)]
            
            row = {'dataset': dataset, 'metric': metric}
            for _, model_row in metric_data.iterrows():
                row[f"{model_row['model']}_rank"] = model_row['rank']
                row[f"{model_row['model']}_value"] = round(model_row['value'], 4)
            
            ranking_results.append(row)
    
    return pd.DataFrame(ranking_results)

def calculate_average_ranks(rankings_df):
    """Calculate average rank for each model across all datasets for each metric."""
    avg_ranks = rankings_df.groupby(['metric', 'model'])['rank'].agg(['mean', 'std', 'count']).round(3)
    avg_ranks.columns = ['avg_rank', 'std_rank', 'count']
    avg_ranks = avg_ranks.reset_index()
    
    return avg_ranks

def perform_wilcoxon_tests(df):
    """Perform Wilcoxon signed-rank tests comparing TabPFN vs baseline models."""
    baseline_models = {
        'rsf': 'RSF',
        'cph': 'CoxPH', 
        'dh': 'DeepHit',
        'ds': 'DeepSurv'
    }
    metrics = ['c_index', 'ibs', 'mean_auc']
    
    wilcoxon_results = []
    
    for metric in metrics:
        higher_is_better = metric != 'ibs'  # IBS: lower is better
        
        for baseline_key, baseline_name in baseline_models.items():
            # Get paired values (remove NaN pairs)
            tabpfn_vals = []
            baseline_vals = []
            
            for idx, row in df.iterrows():
                tabpfn_val = row[f'tabpfn_{metric}']
                baseline_val = row[f'{baseline_key}_{metric}']
                
                if not (pd.isna(tabpfn_val) or pd.isna(baseline_val)):
                    tabpfn_vals.append(tabpfn_val)
                    baseline_vals.append(baseline_val)
            
            if len(tabpfn_vals) < 5:  # Need at least 5 pairs for meaningful test
                wilcoxon_results.append({
                    'metric': metric,
                    'baseline_model': baseline_name,
                    'n_pairs': len(tabpfn_vals),
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'effect_size': np.nan,
                    'interpretation': 'Insufficient data'
                })
                continue
            
            tabpfn_vals = np.array(tabpfn_vals)
            baseline_vals = np.array(baseline_vals)
            
            # For IBS (lower is better), we want to test if TabPFN < baseline
            # For other metrics (higher is better), we want to test if TabPFN > baseline
            if higher_is_better:
                differences = tabpfn_vals - baseline_vals  # Positive = TabPFN better
            else:
                differences = baseline_vals - tabpfn_vals  # Positive = TabPFN better
            
            # Perform Wilcoxon signed-rank test
            try:
                statistic, p_value = wilcoxon(differences, alternative='greater')
                
                # Calculate effect size (r = Z / sqrt(N))
                # Z-score approximation for large samples
                n = len(differences)
                mean_rank = n * (n + 1) / 4
                var_rank = n * (n + 1) * (2 * n + 1) / 24
                z_score = (statistic - mean_rank) / np.sqrt(var_rank)
                effect_size = abs(z_score) / np.sqrt(n)
                
                # Interpretation
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                elif p_value < 0.1:
                    significance = "."
                else:
                    significance = ""
                
                if p_value < 0.05:
                    if higher_is_better:
                        interpretation = f"TabPFN significantly better {significance}"
                    else:
                        interpretation = f"TabPFN significantly better {significance}"
                else:
                    interpretation = f"No significant difference {significance}"
                
            except ValueError as e:
                # Handle case where all differences are zero
                statistic = np.nan
                p_value = 1.0
                effect_size = 0.0
                interpretation = "No differences"
            
            wilcoxon_results.append({
                'metric': metric,
                'baseline_model': baseline_name,
                'n_pairs': len(tabpfn_vals),
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': effect_size,
                'interpretation': interpretation,
                'median_diff': np.median(differences),
                'mean_diff': np.mean(differences)
            })
    
    return pd.DataFrame(wilcoxon_results)

def plot_wilcoxon_results(wilcoxon_df):
    """Create visualization of Wilcoxon signed-rank test results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['c_index', 'ibs', 'mean_auc']
    metric_titles = ['C-Index', 'IBS', 'Mean AUC']
    
    # Define consistent model colors and order
    model_order = ['CoxPH', 'RSF', 'DeepHit', 'DeepSurv']  # Only baseline models for Wilcoxon
    model_colors = {
        'RSF': '#D79B00',
        'CoxPH': '#82B366', 
        'DeepHit': '#B85450',
        'DeepSurv': '#9673A6',
        'TabPFN': '#6C8EBF'
    }
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        data = wilcoxon_df[wilcoxon_df['metric'] == metric].copy()
        
        # Reorder data according to model_order
        data['baseline_model'] = pd.Categorical(data['baseline_model'], categories=model_order, ordered=True)
        data = data.sort_values('baseline_model')
        
        # Use model-specific colors
        colors = []
        for _, row in data.iterrows():
            baseline_model = row['baseline_model']
            colors.append(model_colors.get(baseline_model, 'gray'))
        
        # Plot -log10(p-value) for better visualization
        log_p_values = [-np.log10(p) if not pd.isna(p) and p > 0 else 0 for p in data['p_value']]
        
        bars = axes[i].bar(data['baseline_model'], log_p_values, color=colors)
        axes[i].set_title(f'Wilcoxon Test Results\n{title}', fontsize=18, fontweight='bold')
        axes[i].set_ylabel('-log10(p-value)', fontsize=16)
        axes[i].set_xlabel('Baseline Model', fontsize=16)
        
        # Add significance lines
        axes[i].axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.7, label='p=0.05')
        axes[i].axhline(y=-np.log10(0.01), color='black', linestyle=':', alpha=0.7, label='p=0.01')
        
        # Set y-axis limits to accommodate text labels
        max_height = max(log_p_values) if log_p_values else 1
        axes[i].set_ylim(0, max_height * 1.3)  # Add 30% padding for text labels
        
        # Add value labels on bars with adjusted positioning
        for bar, p_val, interpretation in zip(bars, data['p_value'], data['interpretation']):
            if not pd.isna(p_val):
                # Position text inside the bar if it's tall, otherwise above
                text_y = min(bar.get_height() * 0.9, bar.get_height() - 0.1) if bar.get_height() > max_height * 0.7 else bar.get_height() + max_height * 0.05
                text_color = 'white' if bar.get_height() > max_height * 0.7 else 'black'
                
                axes[i].text(bar.get_x() + bar.get_width()/2, text_y,
                           f'p={p_val:.3f}', ha='center', va='center' if bar.get_height() > max_height * 0.7 else 'bottom', 
                           fontsize=12, rotation=0, color=text_color, fontweight='bold')
        
        axes[i].legend(fontsize=14)
        axes[i].tick_params(axis='x', rotation=45, labelsize=14)
        axes[i].tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    plt.savefig('./figures/wilcoxon_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_effect_sizes(wilcoxon_df):
    """Create visualization of effect sizes from Wilcoxon tests."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['c_index', 'ibs', 'mean_auc']
    metric_titles = ['C-Index', 'IBS', 'Mean AUC']
    
    # Define consistent model colors and order
    model_order = ['CoxPH', 'RSF', 'DeepHit', 'DeepSurv']  # Only baseline models for Wilcoxon
    model_colors = {
        'RSF': '#D79B00',
        'CoxPH': '#82B366', 
        'DeepHit': '#B85450',
        'DeepSurv': '#9673A6',
        'TabPFN': '#6C8EBF'
    }
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        data = wilcoxon_df[wilcoxon_df['metric'] == metric].copy()
        
        # Reorder data according to model_order
        data['baseline_model'] = pd.Categorical(data['baseline_model'], categories=model_order, ordered=True)
        data = data.sort_values('baseline_model')
        
        # Use model-specific colors
        colors = []
        for _, row in data.iterrows():
            baseline_model = row['baseline_model']
            colors.append(model_colors.get(baseline_model, 'gray'))
        
        bars = axes[i].bar(data['baseline_model'], data['effect_size'], color=colors)
        axes[i].set_title(f'Effect Sizes (r)\n{title}', fontsize=18, fontweight='bold')
        axes[i].set_ylabel('Effect Size (r)', fontsize=16)
        axes[i].set_xlabel('Baseline Model', fontsize=16)
        
        # Add effect size interpretation lines
        axes[i].axhline(y=0.1, color='gray', linestyle='--', alpha=0.7, label='Small (0.1)')
        axes[i].axhline(y=0.3, color='gray', linestyle='--', alpha=0.7, label='Medium (0.3)')
        axes[i].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Large (0.5)')
        
        # Add value labels on bars
        for bar, effect_size in zip(bars, data['effect_size']):
            if not pd.isna(effect_size):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{effect_size:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        axes[i].legend(fontsize=14)
        axes[i].tick_params(axis='x', rotation=45, labelsize=14)
        axes[i].tick_params(axis='y', labelsize=14)
        axes[i].set_ylim(0, max(0.6, data['effect_size'].max() * 1.1) if not data['effect_size'].isna().all() else 0.6)
    
    plt.tight_layout()
    plt.savefig('./figures/wilcoxon_effect_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_average_ranks(avg_ranks_df):
    """Create visualization of average ranks with standard deviations and save each metric separately."""
    metrics = ['c_index', 'ibs', 'mean_auc']
    metric_titles = ['C-Index', 'IBS', 'Mean AUC']
    
    # Define model order and colors with specified hex codes
    model_order = ['CoxPH', 'RSF', 'DeepHit', 'DeepSurv', 'TabPFN']
    model_colors = {
        'RSF': '#D79B00',
        'CoxPH': '#82B366', 
        'DeepHit': '#B85450',
        'DeepSurv': '#9673A6',
        'TabPFN': '#6C8EBF'
    }
    
    # Create individual plots for each metric
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        data = avg_ranks_df[avg_ranks_df['metric'] == metric].copy()
        data['model'] = pd.Categorical(data['model'], categories=model_order, ordered=True)
        data = data.sort_values('model')
        
        # Use consistent model colors
        colors = [model_colors.get(model, 'gray') for model in data['model']]
        
        # Create bars with error bars (standard deviations)
        bars = ax.bar(data['model'], data['avg_rank'], 
                     yerr=data['std_rank'], 
                     color=colors, 
                     alpha=0.7,
                     capsize=5,
                     error_kw={'ecolor': 'black', 'capthick': 2})
        
        ax.set_title(f'{title} - Average Rank ± Standard Deviation', fontsize=20, fontweight='bold')
        ax.set_ylabel('Average Rank', fontsize=18)
        ax.set_xlabel('Model', fontsize=18)
        ax.tick_params(axis='x', rotation=45, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        # Add value labels on bars with std dev
        for bar, avg_rank, std_rank in zip(bars, data['avg_rank'], data['std_rank']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_rank + 0.1,
                   f'{avg_rank:.2f}±{std_rank:.2f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=14)
        
        # Add horizontal line at rank 3 (middle rank for 5 models)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.7, label='Middle Rank (3.0)')
        ax.legend(fontsize=16)
        ax.set_ylim(0, 6.0)  # Increased to accommodate error bars
        
        plt.tight_layout()
        plt.savefig(f'./figures/{metric}_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print("\nCreating evaluation metrics table...")
    eval_metrics_df = create_evaluation_metrics_table(df)
    
    print("\nRanking models for each dataset and metric...")
    rankings_df = rank_models_per_dataset(df)
    
    print("\nCreating ranking table per dataset...")
    ranking_table_df = create_ranking_table_per_dataset(rankings_df)
    
    print("\nCalculating average ranks...")
    avg_ranks = calculate_average_ranks(rankings_df)
    
    # Save the two main result files requested
    print("\nSaving main result files...")
    eval_metrics_df.to_csv('evaluation_results.csv', index=False)
    ranking_table_df.to_csv('ranking_results.csv', index=False)

    print("\n" + "="*60)
    print("AVERAGE RANKS BY MODEL AND METRIC")
    print("="*60)
    
    for metric in ['c_index', 'ibs', 'mean_auc']:
        print(f"\n{metric.upper()} Rankings:")
        metric_data = avg_ranks[avg_ranks['metric'] == metric].sort_values('avg_rank')
        print(metric_data.to_string(index=False))
    
    print("\n" + "="*60)
    print("WILCOXON SIGNED-RANK TEST RESULTS")
    print("="*60)
    
    wilcoxon_df = perform_wilcoxon_tests(df)
    
    for metric in ['c_index', 'ibs', 'mean_auc']:
        print(f"\n{metric.upper()}:")
        metric_data = wilcoxon_df[wilcoxon_df['metric'] == metric]
        display_cols = ['baseline_model', 'n_pairs', 'p_value', 'effect_size', 'median_diff', 'interpretation']
        print(metric_data[display_cols].to_string(index=False))
    
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print("="*60)
    
    # Count significant results
    sig_results = wilcoxon_df[wilcoxon_df['p_value'] < 0.05]
    print(f"\nSignificant results (p < 0.05): {len(sig_results)}/{len(wilcoxon_df)}")
    
    if len(sig_results) > 0:
        print("\nSignificant comparisons:")
        for _, row in sig_results.iterrows():
            print(f"  {row['metric']} - TabPFN vs {row['baseline_model']}: "
                  f"p={row['p_value']:.4f}, effect size={row['effect_size']:.3f}")
    
    # Effect size summary
    print(f"\nEffect size summary:")
    print(f"  Large effects (r ≥ 0.5): {len(wilcoxon_df[wilcoxon_df['effect_size'] >= 0.5])}")
    print(f"  Medium effects (0.3 ≤ r < 0.5): {len(wilcoxon_df[(wilcoxon_df['effect_size'] >= 0.3) & (wilcoxon_df['effect_size'] < 0.5)])}")
    print(f"  Small effects (0.1 ≤ r < 0.3): {len(wilcoxon_df[(wilcoxon_df['effect_size'] >= 0.1) & (wilcoxon_df['effect_size'] < 0.3)])}")
    print(f"  Negligible effects (r < 0.1): {len(wilcoxon_df[wilcoxon_df['effect_size'] < 0.1])}")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall ranking summary with standard deviations
    overall_avg_rank = avg_ranks.groupby('model').agg({
        'avg_rank': 'mean',
        'std_rank': 'mean'
    }).round(3)
    overall_avg_rank = overall_avg_rank.sort_values('avg_rank')
    
    print("\nOverall Average Rank Across All Metrics (Mean ± Std):")
    for model, stats in overall_avg_rank.iterrows():
        print(f"{model:10s}: {stats['avg_rank']:.3f} ± {stats['std_rank']:.3f}")
    
    # Best performing model for each metric with confidence
    print("\nBest Model by Metric (lowest average rank ± std):")
    for metric in ['c_index', 'ibs', 'mean_auc']:
        metric_data = avg_ranks[avg_ranks['metric'] == metric]
        best_idx = metric_data['avg_rank'].idxmin()
        best_model = metric_data.loc[best_idx]
        print(f"{metric:12s}: {best_model['model']} (avg rank: {best_model['avg_rank']:.3f} ± {best_model['std_rank']:.3f})")
    
    # Ranking consistency analysis
    print("\nRanking Consistency (Lower Std = More Consistent):")
    consistency = avg_ranks.groupby('model')['std_rank'].mean().sort_values()
    for model, std in consistency.items():
        print(f"{model:10s}: {std:.3f}")
    
    # Statistical significance summary with confidence intervals
    print(f"\nWilcoxon Test Summary:")
    print(f"  Total comparisons: {len(wilcoxon_df)}")
    print(f"  Significant results (p < 0.05): {len(sig_results)}")
    print(f"  Highly significant (p < 0.01): {len(wilcoxon_df[wilcoxon_df['p_value'] < 0.01])}")
    print(f"  Very highly significant (p < 0.001): {len(wilcoxon_df[wilcoxon_df['p_value'] < 0.001])}")
    
    # Effect size distribution
    valid_effects = wilcoxon_df['effect_size'].dropna()
    if len(valid_effects) > 0:
        print(f"\nEffect Size Distribution:")
        print(f"  Mean effect size: {valid_effects.mean():.3f} ± {valid_effects.std():.3f}")
        print(f"  Median effect size: {valid_effects.median():.3f}")
        print(f"  Range: {valid_effects.min():.3f} - {valid_effects.max():.3f}")
    
    print("\nCreating visualizations...")
    plot_average_ranks(avg_ranks)
    plot_wilcoxon_results(wilcoxon_df)
    plot_effect_sizes(wilcoxon_df)
    
    # Save additional analysis files
    avg_ranks.to_csv('average_rankings.csv', index=False)
    wilcoxon_df.to_csv('wilcoxon_test_results.csv', index=False)

if __name__ == "__main__":
    main()
