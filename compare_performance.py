import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def load_and_clean_data(binary_file="tabpfn_binary_evaluation.csv", 
                       multiclass_file="tabpfn_evaluation.csv"):
    """Load and clean the evaluation results from both approaches."""
    try:
        # Load the CSV files
        binary_df = pd.read_csv(binary_file)
        multiclass_df = pd.read_csv(multiclass_file)
        
        # Standardize column names for easier comparison
        # Binary uses 'n_bins', Multi-class now uses 'n_eval_times'
        if 'n_eval_times' in multiclass_df.columns and 'n_bins' not in multiclass_df.columns:
            multiclass_df['n_bins'] = multiclass_df['n_eval_times']  # For backward compatibility
        
        # Add approach column
        binary_df['approach'] = 'Binary'
        multiclass_df['approach'] = 'Multi-class (A/B/C/D)'
        
        # Merge on dataset name to ensure we compare the same datasets
        merged = pd.merge(binary_df, multiclass_df, on='dataset', suffixes=('_binary', '_multiclass'))
        
        print(f"Loaded data for {len(merged)} datasets")
        print(f"Binary approach datasets: {len(binary_df)}")
        print(f"Multi-class approach datasets: {len(multiclass_df)}")
        print(f"Common datasets for comparison: {len(merged)}")
        
        return binary_df, multiclass_df, merged
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def statistical_comparison(merged_df):
    """Perform statistical comparison between approaches."""
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    
    metrics = ['c_index', 'ibs', 'mean_auc']
    results = {}
    
    for metric in metrics:
        binary_values = merged_df[f'{metric}_binary']
        multiclass_values = merged_df[f'{metric}_multiclass']
        
        # Paired t-test (since we're comparing same datasets)
        statistic, p_value = stats.ttest_rel(binary_values, multiclass_values)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        w_statistic, w_p_value = stats.wilcoxon(binary_values, multiclass_values, alternative='two-sided')
        
        # Effect size (Cohen's d for paired samples)
        diff = binary_values - multiclass_values
        pooled_std = np.sqrt((binary_values.var() + multiclass_values.var()) / 2)
        cohens_d = diff.mean() / pooled_std if pooled_std > 0 else 0
        
        # Count wins
        binary_wins = (binary_values > multiclass_values).sum()
        multiclass_wins = (multiclass_values > binary_values).sum()
        ties = (binary_values == multiclass_values).sum()
        
        # For IBS, lower is better, so flip the comparison
        if metric == 'ibs':
            binary_wins = (binary_values < multiclass_values).sum()
            multiclass_wins = (multiclass_values < binary_values).sum()
            better_approach = "Binary" if diff.mean() < 0 else "Multi-class"
        else:
            better_approach = "Binary" if diff.mean() > 0 else "Multi-class"
        
        results[metric] = {
            'binary_mean': binary_values.mean(),
            'multiclass_mean': multiclass_values.mean(),
            'difference_mean': diff.mean(),
            'difference_std': diff.std(),
            't_statistic': statistic,
            'p_value': p_value,
            'w_statistic': w_statistic,
            'w_p_value': w_p_value,
            'cohens_d': cohens_d,
            'binary_wins': binary_wins,
            'multiclass_wins': multiclass_wins,
            'ties': ties,
            'better_approach': better_approach
        }
        
        print(f"\n{metric.upper()} (Higher is better{'*' if metric != 'ibs' else ' - LOWER IS BETTER*'}):")
        print(f"  Binary mean:      {binary_values.mean():.4f} ± {binary_values.std():.4f}")
        print(f"  Multi-class mean: {multiclass_values.mean():.4f} ± {multiclass_values.std():.4f}")
        print(f"  Difference:       {diff.mean():.4f} ± {diff.std():.4f}")
        print(f"  Better approach:  {better_approach}")
        print(f"  Wins: Binary={binary_wins}, Multi-class={multiclass_wins}, Ties={ties}")
        print(f"  Paired t-test:    t={statistic:.3f}, p={p_value:.4f}")
        print(f"  Wilcoxon test:    W={w_statistic:.1f}, p={w_p_value:.4f}")
        print(f"  Effect size (d):  {cohens_d:.3f}")
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        print(f"  Effect size:      {effect_interpretation}")
    
    return results

def summary_statistics(binary_df, multiclass_df, merged_df):
    """Print summary statistics for both approaches."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\nBinary Classification Approach:")
    print(binary_df[['c_index', 'ibs', 'mean_auc']].describe())
    
    print("\nMulti-class (A/B/C/D) Classification Approach:")
    print(multiclass_df[['c_index', 'ibs', 'mean_auc']].describe())
    
    print(f"\nEvaluation Configuration:")
    print("Binary approach n_bins distribution:")
    print(binary_df['n_bins'].value_counts().sort_index())
    print("\nMulti-class approach n_eval_times distribution:")
    if 'n_eval_times' in multiclass_df.columns:
        print(multiclass_df['n_eval_times'].value_counts().sort_index())
    elif 'n_bins' in multiclass_df.columns:
        print(multiclass_df['n_bins'].value_counts().sort_index())
    else:
        print("No evaluation time configuration found")

def create_visualizations(merged_df, save_plots=True):
    """Create visualization plots comparing the approaches."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Paired comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Comparison: Binary vs Multi-class Classification', fontsize=16, fontweight='bold')
    
    metrics = ['c_index', 'ibs', 'mean_auc']
    titles = ['C-Index (Higher is Better)', 'Integrated Brier Score (Lower is Better)', 'Mean AUC (Higher is Better)']
    
    # Scatter plots
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        
        x = merged_df[f'{metric}_multiclass']
        y = merged_df[f'{metric}_binary']
        
        ax.scatter(x, y, alpha=0.7, s=60)
        
        # Add diagonal line (equal performance)
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Equal Performance')
        
        ax.set_xlabel(f'Multi-class {metric.replace("_", " ").title()}')
        ax.set_ylabel(f'Binary {metric.replace("_", " ").title()}')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add dataset labels for extreme points
        diff = y - x
        if metric == 'ibs':  # For IBS, we want lowest values
            extreme_idx = diff.abs().nlargest(3).index
        else:  # For C-index and AUC, we want highest values
            extreme_idx = diff.abs().nlargest(3).index
        
        for idx in extreme_idx:
            ax.annotate(merged_df.loc[idx, 'dataset'], 
                       (x.loc[idx], y.loc[idx]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.8)
    
    # 4. Distribution comparison
    ax = axes[1, 1]
    
    # Create difference plot
    differences = []
    for metric in metrics:
        diff = merged_df[f'{metric}_binary'] - merged_df[f'{metric}_multiclass']
        if metric == 'ibs':  # For IBS, negative difference means binary is better
            diff = -diff
        differences.extend(diff.tolist())
        
    metric_labels = []
    for metric in metrics:
        metric_labels.extend([metric.replace('_', ' ').title()] * len(merged_df))
    
    diff_df = pd.DataFrame({
        'Difference (Binary - Multi-class)': differences,
        'Metric': metric_labels
    })
    
    sns.boxplot(data=diff_df, x='Metric', y='Difference (Binary - Multi-class)', ax=ax)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax.set_title('Performance Differences Distribution')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('performance_comparison_scatter.png', dpi=300, bbox_inches='tight')
        print("Saved: performance_comparison_scatter.png")
    plt.show()
    
    # 2. Dataset-wise comparison
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Calculate overall score for ranking
    merged_df_viz = merged_df.copy()
    
    # Normalize metrics for fair comparison (0-1 scale)
    for approach in ['binary', 'multiclass']:
        merged_df_viz[f'c_index_norm_{approach}'] = merged_df_viz[f'c_index_{approach}']
        merged_df_viz[f'mean_auc_norm_{approach}'] = merged_df_viz[f'mean_auc_{approach}']
        # For IBS, lower is better, so we invert it
        merged_df_viz[f'ibs_norm_{approach}'] = 1 - (merged_df_viz[f'ibs_{approach}'] - merged_df_viz[f'ibs_{approach}'].min()) / (merged_df_viz[f'ibs_{approach}'].max() - merged_df_viz[f'ibs_{approach}'].min())
    
    # Calculate composite scores
    merged_df_viz['score_binary'] = (merged_df_viz['c_index_norm_binary'] + 
                                   merged_df_viz['mean_auc_norm_binary'] + 
                                   merged_df_viz['ibs_norm_binary']) / 3
    
    merged_df_viz['score_multiclass'] = (merged_df_viz['c_index_norm_multiclass'] + 
                                       merged_df_viz['mean_auc_norm_multiclass'] + 
                                       merged_df_viz['ibs_norm_multiclass']) / 3
    
    merged_df_viz['score_diff'] = merged_df_viz['score_binary'] - merged_df_viz['score_multiclass']
    merged_df_viz_sorted = merged_df_viz.sort_values('score_diff')
    
    colors = ['red' if x < 0 else 'blue' for x in merged_df_viz_sorted['score_diff']]
    
    bars = ax.barh(range(len(merged_df_viz_sorted)), merged_df_viz_sorted['score_diff'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(merged_df_viz_sorted)))
    ax.set_yticklabels(merged_df_viz_sorted['dataset'], fontsize=8)
    ax.set_xlabel('Composite Score Difference (Binary - Multi-class)')
    ax.set_title('Dataset-wise Performance Comparison\n(Positive: Binary Better, Negative: Multi-class Better)')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('dataset_wise_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: dataset_wise_comparison.png")
    plt.show()

def detailed_analysis(merged_df):
    """Provide detailed analysis of specific cases."""
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    # Find datasets where one approach significantly outperforms the other
    metrics = ['c_index', 'ibs', 'mean_auc']
    
    print("\nDatasets where Binary Classification significantly outperforms Multi-class:")
    for metric in metrics:
        diff = merged_df[f'{metric}_binary'] - merged_df[f'{metric}_multiclass']
        if metric == 'ibs':  # For IBS, lower is better
            diff = -diff
        
        # Find top 3 improvements
        top_improvements = diff.nlargest(3)
        print(f"\n{metric.upper()}:")
        for idx in top_improvements.index:
            dataset = merged_df.loc[idx, 'dataset']
            binary_val = merged_df.loc[idx, f'{metric}_binary']
            multi_val = merged_df.loc[idx, f'{metric}_multiclass']
            improvement = top_improvements.loc[idx]
            print(f"  {dataset}: Binary={binary_val:.4f}, Multi-class={multi_val:.4f}, Diff={improvement:.4f}")
    
    print("\nDatasets where Multi-class Classification significantly outperforms Binary:")
    for metric in metrics:
        diff = merged_df[f'{metric}_multiclass'] - merged_df[f'{metric}_binary']
        if metric == 'ibs':  # For IBS, lower is better
            diff = -diff
        
        # Find top 3 improvements
        top_improvements = diff.nlargest(3)
        print(f"\n{metric.upper()}:")
        for idx in top_improvements.index:
            dataset = merged_df.loc[idx, 'dataset']
            binary_val = merged_df.loc[idx, f'{metric}_binary']
            multi_val = merged_df.loc[idx, f'{metric}_multiclass']
            improvement = top_improvements.loc[idx]
            print(f"  {dataset}: Binary={binary_val:.4f}, Multi-class={multi_val:.4f}, Diff={improvement:.4f}")

def main():
    """Main function to run the complete comparison analysis."""
    print("="*60)
    print("TABPFN SURVIVAL ANALYSIS: BINARY vs MULTI-CLASS COMPARISON")
    print("="*60)
    
    # Load data
    binary_df, multiclass_df, merged_df = load_and_clean_data()
    
    if merged_df is None or len(merged_df) == 0:
        print("Error: Could not load or merge data. Please check that both CSV files exist.")
        return
    
    # Run analyses
    summary_statistics(binary_df, multiclass_df, merged_df)
    statistical_results = statistical_comparison(merged_df)
    detailed_analysis(merged_df)
    create_visualizations(merged_df, save_plots=True)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    binary_wins = 0
    multiclass_wins = 0
    
    for metric in ['c_index', 'ibs', 'mean_auc']:
        if statistical_results[metric]['better_approach'] == 'Binary':
            binary_wins += 1
        else:
            multiclass_wins += 1
    
    print(f"\nOverall Performance Summary:")
    print(f"  Binary approach wins:     {binary_wins}/3 metrics")
    print(f"  Multi-class approach wins: {multiclass_wins}/3 metrics")
    
    if binary_wins > multiclass_wins:
        print(f"\n🏆 WINNER: Binary Classification Approach")
    elif multiclass_wins > binary_wins:
        print(f"\n🏆 WINNER: Multi-class (A/B/C/D) Classification Approach")
    else:
        print(f"\n🤝 RESULT: Both approaches perform similarly overall")
    
    print(f"\nKey Insights:")
    print(f"- Total datasets compared: {len(merged_df)}")
    print(f"- Binary approach: Uses hyperparameter tuning for n_bins")
    print(f"- Multi-class approach: Uses fixed evaluation time points (n_eval_times)")
    print(f"- Statistical significance should be interpreted considering p-values")
    print(f"- Effect sizes indicate practical significance of differences")
    print(f"- Individual dataset performance may vary significantly")
    
    # Save comparison results
    merged_df.to_csv('binary_vs_multiclass_comparison.csv', index=False)
    print(f"\nComparison results saved to: binary_vs_multiclass_comparison.csv")

if __name__ == "__main__":
    main()
