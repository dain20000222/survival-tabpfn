import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_compare_results():
    # Load the three CSV files
    deephit = pd.read_csv('dynamic_deephit_evaluation.csv')
    tabpfn = pd.read_csv('dynamic_tabpfn_evaluation.csv')
    cox = pd.read_csv('landmark_cox_evaluation.csv')
    
    # Add model names
    deephit['model'] = 'Dynamic DeepHit'
    tabpfn['model'] = 'Dynamic TabPFN'
    cox['model'] = 'Landmark Cox'
    
    # Combine all results
    all_results = pd.concat([deephit, tabpfn, cox], ignore_index=True)
    
    return all_results, deephit, tabpfn, cox

def get_common_datasets(all_results):
    """Find datasets that have results for all three models"""
    models = all_results['model'].unique()
    datasets_by_model = {}
    
    for model in models:
        datasets_by_model[model] = set(all_results[all_results['model'] == model]['dataset'].unique())
    
    # Find intersection of all datasets
    common_datasets = datasets_by_model[models[0]]
    for model in models[1:]:
        common_datasets = common_datasets.intersection(datasets_by_model[model])
    
    print(f"Common datasets across all models: {sorted(common_datasets)}")
    print(f"Total common datasets: {len(common_datasets)}")
    
    return list(common_datasets)

def filter_to_common_data(all_results):
    """Filter results to only include common datasets with matching time points"""
    common_datasets = get_common_datasets(all_results)
    
    # Filter to common datasets
    filtered_results = all_results[all_results['dataset'].isin(common_datasets)].copy()
    
    # For each dataset, find time points that exist for all models
    final_results = []
    
    for dataset in common_datasets:
        dataset_data = filtered_results[filtered_results['dataset'] == dataset]
        
        # Get time points for each model
        time_points_by_model = {}
        for model in dataset_data['model'].unique():
            model_data = dataset_data[dataset_data['model'] == model]
            time_points_by_model[model] = set(model_data['time'].round(6))  # Round to handle floating point precision
        
        # Find common time points
        common_times = time_points_by_model[list(time_points_by_model.keys())[0]]
        for model_times in time_points_by_model.values():
            common_times = common_times.intersection(model_times)
        
        print(f"\n{dataset} - Common time points: {len(common_times)}")
        
        # Filter to common time points
        for time_point in common_times:
            time_data = dataset_data[abs(dataset_data['time'] - time_point) < 1e-5]  # Handle floating point precision
            if len(time_data) == len(dataset_data['model'].unique()):  # All models present
                final_results.append(time_data)
    
    if final_results:
        common_results = pd.concat(final_results, ignore_index=True)
        print(f"\nFinal filtered dataset shape: {common_results.shape}")
        return common_results
    else:
        print("No common data found!")
        return pd.DataFrame()

def compare_by_dataset(all_results):
    """Compare models performance by dataset"""
    print("\n=== Performance Comparison by Dataset ===")
    
    # Pivot table for easier comparison
    pivot_table = all_results.pivot_table(
        index=['dataset', 'time'], 
        columns='model', 
        values='cindex'
    )
    
    print(pivot_table)
    print("\n")
    
    # Calculate mean performance per dataset per model
    dataset_means = all_results.groupby(['dataset', 'model'])['cindex'].mean().unstack()
    print("Mean C-index by Dataset:")
    print(dataset_means)
    print("\n")
    
    # Find best model per dataset
    best_models = dataset_means.idxmax(axis=1)
    print("Best performing model per dataset:")
    for dataset, model in best_models.items():
        print(f"{dataset}: {model} (C-index: {dataset_means.loc[dataset, model]:.4f})")
    
    return pivot_table, dataset_means

def visualize_results(all_results):
    """Create visualizations comparing the models"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison (Common Datasets Only)', fontsize=16, fontweight='bold')
    
    # 1. Box plot of C-index by model
    sns.boxplot(data=all_results, x='model', y='cindex', ax=axes[0,0])
    axes[0,0].set_title('C-index Distribution by Model')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Line plot showing performance over time for each dataset
    datasets = all_results['dataset'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))
    
    for i, dataset in enumerate(datasets):
        dataset_data = all_results[all_results['dataset'] == dataset]
        for model in dataset_data['model'].unique():
            model_data = dataset_data[dataset_data['model'] == model].sort_values('time')
            axes[0,1].plot(model_data['time'], model_data['cindex'], 
                          marker='o', label=f'{dataset}-{model}', 
                          alpha=0.7, color=colors[i])
    
    axes[0,1].set_title('Performance Over Time by Dataset')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('C-index')
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 3. Heatmap of performance by dataset and model
    heatmap_data = all_results.groupby(['dataset', 'model'])['cindex'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', ax=axes[1,0], fmt='.3f')
    axes[1,0].set_title('Mean C-index Heatmap')
    
    # 4. Bar plot comparing average performance
    model_means = all_results.groupby('model')['cindex'].mean()
    bars = axes[1,1].bar(range(len(model_means)), model_means.values)
    axes[1,1].set_title('Average C-index by Model')
    axes[1,1].set_ylabel('Mean C-index')
    axes[1,1].set_xticks(range(len(model_means)))
    axes[1,1].set_xticklabels(model_means.index, rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, model_means.values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def statistical_comparison(common_results):
    """Perform statistical tests to compare models on common datasets"""
    from scipy import stats
    
    print("\n=== Statistical Comparison (Common Datasets Only) ===")
    
    # Create matched pairs for statistical testing
    models = common_results['model'].unique()
    
    # Group by dataset and time to ensure proper pairing
    grouped_data = {}
    for model in models:
        model_data = common_results[common_results['model'] == model]
        # Sort by dataset and time to ensure consistent ordering
        model_data = model_data.sort_values(['dataset', 'time'])
        grouped_data[model] = model_data['cindex'].values
    
    print(f"Sample sizes: {[len(grouped_data[model]) for model in models]}")
    
    # Perform pairwise t-tests
    print("\nPairwise paired t-test results (p-values):")
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            try:
                statistic, p_value = stats.ttest_rel(
                    grouped_data[model1], 
                    grouped_data[model2]
                )
                print(f"{model1} vs {model2}:")
                print(f"  t-statistic: {statistic:.4f}")
                print(f"  p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print(f"  -> Significant difference at α = 0.05")
                else:
                    print(f"  -> No significant difference at α = 0.05")
                print()
            except Exception as e:
                print(f"Error comparing {model1} vs {model2}: {e}")

def generate_summary_report(common_results):
    """Generate a comprehensive summary report"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL COMPARISON REPORT (COMMON DATASETS)")
    print("="*60)
    
    # Overall statistics
    overall_stats = common_results.groupby('model')['cindex'].agg(['count', 'mean', 'std', 'min', 'max'])
    print("\nOverall Performance Statistics:")
    print(overall_stats)
    
    # Best performing model overall
    best_overall = overall_stats['mean'].idxmax()
    print(f"\nBest performing model overall: {best_overall}")
    print(f"Mean C-index: {overall_stats.loc[best_overall, 'mean']:.4f} ± {overall_stats.loc[best_overall, 'std']:.4f}")
    
    # Dataset-specific winners
    print("\nDataset-specific analysis:")
    dataset_performance = {}
    for dataset in common_results['dataset'].unique():
        dataset_data = common_results[common_results['dataset'] == dataset]
        dataset_means = dataset_data.groupby('model')['cindex'].mean()
        winner = dataset_means.idxmax()
        dataset_performance[dataset] = winner
        print(f"{dataset}: {winner} (C-index: {dataset_means[winner]:.4f})")
    
    # Count wins per model
    print("\nModel wins per dataset:")
    win_counts = pd.Series(dataset_performance.values()).value_counts()
    for model, wins in win_counts.items():
        print(f"{model}: {wins} dataset(s)")
    
    return overall_stats, dataset_performance

if __name__ == "__main__":
    # Load data
    all_results, deephit, tabpfn, cox = load_and_compare_results()
    
    print("="*60)
    print("FILTERING TO COMMON DATASETS")
    print("="*60)
    
    # Filter to common datasets and time points
    common_results = filter_to_common_data(all_results)
    
    if len(common_results) == 0:
        print("No common data found. Exiting.")
        exit()
    
    # Perform comparisons on common data
    pivot_table, dataset_means = compare_by_dataset(common_results)
    
    # Create visualizations
    visualize_results(common_results)
    
    # Statistical comparison
    try:
        statistical_comparison(common_results)
    except ImportError:
        print("scipy not available for statistical tests")
    
    # Generate summary report
    generate_summary_report(common_results)