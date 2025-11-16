import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_compare_results():
    # Load the four CSV files
    deephit = pd.read_csv('dynamic_deephit_evaluation.csv')
    tabpfn = pd.read_csv('dynamic_tabpfn_evaluation.csv')
    tabpfn_eval = pd.read_csv('dynamic_tabpfn.csv')
    cox = pd.read_csv('landmark_cox_evaluation.csv')
    
    # Add model names
    deephit['model'] = 'Dynamic DeepHit'
    tabpfn['model'] = 'Dynamic TabPFN'
    tabpfn_eval['model'] = 'Dynamic TabPFN (Eval Time)'
    cox['model'] = 'Landmark Cox'
    
    # Combine all results
    all_results = pd.concat([deephit, tabpfn, tabpfn_eval, cox], ignore_index=True)
    
    return all_results, deephit, tabpfn, tabpfn_eval, cox

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
        
        # Get time points for each model with more precise rounding
        time_points_by_model = {}
        for model in dataset_data['model'].unique():
            model_data = dataset_data[dataset_data['model'] == model]
            # Round to 3 decimal places to handle floating point precision
            time_points_by_model[model] = set(model_data['time'].round(3))
        
        print(f"\n{dataset} - Time points by model:")
        for model, times in time_points_by_model.items():
            print(f"  {model}: {sorted(times)} (count: {len(times)})")
        
        # Find common time points
        common_times = time_points_by_model[list(time_points_by_model.keys())[0]]
        for model_times in time_points_by_model.values():
            common_times = common_times.intersection(model_times)
        
        print(f"  Common time points: {sorted(common_times)} (count: {len(common_times)})")
        
        # Filter to common time points with stricter matching
        for time_point in common_times:
            time_data_list = []
            for model in dataset_data['model'].unique():
                model_data = dataset_data[dataset_data['model'] == model]
                # Use tighter tolerance for matching
                matching_rows = model_data[abs(model_data['time'] - time_point) < 0.001]
                if len(matching_rows) > 0:
                    time_data_list.append(matching_rows)
            
            # Only include if all models have data for this time point
            if len(time_data_list) == len(dataset_data['model'].unique()):
                combined_time_data = pd.concat(time_data_list, ignore_index=True)
                if len(combined_time_data) == len(dataset_data['model'].unique()):
                    final_results.append(combined_time_data)
    
    if final_results:
        common_results = pd.concat(final_results, ignore_index=True)
        print(f"\nFinal filtered dataset shape: {common_results.shape}")
        
        # Verify the final results
        print("\nFinal verification - Time points per dataset:")
        for dataset in common_results['dataset'].unique():
            dataset_final = common_results[common_results['dataset'] == dataset]
            time_points = sorted(dataset_final['time'].round(3).unique())
            print(f"{dataset}: {time_points} (count: {len(time_points)})")
        
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
    
    # Create separate figures for the 4 main comparison plots
    create_main_comparison_plots(all_results)
    
    # Create dataset-specific time-dependent plots
    create_dataset_time_plots(all_results)

def create_main_comparison_plots(all_results):
    """Create the 4 main comparison plots as separate files"""
    
    # 1. Box plot of C-index by model
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=all_results, x='model', y='cindex')
    plt.title('C-index Distribution by Model', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cindex_distribution_by_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Line plot showing performance over time for each dataset
    plt.figure(figsize=(12, 8))
    datasets = all_results['dataset'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))
    
    for i, dataset in enumerate(datasets):
        dataset_data = all_results[all_results['dataset'] == dataset]
        for model in dataset_data['model'].unique():
            model_data = dataset_data[dataset_data['model'] == model].sort_values('time')
            plt.plot(model_data['time'], model_data['cindex'], 
                    marker='o', label=f'{dataset}-{model}', 
                    alpha=0.7, color=colors[i])
    
    plt.title('Performance Over Time by Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('C-index')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig('performance_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Heatmap of performance by dataset and model
    plt.figure(figsize=(10, 6))
    heatmap_data = all_results.groupby(['dataset', 'model'])['cindex'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Mean C-index Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mean_cindex_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Bar plot comparing average performance
    plt.figure(figsize=(8, 6))
    model_means = all_results.groupby('model')['cindex'].mean()
    bars = plt.bar(range(len(model_means)), model_means.values)
    plt.title('Average C-index by Model', fontsize=14, fontweight='bold')
    plt.ylabel('Mean C-index')
    plt.xticks(range(len(model_means)), model_means.index, rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, model_means.values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('average_cindex_by_model.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_dataset_time_plots(all_results):
    """Create separate bar charts for each dataset showing C-index at each time point"""
    
    datasets = all_results['dataset'].unique()
    models = all_results['model'].unique()
    
    # Create a color palette for models
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    color_dict = dict(zip(models, colors))
    
    for dataset in datasets:
        dataset_data = all_results[all_results['dataset'] == dataset]
        time_points = sorted(dataset_data['time'].unique())
        
        # Calculate figure size based on number of time points
        fig_width = max(8, len(time_points) * 1.2)
        plt.figure(figsize=(fig_width, 6))
        
        # Prepare data for grouped bar chart
        x = np.arange(len(time_points))
        width = 0.25  # Width of bars
        
        # Plot bars for each model
        for i, model in enumerate(models):
            model_data = dataset_data[dataset_data['model'] == model]
            cindex_values = []
            
            for time_point in time_points:
                time_data = model_data[abs(model_data['time'] - time_point) < 1e-5]
                if len(time_data) > 0:
                    cindex_values.append(time_data['cindex'].iloc[0])
                else:
                    cindex_values.append(np.nan)
            
            # Plot bars with offset
            offset = (i - len(models)/2 + 0.5) * width
            bars = plt.bar(x + offset, cindex_values, width, 
                          label=model, color=color_dict[model], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, cindex_values):
                if not np.isnan(value):
                    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.title(f'C-index by Time Point - {dataset}', fontsize=14, fontweight='bold')
        plt.xlabel('Time Point')
        plt.ylabel('C-index')
        plt.xticks(x, [f'{t:.2f}' for t in time_points], rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1)  # C-index ranges from 0 to 1
        
        # Save the plot
        filename = f'cindex_timepoints_{dataset.replace(" ", "_").replace("/", "_")}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved time-dependent plot for {dataset}: {filename}")

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
    all_results, deephit, tabpfn, tabpfn_eval, cox = load_and_compare_results()
    
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