import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def create_performance_comparison_table(df):
    """Create a table showing win/loss statistics for TabPFN vs each baseline."""
    baseline_models = {
        'rsf': 'RSF',
        'cph': 'CoxPH', 
        'dh': 'DeepHit',
        'ds': 'DeepSurv'
    }
    metrics = ['c_index', 'ibs', 'mean_auc']
    
    comparison_results = []
    
    for metric in metrics:
        higher_is_better = metric != 'ibs'  # IBS: lower is better
        
        for baseline_key, baseline_name in baseline_models.items():
            wins = 0
            losses = 0
            ties = 0
            
            for idx, row in df.iterrows():
                tabpfn_val = row[f'tabpfn_{metric}']
                baseline_val = row[f'{baseline_key}_{metric}']
                
                if pd.isna(tabpfn_val) or pd.isna(baseline_val):
                    continue
                
                if higher_is_better:
                    if tabpfn_val > baseline_val:
                        wins += 1
                    elif tabpfn_val < baseline_val:
                        losses += 1
                    else:
                        ties += 1
                else:  # Lower is better (IBS)
                    if tabpfn_val < baseline_val:
                        wins += 1
                    elif tabpfn_val > baseline_val:
                        losses += 1
                    else:
                        ties += 1
            
            total = wins + losses + ties
            win_rate = wins / total if total > 0 else 0
            
            comparison_results.append({
                'metric': metric,
                'baseline_model': baseline_name,
                'wins': wins,
                'losses': losses,
                'ties': ties,
                'total': total,
                'win_rate': round(win_rate, 3)
            })
    
    return pd.DataFrame(comparison_results)

def plot_average_ranks(avg_ranks_df):
    """Create visualization of average ranks and save each metric separately."""
    metrics = ['c_index', 'ibs', 'mean_auc']
    metric_titles = ['C-Index', 'IBS', 'Mean AUC']
    
    # Define model order
    model_order = ['CoxPH', 'RSF', 'DeepSurv', 'DeepHit', 'TabPFN']
    
    # Create individual plots for each metric
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        data = avg_ranks_df[avg_ranks_df['metric'] == metric].copy()
        data['model'] = pd.Categorical(data['model'], categories=model_order, ordered=True)
        data = data.sort_values('model')
        
        # Create colors - emphasize TabPFN in red, others in blue
        colors = ['skyblue' if model != 'TabPFN' else 'red' for model in data['model']]
        
        bars = ax.bar(data['model'], data['avg_rank'], color=colors)
        ax.set_title(f'{title} - Average Rank', fontsize=16, fontweight='bold')
        ax.set_ylabel('Average Rank', fontsize=14)
        ax.set_xlabel('Model', fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # Add value labels on bars
        for bar, avg_rank in zip(bars, data['avg_rank']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{avg_rank:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add horizontal line at rank 3 (middle rank for 5 models)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.7, label='Middle Rank (3.0)')
        ax.legend(fontsize=12)
        ax.set_ylim(0, 5.5)
        
        # Emphasize TabPFN bar with border
        for j, model in enumerate(data['model']):
            if model == 'TabPFN':
                bars[j].set_edgecolor('darkred')
                bars[j].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(f'./figures/{metric}_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_win_rates(comparison_df):
    """Create visualization of win rates."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['c_index', 'ibs', 'mean_auc']
    metric_titles = ['C-Index', 'IBS', 'Mean AUC']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        data = comparison_df[comparison_df['metric'] == metric]
        
        bars = axes[i].bar(data['baseline_model'], data['win_rate'], 
                          color=['green' if rate >= 0.5 else 'red' for rate in data['win_rate']])
        axes[i].set_title(f'TabPFN vs Baselines\n{title}')
        axes[i].set_ylabel('TabPFN Win Rate')
        axes[i].set_xlabel('Baseline Model')
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, win_rate in zip(bars, data['win_rate']):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{win_rate:.3f}', ha='center', va='bottom')
        
        # Add horizontal line at 50%
        axes[i].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% Win Rate')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('tabpfn_win_rates.png', dpi=300, bbox_inches='tight')
    plt.show()

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
    eval_metrics_df.to_csv(f'./results/evaluation_results.csv', index=False)
    ranking_table_df.to_csv(f'./results/ranking_results.csv', index=False)

    print("\n" + "="*60)
    print("AVERAGE RANKS BY MODEL AND METRIC")
    print("="*60)
    
    for metric in ['c_index', 'ibs', 'mean_auc']:
        print(f"\n{metric.upper()} Rankings:")
        metric_data = avg_ranks[avg_ranks['metric'] == metric].sort_values('avg_rank')
        print(metric_data.to_string(index=False))
    
    print("\n" + "="*60)
    print("WIN/LOSS ANALYSIS: TabPFN vs Baselines")
    print("="*60)
    
    comparison_df = create_performance_comparison_table(df)
    
    for metric in ['c_index', 'ibs', 'mean_auc']:
        print(f"\n{metric.upper()}:")
        metric_data = comparison_df[comparison_df['metric'] == metric]
        print(metric_data[['baseline_model', 'wins', 'losses', 'ties', 'win_rate']].to_string(index=False))
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall ranking summary
    overall_avg_rank = avg_ranks.groupby('model')['avg_rank'].mean().sort_values()
    print("\nOverall Average Rank Across All Metrics:")
    for model, rank in overall_avg_rank.items():
        print(f"{model:10s}: {rank:.3f}")
    
    # TabPFN overall win rate
    overall_win_rate = comparison_df['win_rate'].mean()
    print(f"\nTabPFN Overall Win Rate: {overall_win_rate:.3f}")
    
    # Best performing model for each metric
    print("\nBest Model by Metric (lowest average rank):")
    for metric in ['c_index', 'ibs', 'mean_auc']:
        best_model = avg_ranks[avg_ranks['metric'] == metric].loc[avg_ranks[avg_ranks['metric'] == metric]['avg_rank'].idxmin()]
        print(f"{metric:12s}: {best_model['model']} (avg rank: {best_model['avg_rank']:.3f})")
    
    print("\nCreating visualizations...")
    plot_average_ranks(avg_ranks)
    
    # Save additional analysis files
    avg_ranks.to_csv(f'./results/average_rankings.csv', index=False)
    comparison_df.to_csv(f'./results/win_loss_analysis.csv', index=False)

if __name__ == "__main__":
    main()
