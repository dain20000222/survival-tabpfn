"""
Dataset Statistics Generator for Survival Analysis Datasets

This script generates comprehensive statistics and visualizations for all datasets
in the test folder to be used in thesis documentation.

Author: Generated for survival-tabpfn project
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Any
import os

# Set style for publication-quality figures
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    """Class to analyze survival datasets and generate statistics."""
    
    def __init__(self, test_folder: str, output_folder: str):
        self.test_folder = Path(test_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Dataset statistics storage
        self.dataset_stats = {}
        self.all_datasets_summary = []
        
    def load_dataset(self, filepath: Path) -> pd.DataFrame:
        """Load a single dataset with error handling."""
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def analyze_single_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Analyze a single dataset and return statistics."""
        stats = {
            'name': dataset_name,
            'n_samples': len(df),
            'n_features': len(df.columns),
            'event_rate': 0,
            'censoring_rate': 0,
            'median_time': 0,
            'time_range': (0, 0),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'feature_types': {},
            'time_stats': {},
            'event_stats': {}
        }
        
        # Identify time and event columns (common patterns)
        time_col = None
        event_col = None
        
        for col in df.columns:
            if 'time' in col.lower() or 'duration' in col.lower() or 'survival' in col.lower():
                time_col = col
            elif 'event' in col.lower() or 'status' in col.lower() or 'censor' in col.lower():
                event_col = col
        
        # If not found by name, assume first numeric columns
        if time_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                time_col = numeric_cols[0]
        
        if event_col is None:
            # Look for binary columns
            for col in df.columns:
                if col != time_col and df[col].dtype in ['int64', 'float64']:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                        event_col = col
                        break
        
        # Calculate survival-specific statistics
        if time_col is not None:
            time_data = df[time_col].dropna()
            stats['median_time'] = time_data.median()
            stats['time_range'] = (time_data.min(), time_data.max())
            stats['time_stats'] = {
                'mean': time_data.mean(),
                'std': time_data.std(),
                'q25': time_data.quantile(0.25),
                'q75': time_data.quantile(0.75)
            }
        
        if event_col is not None:
            event_data = df[event_col].dropna()
            event_rate = event_data.mean() if len(event_data) > 0 else 0
            stats['event_rate'] = event_rate
            stats['censoring_rate'] = 1 - event_rate
            stats['event_stats'] = {
                'n_events': event_data.sum(),
                'n_censored': len(event_data) - event_data.sum()
            }
        
        # Feature type analysis
        for col in df.columns:
            if col.startswith('num_'):
                stats['feature_types']['numerical'] = stats['feature_types'].get('numerical', 0) + 1
            elif col.startswith('fac_'):
                stats['feature_types']['categorical'] = stats['feature_types'].get('categorical', 0) + 1
            else:
                # Infer type
                if df[col].dtype in ['int64', 'float64']:
                    unique_ratio = len(df[col].unique()) / len(df)
                    if unique_ratio > 0.1:  # Likely continuous
                        stats['feature_types']['numerical'] = stats['feature_types'].get('numerical', 0) + 1
                    else:  # Likely categorical
                        stats['feature_types']['categorical'] = stats['feature_types'].get('categorical', 0) + 1
                else:
                    stats['feature_types']['categorical'] = stats['feature_types'].get('categorical', 0) + 1
        
        return stats
    
    def generate_dataset_overview_plots(self):
        """Generate overview plots for all datasets."""
        # Prepare data for plotting
        df_summary = pd.DataFrame(self.all_datasets_summary)
        
        # 1. Sample sizes
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sample_sizes = df_summary.set_index('name')['n_samples']  # Keep original order
        
        # Color bars based on dataset size categories
        colors = []
        for size in sample_sizes.values:
            if size < 500:
                colors.append('lightcoral')  # Small
            elif size < 2000:
                colors.append('gold')  # Medium
            else:
                colors.append('lightgreen')  # Large
        
        bars1 = ax1.bar(range(len(sample_sizes)), sample_sizes.values, color=colors)
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Dataset Sample Sizes', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Dataset')
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'sample_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Number of features
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        n_features = df_summary.set_index('name')['n_features']  # Keep original order
        sample_sizes_for_coloring = df_summary.set_index('name')['n_samples']  # Get sample sizes for coloring
        
        # Color bars based on dataset size categories
        colors = []
        for size in sample_sizes_for_coloring.values:
            if size < 500:
                colors.append('lightcoral')  # Small
            elif size < 2000:
                colors.append('gold')  # Medium
            else:
                colors.append('lightgreen')  # Large
        
        bars2 = ax2.bar(range(len(n_features)), n_features.values, color=colors)
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Dataset Feature Counts', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Dataset')
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'feature_counts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Censoring rates
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        censoring_rates = df_summary.set_index('name')['censoring_rate']  # Keep original order
        sample_sizes_for_coloring = df_summary.set_index('name')['n_samples']  # Get sample sizes for coloring
        
        # Color bars based on dataset size categories
        colors = []
        for size in sample_sizes_for_coloring.values:
            if size < 500:
                colors.append('lightcoral')  # Small
            elif size < 2000:
                colors.append('gold')  # Medium
            else:
                colors.append('lightgreen')  # Large
        
        bars3 = ax3.bar(range(len(censoring_rates)), censoring_rates.values, color=colors)
        ax3.set_ylabel('Censoring Rate')
        ax3.set_title('Censoring Rates by Dataset', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        ax3.set_xlabel('Dataset')
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'censoring_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Missing data percentage
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        missing_pct = df_summary['missing_percentage'].sort_values(ascending=False)
        bars4 = ax4.bar(range(len(missing_pct)), missing_pct.values)
        ax4.set_ylabel('Missing Data (%)')
        ax4.set_title('Missing Data by Dataset', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('Dataset')  # Remove x-axis labels
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'missing_data.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Sample size vs Features scatter
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        scatter = ax5.scatter(df_summary['n_features'], df_summary['n_samples'], 
                            c=df_summary['event_rate'], cmap='viridis', s=100, alpha=0.7)
        ax5.set_xlabel('Number of Features')
        ax5.set_ylabel('Number of Samples')
        ax5.set_title('Sample Size vs Features (colored by event rate)', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Event Rate')
        
        # Add dataset labels
        for i, name in enumerate(df_summary['name']):
            ax5.annotate(name, (df_summary.iloc[i]['n_features'], df_summary.iloc[i]['n_samples']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'sample_vs_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Distribution of median survival times
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        median_times = df_summary['median_time'].dropna()
        if len(median_times) > 0:
            ax6.hist(median_times, bins=20, alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Median Survival Time')
            ax6.set_ylabel('Number of Datasets')
            ax6.set_title('Distribution of Median Survival Times', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'median_survival_times.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a combined overview figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Overview Statistics', fontsize=16, fontweight='bold')
        
        # Recreate all plots in the combined figure
        # 1. Sample sizes
        ax1 = axes[0, 0]
        sample_sizes = df_summary['n_samples'].sort_values(ascending=True)
        bars1 = ax1.barh(range(len(sample_sizes)), sample_sizes.values)
        ax1.set_yticks(range(len(sample_sizes)))
        ax1.set_yticklabels(sample_sizes.index, fontsize=8)
        ax1.set_xlabel('Number of Samples')
        ax1.set_title('Dataset Sample Sizes')
        ax1.grid(True, alpha=0.3)
        
        # 2. Number of features
        ax2 = axes[0, 1]
        n_features = df_summary['n_features'].sort_values(ascending=True)
        bars2 = ax2.barh(range(len(n_features)), n_features.values)
        ax2.set_yticks(range(len(n_features)))
        ax2.set_yticklabels(n_features.index, fontsize=8)
        ax2.set_xlabel('Number of Features')
        ax2.set_title('Dataset Feature Counts')
        ax2.grid(True, alpha=0.3)
        
        # 3. Event rates
        ax3 = axes[0, 2]
        censoring_rates = df_summary['censoring_rate'].sort_values(ascending=False)
        bars3 = ax3.bar(range(len(censoring_rates)), censoring_rates.values)
        ax3.set_xticks(range(len(censoring_rates)))
        ax3.set_xticklabels(censoring_rates.index, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Censoring Rate')
        ax3.set_title('Censoring Rates by Dataset')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. Missing data percentage
        ax4 = axes[1, 0]
        missing_pct = df_summary['missing_percentage'].sort_values(ascending=False)
        bars4 = ax4.bar(range(len(missing_pct)), missing_pct.values)
        ax4.set_xticks(range(len(missing_pct)))
        ax4.set_xticklabels(missing_pct.index, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Missing Data (%)')
        ax4.set_title('Missing Data by Dataset')
        ax4.grid(True, alpha=0.3)
        
        # 5. Sample size vs Features scatter
        ax5 = axes[1, 1]
        scatter = ax5.scatter(df_summary['n_features'], df_summary['n_samples'], 
                            c=df_summary['event_rate'], cmap='viridis', s=50, alpha=0.7)
        ax5.set_xlabel('Number of Features')
        ax5.set_ylabel('Number of Samples')
        ax5.set_title('Sample Size vs Features')
        ax5.grid(True, alpha=0.3)
        
        # 6. Distribution of median survival times
        ax6 = axes[1, 2]
        median_times = df_summary['median_time'].dropna()
        if len(median_times) > 0:
            ax6.hist(median_times, bins=20, alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Median Survival Time')
            ax6.set_ylabel('Number of Datasets')
            ax6.set_title('Distribution of Median Survival Times')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_detailed_statistics_table(self):
        """Generate a detailed statistics table."""
        df_summary = pd.DataFrame(self.all_datasets_summary)
        
        # Create a comprehensive table
        table_data = []
        for _, row in df_summary.iterrows():
            table_data.append({
                'Dataset': row['name'],
                'Samples': row['n_samples'],
                'Features': row['n_features'],
                'Event Rate': f"{row['event_rate']:.3f}",
                'Median Time': f"{row['median_time']:.1f}" if not pd.isna(row['median_time']) else 'N/A',
                'Missing (%)': f"{row['missing_percentage']:.1f}",
                'Numerical Features': row['feature_types'].get('numerical', 0),
                'Categorical Features': row['feature_types'].get('categorical', 0)
            })
        
        table_df = pd.DataFrame(table_data)
        
        # Create figure for the table
        fig, ax = plt.subplots(figsize=(16, max(8, len(table_df) * 0.3)))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=table_df.values,
                        colLabels=table_df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.08, 0.08, 0.1, 0.1, 0.1, 0.12, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_df) + 1):
            for j in range(len(table_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f1f1f2')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title('Comprehensive Dataset Statistics Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.output_folder / 'dataset_statistics_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save as CSV for reference
        table_df.to_csv(self.output_folder / 'dataset_statistics.csv', index=False)
        
    def generate_survival_characteristics_plots(self):
        """Generate plots focusing on survival analysis characteristics."""
        df_summary = pd.DataFrame(self.all_datasets_summary)
        
        # 1. Event vs Censoring rates
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        datasets = df_summary['name']
        event_rates = df_summary['event_rate']
        censoring_rates = df_summary['censoring_rate']
        
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, event_rates, width, label='Event Rate', alpha=0.8)
        bars2 = ax1.bar(x + width/2, censoring_rates, width, label='Censoring Rate', alpha=0.8)
        
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Rate')
        ax1.set_title('Event vs Censoring Rates', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'event_vs_censoring.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Time range analysis
        fig2, ax2 = plt.subplots(figsize=(10, 12))
        time_ranges = []
        names = []
        for _, row in df_summary.iterrows():
            if not pd.isna(row['median_time']):
                time_ranges.append(row['time_range'])
                names.append(row['name'])
        
        if time_ranges:
            min_times = [tr[0] for tr in time_ranges]
            max_times = [tr[1] for tr in time_ranges]
            
            x_pos = np.arange(len(names))
            ax2.bar(x_pos, max_times, alpha=0.6, label='Max Time')
            ax2.bar(x_pos, min_times, alpha=0.8, label='Min Time')
            ax2.set_ylabel('Time')
            ax2.set_title('Time Range by Dataset', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel('Dataset')  # Remove x-axis labels
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'time_ranges.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Sample size categories
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        small = sum(df_summary['n_samples'] < 500)
        medium = sum((df_summary['n_samples'] >= 500) & (df_summary['n_samples'] < 2000))
        large = sum(df_summary['n_samples'] >= 2000)
        
        sizes = [small, medium, large]
        labels = ['Small (<500)', 'Medium (500-2000)', 'Large (≥2000)']
        colors = ['lightcoral', 'gold', 'lightgreen']
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Distribution of Dataset Sizes', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'dataset_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Feature type distribution
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        total_numerical = sum([row['feature_types'].get('numerical', 0) for _, row in df_summary.iterrows()])
        total_categorical = sum([row['feature_types'].get('categorical', 0) for _, row in df_summary.iterrows()])
        
        feature_types = [total_numerical, total_categorical]
        labels = ['Numerical', 'Categorical']
        colors = ['skyblue', 'lightpink']
        
        wedges, texts, autotexts = ax4.pie(feature_types, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Overall Feature Type Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'feature_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a combined survival characteristics figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Survival Analysis Characteristics', fontsize=16, fontweight='bold')
        
        # Recreate all plots in combined figure
        # 1. Event vs Censoring rates
        ax1 = axes[0, 0]
        datasets = df_summary['name']
        event_rates = df_summary['event_rate']
        censoring_rates = df_summary['censoring_rate']
        
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, event_rates, width, label='Event Rate', alpha=0.8)
        bars2 = ax1.bar(x + width/2, censoring_rates, width, label='Censoring Rate', alpha=0.8)
        
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Rate')
        ax1.set_title('Event vs Censoring Rates')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Time range analysis
        ax2 = axes[0, 1]
        time_ranges = []
        names = []
        for _, row in df_summary.iterrows():
            if not pd.isna(row['median_time']):
                time_ranges.append(row['time_range'])
                names.append(row['name'])
        
        if time_ranges:
            min_times = [tr[0] for tr in time_ranges]
            max_times = [tr[1] for tr in time_ranges]
            
            y_pos = np.arange(len(names))
            ax2.barh(y_pos, max_times, alpha=0.6, label='Max Time')
            ax2.barh(y_pos, min_times, alpha=0.8, label='Min Time')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(names, fontsize=8)
            ax2.set_xlabel('Time')
            ax2.set_title('Time Range by Dataset')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Sample size categories
        ax3 = axes[1, 0]
        small = sum(df_summary['n_samples'] < 500)
        medium = sum((df_summary['n_samples'] >= 500) & (df_summary['n_samples'] < 2000))
        large = sum(df_summary['n_samples'] >= 2000)
        
        sizes = [small, medium, large]
        labels = ['Small (<500)', 'Medium (500-2000)', 'Large (≥2000)']
        colors = ['lightcoral', 'gold', 'lightgreen']
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Distribution of Dataset Sizes')
        
        # 4. Feature type distribution
        ax4 = axes[1, 1]
        total_numerical = sum([row['feature_types'].get('numerical', 0) for _, row in df_summary.iterrows()])
        total_categorical = sum([row['feature_types'].get('categorical', 0) for _, row in df_summary.iterrows()])
        
        feature_types = [total_numerical, total_categorical]
        labels = ['Numerical', 'Categorical']
        colors = ['skyblue', 'lightpink']
        
        wedges, texts, autotexts = ax4.pie(feature_types, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Overall Feature Type Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'survival_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_individual_dataset_summaries(self):
        """Generate individual summary cards for each dataset."""
        n_datasets = len(self.all_datasets_summary)
        n_cols = 4
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Individual Dataset Summary Cards', fontsize=18, fontweight='bold')
        
        for idx, dataset in enumerate(self.all_datasets_summary):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Create a summary card
            ax.text(0.5, 0.9, dataset['name'], ha='center', va='top', fontsize=14, 
                   fontweight='bold', transform=ax.transAxes)
            
            summary_text = f"""
Samples: {dataset['n_samples']:,}
Features: {dataset['n_features']}
Event Rate: {dataset['event_rate']:.3f}
Censoring: {dataset['censoring_rate']:.3f}
Median Time: {dataset['median_time']:.1f}
Missing Data: {dataset['missing_percentage']:.1f}%

Feature Types:
  Numerical: {dataset['feature_types'].get('numerical', 0)}
  Categorical: {dataset['feature_types'].get('categorical', 0)}
            """
            
            ax.text(0.05, 0.75, summary_text, ha='left', va='top', fontsize=10, 
                   transform=ax.transAxes, family='monospace')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Add a border
            rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=2, 
                               edgecolor='gray', facecolor='none', transform=ax.transAxes)
            ax.add_patch(rect)
        
        # Hide unused subplots
        for idx in range(n_datasets, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_folder / 'individual_dataset_summaries.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting dataset analysis...")
        
        # Get all CSV files
        csv_files = list(self.test_folder.glob("*.csv"))
        print(f"Found {len(csv_files)} datasets to analyze")
        
        # Analyze each dataset
        for csv_file in csv_files:
            print(f"Analyzing {csv_file.name}...")
            df = self.load_dataset(csv_file)
            
            if df is not None:
                dataset_name = csv_file.stem
                stats = self.analyze_single_dataset(df, dataset_name)
                self.dataset_stats[dataset_name] = stats
                self.all_datasets_summary.append(stats)
        
        print(f"Successfully analyzed {len(self.all_datasets_summary)} datasets")
        
        # Generate visualizations
        print("Generating overview plots...")
        self.generate_dataset_overview_plots()
        
        print("Generating statistics table...")
        self.generate_detailed_statistics_table()
        
        print("Generating survival characteristics plots...")
        self.generate_survival_characteristics_plots()
        
        print("Generating individual dataset summaries...")
        self.generate_individual_dataset_summaries()
        
        print(f"Analysis complete! Results saved to {self.output_folder}")
        
        # Print summary to console
        self.print_summary()
    
    def print_summary(self):
        """Print a summary of the analysis to console."""
        print("\n" + "="*80)
        print("DATASET ANALYSIS SUMMARY")
        print("="*80)
        
        df_summary = pd.DataFrame(self.all_datasets_summary)
        
        print(f"Total datasets analyzed: {len(df_summary)}")
        print(f"Total samples across all datasets: {df_summary['n_samples'].sum():,}")
        print(f"Average dataset size: {df_summary['n_samples'].mean():.0f} samples")
        print(f"Average number of features: {df_summary['n_features'].mean():.1f}")
        print(f"Average event rate: {df_summary['event_rate'].mean():.3f}")
        print(f"Average missing data: {df_summary['missing_percentage'].mean():.1f}%")
        
        print(f"\nLargest dataset: {df_summary.loc[df_summary['n_samples'].idxmax(), 'name']} "
              f"({df_summary['n_samples'].max():,} samples)")
        print(f"Smallest dataset: {df_summary.loc[df_summary['n_samples'].idxmin(), 'name']} "
              f"({df_summary['n_samples'].min():,} samples)")
        
        print(f"\nHighest event rate: {df_summary.loc[df_summary['event_rate'].idxmax(), 'name']} "
              f"({df_summary['event_rate'].max():.3f})")
        print(f"Lowest event rate: {df_summary.loc[df_summary['event_rate'].idxmin(), 'name']} "
              f"({df_summary['event_rate'].min():.3f})")
        
        print("\n" + "="*80)


def main():
    """Main function to run the analysis."""
    # Set up paths
    current_dir = Path(__file__).parent
    test_folder = current_dir / "test"
    output_folder = current_dir / "figures" / "dataset"
    
    # Create analyzer and run analysis
    analyzer = DatasetAnalyzer(test_folder, output_folder)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
