import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def finalize_dataset():
    """
    Create the final dataset incorporating all synthetic earthquake types:
    1. Real catalog events
    2. Bootstrap synthetic events
    3. Physics-based synthetic events
    4. Simple synthetic events
    
    This completes the synthetic data generation pipeline.
    """
    print("Finalizing combined earthquake dataset...")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    processed_data_path = os.path.join(project_dir, "data", "processed")
    outputs_path = os.path.join(project_dir, "outputs")
    
    # Ensure output directory exists
    os.makedirs(outputs_path, exist_ok=True)
    
    # Load datasets
    real_file = os.path.join(processed_data_path, "processed_earthquakes.csv")
    bootstrap_file = os.path.join(processed_data_path, "synthetic_bootstrap_v1.csv")
    physics_file = os.path.join(processed_data_path, "synthetic_physics_snapshots_v1.csv")
    simple_file = os.path.join(processed_data_path, "synthetic_simple_v1.csv")
    
    # Load real data
    real_data = pd.read_csv(real_file)
    print(f"Loaded {len(real_data)} real earthquakes")
    
    # Add is_synthetic and sample_weight columns to real data if not present
    if 'is_synthetic' not in real_data.columns:
        real_data['is_synthetic'] = 0
    if 'sample_weight' not in real_data.columns:
        real_data['sample_weight'] = 1.0
    if 'method' not in real_data.columns:
        real_data['method'] = 'real'
    
    # Load synthetic data
    bootstrap_data = pd.read_csv(bootstrap_file)
    physics_data = pd.read_csv(physics_file)
    simple_data = pd.read_csv(simple_file)
    
    print(f"Loaded {len(bootstrap_data)} bootstrap synthetic earthquakes")
    print(f"Loaded {len(physics_data)} physics-based synthetic earthquakes")
    print(f"Loaded {len(simple_data)} simple synthetic earthquakes")
    
    # Combine all datasets
    all_data = pd.concat([real_data, bootstrap_data, physics_data, simple_data], ignore_index=True)
    print(f"Final combined dataset has {len(all_data)} earthquakes")
    
    # Convert time to datetime for time-based partitioning
    all_data['datetime'] = pd.to_datetime(all_data['time'])
    all_data['year'] = all_data['datetime'].dt.year
    
    # Create time-based CV folds (e.g., 3-year blocks)
    fold_years = [
        (2003, 2005), (2006, 2008), (2009, 2011),
        (2012, 2014), (2015, 2017), (2018, 2020),
        (2021, 2025)
    ]
    
    all_data['cv_fold'] = -1
    for fold_idx, (start_year, end_year) in enumerate(fold_years):
        mask = (all_data['year'] >= start_year) & (all_data['year'] <= end_year)
        all_data.loc[mask, 'cv_fold'] = fold_idx
    
    # Print summary of synthetic vs real events by magnitude range
    print("\nFinal data distribution by magnitude range and type:")
    mag_bins = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    all_data['mag_range'] = pd.cut(all_data['magnitude'], mag_bins)
    
    summary = all_data.groupby(['mag_range', 'method']).size().unstack(fill_value=0)
    print(summary)
    
    # Create stacked bar chart of magnitude distribution
    plt.figure(figsize=(12, 8))
    
    # Get unique methods in a sensible order
    methods = ['real', 'bootstrap', 'physics', 'simple']
    methods = [m for m in methods if m in summary.columns]
    
    # Create stacked bar chart
    bottom = np.zeros(len(summary.index))
    
    colors = {'real': 'blue', 'bootstrap': 'green', 'physics': 'red', 'simple': 'purple'}
    
    for method in methods:
        if method in summary.columns:
            plt.bar(range(len(summary.index)), summary[method], bottom=bottom, 
                   label=method.capitalize(), color=colors.get(method, 'gray'), alpha=0.7)
            bottom += summary[method].values
    
    plt.xticks(range(len(summary.index)), [str(idx) for idx in summary.index])
    plt.xlabel('Magnitude Range')
    plt.ylabel('Number of Earthquakes')
    plt.title('Distribution of Earthquakes by Magnitude and Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save magnitude distribution chart
    mag_plot_file = os.path.join(outputs_path, "magnitude_distribution.png")
    plt.savefig(mag_plot_file, dpi=300)
    plt.close()
    print(f"Magnitude distribution chart saved to {mag_plot_file}")
    
    # Create map visualization of all events
    plt.figure(figsize=(12, 8))
    
    # Create a colormap for different data sources
    colors = {'real': 'blue', 'bootstrap': 'green', 'physics': 'red', 'simple': 'purple'}
    markers = {'real': 'o', 'bootstrap': '^', 'physics': '*', 'simple': 'd'}
    
    for method, group in all_data.groupby('method'):
        plt.scatter(group['longitude'], group['latitude'],
                   s=group['magnitude']**2, alpha=0.6, 
                   c=colors.get(method, 'gray'),
                   marker=markers.get(method, 'o'),
                   label=f"{method.capitalize()} ({len(group)})")
    
    # Add labels and legend
    plt.xlabel('Longitude (E)')
    plt.ylabel('Latitude (N)')
    plt.title('Final Combined Earthquake Dataset for Marmara Region')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set plot limits to Marmara region
    plt.xlim(26.0, 30.5)
    plt.ylim(39.5, 41.5)
    
    # Save plot
    plot_file = os.path.join(outputs_path, "final_dataset.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Final dataset visualization saved to {plot_file}")
    
    # Save final dataset
    output_file = os.path.join(processed_data_path, "final_dataset_v1.csv")
    all_data.to_csv(output_file, index=False)
    print(f"Final combined dataset saved to {output_file}")
    
    # Generate summary report
    report_file = os.path.join(outputs_path, "synthetic_data_report.md")
    with open(report_file, 'w') as f:
        f.write("# Synthetic Earthquake Dataset Report\n\n")
        
        f.write("## Dataset Summary\n\n")
        f.write(f"Total events: {len(all_data)}\n")
        f.write(f"Real events: {len(real_data)}\n")
        f.write(f"Synthetic events: {len(all_data) - len(real_data)}\n\n")
        
        f.write("## Synthetic Data Methods\n\n")
        f.write(f"1. **Bootstrap method**: {len(bootstrap_data)} events\n")
        f.write("   - Created by scaling up moderate (M5-6) real events\n")
        f.write("   - Sample weight: 0.3\n\n")
        
        f.write(f"2. **Physics-based method**: {len(physics_data)} events\n")
        f.write("   - Generated using fault geometry and physical parameters\n")
        f.write("   - Based on Gutenberg-Richter relation with b-value = 0.77\n")
        f.write("   - Sample weight: 0.5\n\n")
        
        f.write(f"3. **Simple method**: {len(simple_data)} events\n")
        f.write("   - Created by spatial jittering from template events\n")
        f.write("   - Sample weight: 0.2\n\n")
        
        f.write("## Magnitude Distribution\n\n")
        f.write("```\n")
        f.write(str(summary))
        f.write("\n```\n\n")
        
        f.write("## Time Period\n\n")
        f.write(f"Date range: {all_data['datetime'].min()} to {all_data['datetime'].max()}\n\n")
        
        f.write("## Cross-Validation Folds\n\n")
        f.write("The dataset is divided into time-based CV folds:\n\n")
        for fold_idx, (start_year, end_year) in enumerate(fold_years):
            fold_count = sum(all_data['cv_fold'] == fold_idx)
            f.write(f"- Fold {fold_idx} ({start_year}-{end_year}): {fold_count} events\n")
        
        f.write("\n## Usage for Forecasting\n\n")
        f.write("When training earthquake forecasting models:\n\n")
        f.write("1. Use the `sample_weight` column to give appropriate importance to each event\n")
        f.write("2. Use the `is_synthetic` column to differentiate between real and synthetic events\n")
        f.write("3. Use the `cv_fold` column for time-based cross-validation\n")
        f.write("4. Consider evaluating model performance with and without synthetic data\n")
    
    print(f"Summary report saved to {report_file}")
    
    return all_data

if __name__ == "__main__":
    finalize_dataset()