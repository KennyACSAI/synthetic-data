# -*- coding: utf-8 -*-
# CHANGE CHECK GITHUB
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from datetime import datetime

def assemble_dataset():
    """
    Assemble the final dataset by combining:
    1. Real earthquake catalog
    2. Bootstrap synthetic events
    3. Physics-based synthetic events
    
    Also creates time-based CV folds for model training.
    """
    print("Assembling final earthquake dataset...")
    
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
    print(f"Loaded {len(bootstrap_data)} bootstrap synthetic earthquakes")
    print(f"Loaded {len(physics_data)} physics-based synthetic earthquakes")
    
    # Combine all datasets
    all_data = pd.concat([real_data, bootstrap_data, physics_data], ignore_index=True)
    print(f"Combined dataset has {len(all_data)} earthquakes")
    
    # Verify all required columns exist
    required_columns = ['id', 'time', 'magnitude', 'longitude', 'latitude', 
                       'depth_km', 'is_synthetic', 'sample_weight', 'method']
    
    missing_columns = [col for col in required_columns if col not in all_data.columns]
    if missing_columns:
        print(f"Warning: Missing columns in combined dataset: {missing_columns}")
    
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
    
    print("Created CV folds based on time blocks:")
    for fold_idx, (start_year, end_year) in enumerate(fold_years):
        fold_count = sum(all_data['cv_fold'] == fold_idx)
        print(f"  Fold {fold_idx} ({start_year}-{end_year}): {fold_count} events")
    
    # Print summary of synthetic vs real events by magnitude range
    print("\nData distribution by magnitude range and type:")
    mag_bins = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    all_data['mag_range'] = pd.cut(all_data['magnitude'], mag_bins)
    
    summary = all_data.groupby(['mag_range', 'method']).size().unstack(fill_value=0)
    print(summary)
    
    # Visualize the combined dataset
    plt.figure(figsize=(12, 8))
    
    # Create a colormap for different data sources
    colors = {'real': 'blue', 'bootstrap': 'green', 'physics': 'red'}
    markers = {'real': 'o', 'bootstrap': '^', 'physics': '*'}
    
    for method, group in all_data.groupby('method'):
        plt.scatter(group['longitude'], group['latitude'],
                   s=group['magnitude']**2, alpha=0.6, 
                   c=colors.get(method, 'gray'),
                   marker=markers.get(method, 'o'),
                   label=f"{method.capitalize()} ({len(group)})")
    
    # Add labels and legend
    plt.xlabel('Longitude (E)')
    plt.ylabel('Latitude (N)')
    plt.title('Combined Earthquake Dataset for Marmara Region')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set plot limits to Marmara region
    plt.xlim(26.0, 30.5)
    plt.ylim(39.5, 41.5)
    
    # Save plot
    plot_file = os.path.join(outputs_path, "combined_dataset.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Combined dataset visualization saved to {plot_file}")
    
    # Save final dataset
    output_file = os.path.join(processed_data_path, "combined_dataset_v1.csv")
    all_data.to_csv(output_file, index=False)
    print(f"Combined dataset saved to {output_file}")
    
    # Generate dataset metrics
    metrics = {
        "total_earthquakes": len(all_data),
        "real_events": sum(all_data['is_synthetic'] == 0),
        "synthetic_events": sum(all_data['is_synthetic'] == 1),
        "bootstrap_synthetics": sum(all_data['method'] == 'bootstrap'),
        "physics_synthetics": sum(all_data['method'] == 'physics'),
        "magnitude_min": all_data['magnitude'].min(),
        "magnitude_max": all_data['magnitude'].max(),
        "real_magnitude_max": real_data['magnitude'].max(),
        "synthetic_magnitude_min": min(
            bootstrap_data['magnitude'].min(),
            physics_data['magnitude'].min()
        ),
        "date_range": f"{all_data['datetime'].min()} to {all_data['datetime'].max()}",
        "cv_folds": len(fold_years)
    }
    
    # Save metrics
    metrics_file = os.path.join(outputs_path, "dataset_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("=== Dataset Metrics ===\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Dataset metrics saved to {metrics_file}")
    
    return all_data

if __name__ == "__main__":
    assemble_dataset()