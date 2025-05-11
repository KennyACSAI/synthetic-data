import os
import pandas as pd
import numpy as np
from datetime import datetime

def prepare_marmara_catalog():
    """
    Prepare the pre-processed Marmara catalog for synthetic generation.
    """
    # Path to your processed data file
    processed_file = os.path.join("data", "processed", "marmara_catalog_processed.csv")
    
    print(f"Preparing Marmara catalog from: {processed_file}")
    
    # Get project directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    processed_data_path = os.path.join(project_dir, "data", "processed")
    
    # Ensure the path is correct
    if not os.path.exists(processed_file):
        processed_file = os.path.join(project_dir, "data", "processed", "marmara_catalog_processed.csv")
        if not os.path.exists(processed_file):
            print(f"ERROR: Could not find the processed catalog at: {processed_file}")
            return None
    
    # Load the processed data
    raw_data = pd.read_csv(processed_file)
    print(f"Loaded earthquake catalog with {len(raw_data)} events")
    
    # Display column names to verify format
    print(f"Columns in the data: {list(raw_data.columns)}")
    
    # Create a new DataFrame with expected column names
    processed_data = pd.DataFrame()
    
    # Map columns based on expected structure
    if 'datetime' in raw_data.columns:
        processed_data['time'] = raw_data['datetime']
    elif 'time' in raw_data.columns:
        processed_data['time'] = raw_data['time']
    else:
        # Try to combine date and time if they exist separately
        if 'date' in raw_data.columns and 'time' in raw_data.columns:
            processed_data['time'] = raw_data['date'] + ' ' + raw_data['time']
        else:
            print("ERROR: Could not find datetime, time, or date columns")
            return None
    
    # Map other columns
    if 'magnitude' in raw_data.columns:
        processed_data['magnitude'] = raw_data['magnitude']
    else:
        print("ERROR: Could not find magnitude column")
        return None
        
    if 'longitude' in raw_data.columns:
        processed_data['longitude'] = raw_data['longitude']
    else:
        print("ERROR: Could not find longitude column")
        return None
        
    if 'latitude' in raw_data.columns:
        processed_data['latitude'] = raw_data['latitude']
    else:
        print("ERROR: Could not find latitude column")
        return None
        
    if 'depth' in raw_data.columns:
        processed_data['depth_km'] = raw_data['depth']
    elif 'depth_km' in raw_data.columns:
        processed_data['depth_km'] = raw_data['depth_km']
    else:
        print("WARNING: Could not find depth column, using default depth of 10 km")
        processed_data['depth_km'] = 10.0
    
    # Add ID column if not present
    if 'id' not in raw_data.columns:
        processed_data['id'] = [f"EQ_{i+1:06d}" for i in range(len(raw_data))]
    else:
        processed_data['id'] = raw_data['id']
    
    # Add columns needed for synthetic generation
    processed_data['is_synthetic'] = 0
    processed_data['sample_weight'] = 1.0
    processed_data['method'] = 'real'
    
    # Calculate log energy (needed for bootstrap method)
    processed_data['log_energy'] = 1.5 * processed_data['magnitude'] + 4.8
    
    # Convert time to datetime for analysis
    try:
        processed_data['datetime'] = pd.to_datetime(processed_data['time'])
        processed_data['year'] = processed_data['datetime'].dt.year
    except Exception as e:
        print(f"Warning: Could not parse time column: {e}")
    
    # Save the data in the expected format
    output_file = os.path.join(processed_data_path, "processed_earthquakes.csv")
    processed_data.to_csv(output_file, index=False)
    print(f"Saved prepared data to: {output_file}")
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Total events: {len(processed_data)}")
    print(f"Magnitude range: {processed_data['magnitude'].min():.1f} to {processed_data['magnitude'].max():.1f}")
    
    if 'datetime' in processed_data.columns:
        print(f"Time range: {processed_data['datetime'].min()} to {processed_data['datetime'].max()}")
    
    print(f"Depth range: {processed_data['depth_km'].min():.1f} to {processed_data['depth_km'].max():.1f} km")
    
    # Count events by magnitude
    print("\nEvents by magnitude range:")
    mag_bins = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    mag_counts = pd.cut(processed_data['magnitude'], mag_bins).value_counts().sort_index()
    for i, count in enumerate(mag_counts):
        if i < len(mag_bins) - 1:
            print(f"M{mag_bins[i]:.1f}-{mag_bins[i+1]:.1f}: {count}")
    
    return processed_data

if __name__ == "__main__":
    prepare_marmara_catalog()