import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_simple_synthetics():
    """
    Generate additional synthetic earthquakes using a simple jittering approach.
    """
    print("Generating simple synthetic earthquakes...")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    processed_data_path = os.path.join(project_dir, "data", "processed")
    outputs_path = os.path.join(project_dir, "outputs")
    
    # Ensure output directory exists
    os.makedirs(outputs_path, exist_ok=True)
    
    # Load the combined dataset
    combined_file = os.path.join(processed_data_path, "combined_dataset_v1.csv")
    all_data = pd.read_csv(combined_file)
    print(f"Loaded {len(all_data)} events from combined dataset")
    
    # Get high-magnitude template events (>= 5.0)
    templates = all_data[all_data['magnitude'] >= 5.0].copy()
    print(f"Using {len(templates)} high-magnitude events as templates")
    
    # Create synthetic events
    n_synthetic = 10
    synthetic_events = []
    
    for i in range(n_synthetic):
        # Sample a random template event
        template = templates.sample(1).iloc[0]
        
        # Generate magnitude (6.5-7.3 range)
        magnitude = random.uniform(6.5, 7.3)
        
        # Jitter location slightly but stay in the Marmara region
        # Add random noise (±0.3 degrees, about 30km)
        longitude = template['longitude'] + random.uniform(-0.3, 0.3)
        latitude = template['latitude'] + random.uniform(-0.2, 0.2)
        
        # Ensure we stay in the Marmara region
        longitude = max(26.0, min(30.5, longitude))
        latitude = max(39.5, min(41.5, latitude))
        
        # Generate depth (typical seismogenic depth range)
        depth = random.uniform(5, 20)
        
        # Create synthetic event
        synthetic = {
            'id': f"SYN_SIMPLE_{i+1:03d}",
            'longitude': longitude,
            'latitude': latitude,
            'depth_km': depth,
            'magnitude': magnitude,
            'is_synthetic': 1,
            'sample_weight': 0.2,
            'method': 'simple'
        }
        
        # Assign random timestamp between 2003-2025
        year = random.randint(2003, 2025)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        timestamp = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
        synthetic['time'] = timestamp
        
        # Calculate rupture dimensions (approximate)
        # Log(A) = M - 4.0
        area = 10 ** (magnitude - 4.0)
        length = np.sqrt(area * 2)  # Using 2:1 aspect ratio
        width = length / 2
        
        synthetic['rupture_length_km'] = length
        synthetic['rupture_width_km'] = width
        synthetic['rupture_area_km2'] = area
        
        synthetic_events.append(synthetic)
        print(f"Created synthetic event {i+1}/{n_synthetic} with M{magnitude:.1f}")
    
    # Create DataFrame
    synthetics_df = pd.DataFrame(synthetic_events)
    
    # Save synthetic events
    output_file = os.path.join(processed_data_path, "synthetic_simple_v1.csv")
    synthetics_df.to_csv(output_file, index=False)
    print(f"Generated {len(synthetics_df)} simple synthetic events")
    print(f"Simple synthetic data saved to {output_file}")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot original data
    real_data = all_data[all_data['is_synthetic'] == 0]
    plt.scatter(real_data['longitude'], real_data['latitude'],
               s=real_data['magnitude']**1.5, alpha=0.3, c='blue',
               label=f"Real Events ({len(real_data)})")
    
    # Plot existing synthetics
    bootstrap_data = all_data[all_data['method'] == 'bootstrap']
    physics_data = all_data[all_data['method'] == 'physics']
    
    if not bootstrap_data.empty:
        plt.scatter(bootstrap_data['longitude'], bootstrap_data['latitude'],
                   s=bootstrap_data['magnitude']**1.5, alpha=0.4, c='green',
                   label=f"Bootstrap ({len(bootstrap_data)})")
    
    if not physics_data.empty:
        plt.scatter(physics_data['longitude'], physics_data['latitude'],
                   s=physics_data['magnitude']**1.5, alpha=0.4, c='red',
                   label=f"Physics ({len(physics_data)})")
    
    # Plot simple synthetics
    plt.scatter(synthetics_df['longitude'], synthetics_df['latitude'],
               s=synthetics_df['magnitude']**2, alpha=0.9, c='purple',
               marker='d', label=f"Simple ({len(synthetics_df)})")
    
    # Add labels and legend
    plt.xlabel('Longitude (E)')
    plt.ylabel('Latitude (N)')
    plt.title('Simple Synthetic Earthquakes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set plot limits to Marmara region
    plt.xlim(26.0, 30.5)
    plt.ylim(39.5, 41.5)
    
    # Save plot
    plot_file = os.path.join(outputs_path, "simple_synthetics.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Simple synthetics visualization saved to {plot_file}")
    
    return synthetics_df

if __name__ == "__main__":
    generate_simple_synthetics()