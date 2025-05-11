import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

def generate_bootstrap_synthetics():
    """
    Generate synthetic M>=6.5 earthquakes by scaling up real moderate-sized (5<=M<6) events.
    This implements the "quick-win bootstrapping" approach from the roadmap.
    """
    print("Generating bootstrap synthetic earthquakes...")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    processed_data_path = os.path.join(project_dir, "data", "processed")
    outputs_path = os.path.join(project_dir, "outputs")
    
    # Ensure outputs directory exists
    os.makedirs(outputs_path, exist_ok=True)
    
    # Load processed earthquake data
    eq_file = os.path.join(processed_data_path, "processed_earthquakes.csv")
    eq_data = pd.read_csv(eq_file)
    print(f"Loaded {len(eq_data)} earthquakes from {eq_file}")
    
    # Convert time to datetime for time-based filtering
    eq_data['datetime'] = pd.to_datetime(eq_data['time'])
    
    # Step 1: Filter moderate events (5<=M<6) to use as templates
    template_events = eq_data[(eq_data['magnitude'] >= 5.0) & (eq_data['magnitude'] < 6.0)].copy()
    print(f"Found {len(template_events)} template events (5.0 <= M < 6.0)")
    
    # Step 2: Create synthetic events by scaling magnitude and adjusting parameters
    synthetic_events = []
    
    for _, event in template_events.iterrows():
        # Define parameters for the synthetic event
        original_mag = event['magnitude']
        
        # Generate a random magnitude increase (target M6.5-7.3)
        magnitude_increase = random.uniform(1.5, 2.3)
        new_mag = original_mag + magnitude_increase
        
        # Copy the original event
        synthetic = event.copy()
        
        # Scale magnitude-dependent parameters according to roadmap
        
        # 1. Adjust magnitude
        synthetic['magnitude'] = new_mag
        synthetic['id'] = f"SYN_{event['id']}"
        
        # 2. Scale Benioff strain ∝ 10^(1.5 ΔM)
        benioff_scaling = 10 ** (1.5 * magnitude_increase)
        synthetic['log_energy'] = event['log_energy'] + 1.5 * magnitude_increase
        
        # 3. Scale stress drop using Leonard (2010) scaling
        # Simplified: stress drop ~ (10^(0.5*magnitude_increase))
        stress_scaling = 10 ** (0.5 * magnitude_increase)
        
        # 4. Calculate rupture dimensions using Wells-Coppersmith relations
        # Log(L) = -2.44 + 0.59M (subsurface rupture length in km)
        original_length = 10 ** (-2.44 + 0.59 * original_mag)
        new_length = 10 ** (-2.44 + 0.59 * new_mag)
        
        # Log(W) = -1.01 + 0.32M (subsurface rupture width in km)
        original_width = 10 ** (-1.01 + 0.32 * original_mag)
        new_width = 10 ** (-1.01 + 0.32 * new_mag)
        
        # Add rupture dimensions to synthetic event
        synthetic['rupture_length_km'] = new_length
        synthetic['rupture_width_km'] = new_width
        synthetic['rupture_area_km2'] = new_length * new_width
        
        # Keep location (grid_id) and original time
        # As per roadmap: "Keep time, grid_id, background stress unchanged"
        
        # Tag as synthetic with weight=0.3
        synthetic['is_synthetic'] = 1
        synthetic['sample_weight'] = 0.3
        synthetic['method'] = 'bootstrap'
        
        # Add to synthetic events list
        synthetic_events.append(synthetic)
    
    # Create a DataFrame with synthetic events
    if synthetic_events:
        synthetics_df = pd.DataFrame(synthetic_events)
        
        # Perform sanity checks (as per roadmap)
        invalid_events = synthetics_df[
            (synthetics_df['depth_km'] < 0) | 
            (synthetics_df['rupture_length_km'] > 200)
        ]
        
        if not invalid_events.empty:
            print(f"Warning: Found {len(invalid_events)} events with unphysical values. Removing them.")
            synthetics_df = synthetics_df[~synthetics_df.index.isin(invalid_events.index)]
        
        # Save synthetic events
        output_file = os.path.join(processed_data_path, "synthetic_bootstrap_v1.csv")
        synthetics_df.to_csv(output_file, index=False)
        print(f"Generated {len(synthetics_df)} synthetic events with magnitudes {synthetics_df['magnitude'].min():.1f}-{synthetics_df['magnitude'].max():.1f}")
        print(f"Synthetic data saved to {output_file}")
        
        # Visualize original and synthetic events
        plt.figure(figsize=(12, 8))
        
        # Plot original events
        plt.scatter(eq_data['longitude'], eq_data['latitude'], 
                   s=eq_data['magnitude']**2, alpha=0.5, c='blue',
                   label='Original Events')
        
        # Plot synthetic events
        plt.scatter(synthetics_df['longitude'], synthetics_df['latitude'],
                   s=synthetics_df['magnitude']**2, alpha=0.8, c='red',
                   marker='*', label='Synthetic Events (Bootstrap)')
        
        # Add labels and legend
        plt.xlabel('Longitude (E)')
        plt.ylabel('Latitude (N)')
        plt.title('Original vs. Bootstrap Synthetic Earthquakes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = os.path.join(outputs_path, "bootstrap_synthetics.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Comparison plot saved to {plot_file}")
        
        return synthetics_df
    else:
        print("No synthetic events were generated. Check if there are template events available.")
        return None

if __name__ == "__main__":
    generate_bootstrap_synthetics()