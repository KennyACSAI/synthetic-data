import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import math

def generate_physics_snapshots():
    """
    Generate physics-based synthetic M>=6.5 earthquakes using the Okada model.
    This implements the "Physics snapshot generator" from the roadmap.
    
    Target: ~100 synthetic Mw 6.5-7.3 ruptures
    """
    print("Generating physics-based synthetic earthquakes...")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    raw_data_path = os.path.join(project_dir, "data", "raw")
    processed_data_path = os.path.join(project_dir, "data", "processed")
    outputs_path = os.path.join(project_dir, "outputs")
    
    # Ensure directories exist
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(outputs_path, exist_ok=True)
    
    # Load fault data
    fault_file = os.path.join(raw_data_path, "marmara_faults.csv")
    fault_data = pd.read_csv(fault_file)
    print(f"Loaded {len(fault_data)} fault segments from {fault_file}")
    
    # Load earthquake data (for reference)
    eq_file = os.path.join(processed_data_path, "processed_earthquakes.csv")
    eq_data = pd.read_csv(eq_file)
    
    # Try to read b-value from file
    b_value_file = os.path.join(processed_data_path, "b_value.txt")
    if os.path.exists(b_value_file):
        try:
            with open(b_value_file, 'r') as f:
                b_value = float(f.read().strip())
            print(f"Using b-value = {b_value} from catalog analysis file")
        except:
            # Fall back to default if file can't be read
            b_value = 1.15  # Value from your analysis
            print(f"Using fallback b-value = {b_value} (could not read from file)")
    else:
        # Fall back to default if file doesn't exist
        b_value = 1.15  # Value from your analysis
        print(f"Using fallback b-value = {b_value} (b_value file not found)")
    
    # Number of synthetic events to generate
    n_synthetics = 20  # Target is ~100, but start with fewer for testing
    
    # Step 1: Generate magnitudes from truncated GR distribution
    magnitudes = []
    for _ in range(n_synthetics):
        # Using the formula from the roadmap: mag = 6.5 - np.log10(1 - np.random.rand()) / b_value
        r = random.random()  # Random number between 0 and 1
        mag = 6.5 - np.log10(1 - r) / b_value
        # Cap at M7.3 as per roadmap
        mag = min(mag, 7.3)
        magnitudes.append(mag)
    
    print(f"Generated {len(magnitudes)} synthetic magnitudes from truncated GR distribution")
    print(f"Magnitude range: {min(magnitudes):.2f} to {max(magnitudes):.2f}")
    
    # Step 2: Assign to fault segments and calculate rupture dimensions
    synthetic_events = []
    
    # Create time spacing for events (6-12 months apart as per roadmap)
    start_date = datetime(2003, 1, 1)
    end_date = datetime(2025, 1, 1)
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    months_per_event = total_months / n_synthetics
    
    # Create a function to check if segment length is sufficient for rupture length
    def segment_can_host_rupture(segment, rupture_length):
        # Parse coordinates from string format
        coords_str = segment['coordinates']
        coord_pairs = coords_str.split(';')
        
        # Convert to float coordinates
        coords = []
        for pair in coord_pairs:
            lon, lat = map(float, pair.split(','))
            coords.append((lon, lat))
        
        # Calculate total segment length (simplified)
        segment_length = 0
        for i in range(len(coords) - 1):
            # Haversine formula for distance between two points
            lat1, lon1 = coords[i][1], coords[i][0]
            lat2, lon2 = coords[i+1][1], coords[i+1][0]
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371  # Radius of Earth in kilometers
            distance = c * r
            
            segment_length += distance
        
        return segment_length >= rupture_length
    
    # For each magnitude, calculate rupture dimensions and assign to fault segment
    for i, magnitude in enumerate(magnitudes):
        # Calculate rupture area using M0 = 10^(1.5*Mw + 9.1)
        # log10(A_km2) = Mw – 4.0
        rupture_area = 10 ** (magnitude - 4.0)
        
        # Using 2:1 aspect ratio as suggested in the roadmap
        rupture_length = math.sqrt(rupture_area * 2)  # in km
        rupture_width = rupture_length / 2  # in km
        
        # Find a suitable fault segment
        # Start with random selection, then check if it can host the rupture
        suitable_segments = []
        for _, segment in fault_data.iterrows():
            if segment_can_host_rupture(segment, rupture_length):
                suitable_segments.append(segment)
        
        if not suitable_segments:
            print(f"Warning: No suitable segment found for M{magnitude:.1f} event (needed {rupture_length:.1f} km)")
            continue
        
        # Select a random suitable segment
        selected_segment = random.choice(suitable_segments)
        
        # Calculate seismic moment
        moment = 10 ** (1.5 * magnitude + 9.1)  # in Nm
        
        # Calculate average slip
        mu = 3.2e10  # Shear modulus in Pa
        avg_slip = moment / (mu * rupture_area * 1e6)  # Convert km² to m²
        
        # Add log-normal scatter to slip (±30% as per roadmap)
        slip_variation = np.random.lognormal(0, 0.3)
        slip = avg_slip * slip_variation
        
        # Generate origin time (spaced 6-12 months apart)
        months_offset = random.uniform(6, 12) * i
        origin_time = start_date + timedelta(days=months_offset * 30)  # Approximate
        
        # Create the synthetic event record
        synthetic = {
            'id': f"SYN_PHYS_{i+1:03d}",
            'time': origin_time.strftime("%Y-%m-%d %H:%M:%S"),
            'magnitude': magnitude,
            'depth_km': random.uniform(5, 15),  # Typical seismogenic depth in region
            'segment_id': selected_segment['segment_id'],
            'rupture_length_km': rupture_length,
            'rupture_width_km': rupture_width,
            'rupture_area_km2': rupture_area,
            'avg_slip_m': avg_slip,
            'is_synthetic': 1,
            'sample_weight': 0.5,  # As per roadmap
            'method': 'physics',
            'strike': selected_segment['strike'],
            'dip': selected_segment['dip'],
            'rake': selected_segment['rake']
        }
        
        # Parse coordinates to get event location (simplified)
        # This just places the event at a random point along the segment
        coords_str = selected_segment['coordinates']
        coord_pairs = coords_str.split(';')
        
        # Choose a random point along the segment
        rand_index = random.randint(0, len(coord_pairs) - 1)
        lon, lat = map(float, coord_pairs[rand_index].split(','))
        
        # Add 3D jitter (±5 km as per roadmap)
        # Convert ~5km to degrees (approximate)
        degree_jitter = 0.045  # ~5km at this latitude
        lon += random.uniform(-degree_jitter, degree_jitter)
        lat += random.uniform(-degree_jitter, degree_jitter)
        
        synthetic['longitude'] = lon
        synthetic['latitude'] = lat
        
        # Add to list of synthetic events
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
        output_file = os.path.join(processed_data_path, "synthetic_physics_snapshots_v1.csv")
        synthetics_df.to_csv(output_file, index=False)
        print(f"Generated {len(synthetics_df)} physics-based synthetic events")
        print(f"Synthetic data saved to {output_file}")
        
        # Visualize original and synthetic events
        plt.figure(figsize=(12, 8))
        
        # Plot fault segments
        for _, fault in fault_data.iterrows():
            coords = fault['coordinates'].split(';')
            x_coords = []
            y_coords = []
            
            for coord in coords:
                lon, lat = coord.split(',')
                x_coords.append(float(lon))
                y_coords.append(float(lat))
            
            plt.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.7)
        
        # Plot original events
        plt.scatter(eq_data['longitude'], eq_data['latitude'], 
                   s=eq_data['magnitude']**2, alpha=0.3, c='blue',
                   label='Original Events')
        
        # Plot bootstrap synthetic events
        try:
            bootstrap_file = os.path.join(processed_data_path, "synthetic_bootstrap_v1.csv")
            bootstrap_data = pd.read_csv(bootstrap_file)
            plt.scatter(bootstrap_data['longitude'], bootstrap_data['latitude'],
                       s=bootstrap_data['magnitude']**2, alpha=0.6, c='green',
                       marker='o', label='Bootstrap Synthetics')
        except:
            print("Bootstrap synthetic data not found for comparison plot.")
        
        # Plot physics-based synthetic events
        plt.scatter(synthetics_df['longitude'], synthetics_df['latitude'],
                   s=synthetics_df['magnitude']**2, alpha=0.8, c='red',
                   marker='*', label='Physics-based Synthetics')
        
        # Add labels and legend
        plt.xlabel('Longitude (E)')
        plt.ylabel('Latitude (N)')
        plt.title('Original vs. Synthetic Earthquakes in Marmara Region')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set plot limits to Marmara region
        plt.xlim(26.0, 30.5)
        plt.ylim(39.5, 41.5)
        
        # Save plot
        plot_file = os.path.join(outputs_path, "physics_synthetics.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Comparison plot saved to {plot_file}")
        
        return synthetics_df
    else:
        print("No synthetic events were generated.")
        return None

if __name__ == "__main__":
    generate_physics_snapshots()