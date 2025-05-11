import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

def analyze_earthquake_catalog():
    """
    Analyze the Marmara region earthquake catalog:
    - Check for completeness
    - Analyze magnitude distribution
    - Calculate b-value (Gutenberg-Richter relation)
    - Analyze spatial and temporal patterns
    """
    print("Analyzing earthquake catalog data...")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    processed_data_path = os.path.join(project_dir, "data", "processed")
    outputs_path = os.path.join(project_dir, "outputs")
    
    # Ensure directories exist
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(outputs_path, exist_ok=True)
    
    # Load earthquake data
    earthquake_file = os.path.join(processed_data_path, "processed_earthquakes.csv")
    eq_data = pd.read_csv(earthquake_file)
    
    print(f"Loaded {len(eq_data)} earthquakes from {earthquake_file}")
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Date range: {eq_data['time'].min()} to {eq_data['time'].max()}")
    print(f"Magnitude range: {eq_data['magnitude'].min():.1f} to {eq_data['magnitude'].max():.1f}")
    print(f"Depth range: {eq_data['depth_km'].min():.1f} to {eq_data['depth_km'].max():.1f} km")
    
    # Convert time to datetime for time-based analysis
    eq_data['datetime'] = pd.to_datetime(eq_data['time'])
    eq_data['year'] = eq_data['datetime'].dt.year
    
    # Count earthquakes by year
    yearly_counts = eq_data.groupby('year').size()
    print("\n=== Yearly Earthquake Counts ===")
    print(yearly_counts)
    
    # Filter to include only M3.0+ events for analysis
    m3plus_data = eq_data[eq_data['magnitude'] >= 3.0]
    print(f"\nFound {len(m3plus_data)} events with magnitude >= 3.0")
    
    # Count by magnitude bins
    mag_bins = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    m3plus_data['mag_bin'] = pd.cut(m3plus_data['magnitude'], mag_bins)
    mag_counts = m3plus_data.groupby('mag_bin').size()
    print("\n=== Magnitude Distribution ===")
    print(mag_counts)
    
    # Estimate b-value using maximum likelihood method
    # b-value is an important parameter in Gutenberg-Richter relation
    def estimate_b_value(magnitudes, m_min):
        # Filter magnitudes above or equal to m_min
        mags = magnitudes[magnitudes >= m_min]
        if len(mags) < 2:
            return None, None
        
        # Maximum likelihood estimate of b-value
        mean_mag = np.mean(mags)
        b_value = np.log10(np.e) / (mean_mag - m_min)
        
        # Uncertainty in b-value
        uncertainty = 2.3 * b_value / np.sqrt(len(mags))
        
        return b_value, uncertainty
    
    # Calculate b-value for different minimum magnitudes
    print("\n=== Gutenberg-Richter b-value Estimates ===")
    for m_min in [3.0, 3.5, 4.0, 4.5]:
        b_val, uncertainty = estimate_b_value(m3plus_data['magnitude'].values, m_min)
        if b_val:
            print(f"For M >= {m_min}: b-value = {b_val:.2f} +/- {uncertainty:.2f}")
    
    # Check for catalog completeness by plotting cumulative frequency vs magnitude
    def plot_magnitude_completeness():
        plt.figure(figsize=(10, 6))
        
        # Create logarithmic bins for magnitude
        min_mag = np.floor(m3plus_data['magnitude'].min())
        max_mag = np.ceil(m3plus_data['magnitude'].max())
        mag_bins = np.arange(min_mag, max_mag + 0.1, 0.1)
        
        # Count earthquakes in each bin
        hist, bin_edges = np.histogram(m3plus_data['magnitude'], bins=mag_bins)
        
        # Calculate cumulative counts (reversed)
        cumulative_counts = np.cumsum(hist[::-1])[::-1]
        
        # Plot cumulative counts vs magnitude (log scale)
        plt.semilogy(bin_edges[:-1], cumulative_counts, 'bo-', linewidth=1.5)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Magnitude', fontsize=12)
        plt.ylabel('Cumulative Number', fontsize=12)
        plt.title('Gutenberg-Richter Relation for Marmara Region', fontsize=14)
        
        # Estimate Mc (magnitude of completeness) as point where curve deviates from log-linear
        # This is a simplified approach - actual method would use more sophisticated techniques
        plt.axvline(x=3.5, color='r', linestyle='--', label='Estimated Mc = 3.5')
        plt.legend()
        
        # Save the plot
        plot_file = os.path.join(outputs_path, "magnitude_completeness.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Magnitude completeness plot saved to {plot_file}")
    
    # Plot spatial distribution of earthquakes
    def plot_spatial_distribution():
        plt.figure(figsize=(12, 8))
        
        # Plot earthquakes with size proportional to magnitude
        magnitudes = m3plus_data['magnitude'].values
        sizes = [(m**2) * 5 for m in magnitudes]  # Scale for visibility
        sc = plt.scatter(m3plus_data['longitude'], m3plus_data['latitude'], s=sizes, 
                         c=m3plus_data['depth_km'], cmap='viridis_r', alpha=0.6)
        
        # Add colorbar for depth
        cbar = plt.colorbar(sc)
        cbar.set_label('Depth (km)')
        
        # Load fault segments
        fault_file = os.path.join(project_dir, "data", "raw", "marmara_faults.csv")
        fault_data = pd.read_csv(fault_file)
        
        # Plot fault segments
        for _, fault in fault_data.iterrows():
            coords = fault['coordinates'].split(';')
            x_coords = []
            y_coords = []
            
            for coord in coords:
                lon, lat = coord.split(',')
                x_coords.append(float(lon))
                y_coords.append(float(lat))
            
            plt.plot(x_coords, y_coords, 'r-', linewidth=2, label=fault['name'] if len(x_coords) > 0 else None)
        
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        
        # Set plot limits to Marmara region
        plt.xlim(26.0, 30.5)
        plt.ylim(39.5, 41.5)
        
        # Add borders of the Marmara Sea (simplified)
        plt.axhspan(40.3, 41.1, xmin=0.1, xmax=0.9, alpha=0.1, color='blue', label='Marmara Sea')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Longitude (E)')
        plt.ylabel('Latitude (N)')
        plt.title('Spatial Distribution of Earthquakes in Marmara Region (2003-2025)')
        
        # Save the plot
        plot_file = os.path.join(outputs_path, "spatial_distribution.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Spatial distribution plot saved to {plot_file}")
    
    # Plot temporal distribution (earthquakes over time)
    def plot_temporal_distribution():
        plt.figure(figsize=(12, 6))
        
        # Group by year and magnitude bins
        m3plus_data['year_month'] = m3plus_data['datetime'].dt.to_period('M')
        monthly_counts = m3plus_data.groupby('year_month').size()
        
        # Convert period index to datetime for plotting
        dates = [pd.to_datetime(str(period)) for period in monthly_counts.index]
        
        plt.bar(dates, monthly_counts.values, width=20, alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Date')
        plt.ylabel('Number of Earthquakes (M >= 3.0)')
        plt.title('Monthly Earthquake Counts in Marmara Region (2003-2025)')
        
        # Add markers for major events (M >= 5.5)
        major_eq = m3plus_data[m3plus_data['magnitude'] >= 5.5]
        for _, eq in major_eq.iterrows():
            plt.axvline(x=eq['datetime'], color='r', linestyle='--', alpha=0.7)
            plt.text(eq['datetime'], 0.9*max(monthly_counts.values), 
                     f"M{eq['magnitude']:.1f}", rotation=90, verticalalignment='top')
        
        # Save the plot
        plot_file = os.path.join(outputs_path, "temporal_distribution.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Temporal distribution plot saved to {plot_file}")
    
    # Create the plots
    plot_magnitude_completeness()
    plot_spatial_distribution()
    plot_temporal_distribution()
    
    # Create a processed version of the earthquake catalog with additional fields
    processed_eq = m3plus_data.copy()
    
    # Add some derived fields useful for modeling
    processed_eq['log_energy'] = 1.5 * processed_eq['magnitude'] + 4.8  # Approximate energy release in log10(joules)
    
    # Save processed data
    processed_file = os.path.join(processed_data_path, "processed_earthquakes_analyzed.csv")
    processed_eq.to_csv(processed_file, index=False)
    print(f"Processed earthquake data saved to {processed_file}")
    
    # Calculate b-value for future use in synthetic generation
    b_val, _ = estimate_b_value(m3plus_data['magnitude'].values, 3.5)
    
    # Save b-value to a file for use in other scripts
    b_value_file = os.path.join(processed_data_path, "b_value.txt")
    with open(b_value_file, 'w') as f:
        f.write(str(b_val))
    print(f"B-value {b_val:.2f} saved to {b_value_file}")
    
    return b_val

if __name__ == "__main__":
    b_value = analyze_earthquake_catalog()
    print(f"\nEstimated b-value for catalog: {b_value:.2f}")
    print("This b-value will be used for synthetic earthquake generation.")