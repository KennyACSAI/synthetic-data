import os
import pandas as pd

def download_fault_data():
    """
    Create simplified fault data for the Marmara region
    """
    print("Creating simplified fault data for Marmara region...")
    
    # Get the absolute path to the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    raw_data_path = os.path.join(project_dir, "data", "raw")
    
    print(f"Script directory: {script_dir}")
    print(f"Project directory: {project_dir}")
    print(f"Raw data path: {raw_data_path}")
    
    # Create raw data directory if it doesn't exist
    os.makedirs(raw_data_path, exist_ok=True)
    
    # Main segments of the North Anatolian Fault in Marmara region
    # These are approximate and simplified locations
    fault_segments = [
        {
            "segment_id": "NAF_Western",
            "name": "North Anatolian Fault (Western Segment)",
            "coordinates": "26.7,40.4;27.2,40.5;27.7,40.7",
            "strike": 275,  # approximate strike in degrees
            "dip": 85,      # approximate dip in degrees
            "rake": 180,    # approximate rake in degrees 
            "length_km": 85,
            "seismogenic_thickness_km": 15
        },
        {
            "segment_id": "NAF_Central",
            "name": "North Anatolian Fault (Central Marmara)",
            "coordinates": "27.7,40.7;28.3,40.8;28.9,40.9",
            "strike": 270,
            "dip": 80,
            "rake": 175,
            "length_km": 70,
            "seismogenic_thickness_km": 17
        },
        {
            "segment_id": "NAF_Eastern",
            "name": "North Anatolian Fault (Eastern Marmara)",
            "coordinates": "28.9,40.9;29.5,40.7;30.0,40.6",
            "strike": 265,
            "dip": 75,
            "rake": 170,
            "length_km": 65,
            "seismogenic_thickness_km": 15
        },
        {
            "segment_id": "NAF_Southern", 
            "name": "North Anatolian Fault (Southern Branch)",
            "coordinates": "27.5,40.5;28.2,40.4;28.9,40.3;29.5,40.2",
            "strike": 260,
            "dip": 80,
            "rake": 165,
            "length_km": 120,
            "seismogenic_thickness_km": 12
        }
    ]
    
    # Create a DataFrame
    fault_data = pd.DataFrame(fault_segments)
    
    # Save to CSV
    fault_csv_path = os.path.join(raw_data_path, "marmara_faults.csv")
    try:
        fault_data.to_csv(fault_csv_path, index=False)
        
        # Verify file was created and has content
        if os.path.exists(fault_csv_path):
            file_size = os.path.getsize(fault_csv_path)
            print(f"Fault data saved to {fault_csv_path} (Size: {file_size} bytes)")
        else:
            print(f"Error: File not found at {fault_csv_path} after saving")
            
        print(f"Created simplified fault data with {len(fault_segments)} segments")
    except Exception as e:
        print(f"Error saving fault data: {str(e)}")

if __name__ == "__main__":
    download_fault_data()