import numpy as np
import pandas as pd

def get_hko_idw_weights(grid_csv_path, target_lat=22.3019, target_lon=114.1742, buffer=0.015, power=2):
    """Pre-calculates the local 50x50 grid indices and IDW weights for the HKO station."""
    grid_df = pd.read_csv(grid_csv_path)
    lat_vals = grid_df['latitude_50'].dropna().values
    lon_vals = grid_df['longitude_50'].dropna().values
    
    # Find grid indices within the buffer zone
    lat_idx = np.where((lat_vals >= target_lat - buffer) & (lat_vals <= target_lat + buffer))[0]
    lon_idx = np.where((lon_vals >= target_lon - buffer) & (lon_vals <= target_lon + buffer))[0]
    
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError("Target coordinates are outside the 50x50 grid!")
        
    # Build normalized weight matrix
    weights = np.zeros((len(lat_idx), len(lon_idx)))
    for ii, i in enumerate(lat_idx):
        for jj, j in enumerate(lon_idx):
            dist = np.sqrt((lat_vals[i] - target_lat)**2 + (lon_vals[j] - target_lon)**2)
            weights[ii, jj] = 1.0 / (dist + 1e-12)**2
            
    weights /= np.sum(weights) # Normalize so weights sum to 1
    return lat_idx, lon_idx, weights

def is_valid_csi(frame, threshold=0.01):
    return np.nanmean(frame) > threshold