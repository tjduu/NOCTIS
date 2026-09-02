import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from nocitis.forecast.utils import get_hko_idw_weights, is_valid_csi,


def run_rolling_oflow_short_horizon(
    ref_npz_path, csi_npz_path, output_csv, lat_idx, lon_idx, weights,
    pre_steps=96, future_steps=16  # 16 steps = exactly 4 hours of prediction!
):
    print(f"\n🚀 Running 24h Rolling Init OF (4h Horizon) -> {os.path.basename(output_csv)}")
    
    ref_data_dict = np.load(ref_npz_path)
    time_utc = ref_data_dict["time"]
    ref_data = ref_data_dict["data"] 
    
    csi_data_dict = np.load(csi_npz_path)
    csi_data = csi_data_dict["data"] 
    csi_chan = 0 if csi_data.shape[1] == 1 else 2
    _, _, H, W = csi_data.shape 
    
    total_steps = pre_steps + future_steps
    dt_hk = pd.DatetimeIndex(time_utc + np.timedelta64(8, "h"))
    a, b = int(len(time_utc) * (2/3)), len(time_utc) - total_steps
    
    # Init at EVERY hour (0-23)
    hourly_candidates = np.where(dt_hk.minute == 0)[0] 
    hourly_candidates = hourly_candidates[(hourly_candidates >= a) & (hourly_candidates + total_steps <= b)]
    
    valid_cumsum = np.concatenate(([0], np.cumsum(np.diff(time_utc).astype("timedelta64[m]").astype(np.int64) == 15)))
    starts = [s for s in hourly_candidates if valid_cumsum[s + total_steps - 1] - valid_cumsum[s] == total_steps - 1]

    results = []
    for s in tqdm(starts, desc="Advecting Clouds"):
        past_csi = csi_data[s : s + pre_steps, csi_chan, :, :]
        true_ghi = ref_data[s + pre_steps : s + total_steps, 0, :, :]
        true_csghi = ref_data[s + pre_steps : s + total_steps, 1, :, :]
        
        flow = None
        base_frame = past_csi[-1]
        valid_t = pre_steps - 1
        
        for t in range(pre_steps - 1, 0, -1):
            if is_valid_csi(past_csi[t]) and is_valid_csi(past_csi[t-1]):
                prev_c = (np.clip(past_csi[t-1], 0, 1) * 255).astype(np.uint8)
                curr_c = (np.clip(past_csi[t], 0, 1) * 255).astype(np.uint8)
                flow = cv2.calcOpticalFlowFarneback(prev_c, curr_c, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                base_frame = past_csi[t]
                valid_t = t
                break
                
        if flow is None:
            flow = np.zeros((H, W, 2), dtype=np.float32)
            for t in range(pre_steps - 1, -1, -1):
                if is_valid_csi(past_csi[t]):
                    base_frame = past_csi[t]
                    break
                    
        h_grid, w_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        gap = (pre_steps - 1) - valid_t
        
        init_hkt = time_utc[s + pre_steps - 1] + np.timedelta64(8, "h")
        t_hkt_future = time_utc[s + pre_steps : s + total_steps] + np.timedelta64(8, "h")
        
        for k in range(1, future_steps + 1):
            flow_multi = k + gap
            map_x = (w_grid + flow_multi * flow[..., 0]).astype(np.float32)
            map_y = (h_grid + flow_multi * flow[..., 1]).astype(np.float32)
            
            p_csi_raw = cv2.remap(base_frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            p_csi_raw = np.clip(p_csi_raw, 0.0, 1.0)
            
            idx_k = k - 1
            p_csi_val = np.sum(p_csi_raw[lat_idx[:, None], lon_idx] * weights)
            t_ghi_val = np.sum(true_ghi[idx_k, lat_idx[:, None], lon_idx] * weights)
            c_ghi_val = np.sum(true_csghi[idx_k, lat_idx[:, None], lon_idx] * weights)
            
            p_ghi_val = (p_csi_val * c_ghi_val) if (c_ghi_val * 4.0 > 10.0) else 0.0
            
            results.append({
                "Init_Time_HKT": init_hkt,
                "Time_HKT": t_hkt_future[idx_k],
                "Pred_CSI": p_csi_val,
                "ClearSky_GHI": c_ghi_val,
                "Pred_GHI": p_ghi_val,
                "Truth_GHI": t_ghi_val
            })
            
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # 1. Dynamically anchor to repository root (assuming this is in scripts/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # 2. Set dynamic paths for local data locations
    CAMS_DIR = PROJECT_ROOT / "data" / "cams"
    SYN_DIR = PROJECT_ROOT / "data" / "gum" / "synthetic"
    GRID_CSV = PROJECT_ROOT / "data" / "grid" / "cams_lon_lat_vectors.csv"

    raw_npz = str(CAMS_DIR / "ghi_grid2500_3ch_2021_2023_baseline_15mins_robust(stale)_csi.npz")
    syn_npz = str(SYN_DIR / "15mins_synthetic_label.npz")

    # 3. Create organized subfolders in the results directory
    RESULTS_OFLOW = PROJECT_ROOT / "results" / "optical_flow"
    RESULTS_OFLOW.mkdir(parents=True, exist_ok=True)

    OUTPUT_RAW_OFLOW = str(RESULTS_OFLOW / "OpticalFlow_Raw_hko_station_interpolated_fold4_15min_slide.csv")
    OUTPUT_SYN_OFLOW = str(RESULTS_OFLOW / "OpticalFlow_Syn_hko_station_interpolated_fold4_15min_slide.csv")

    # 4. Execute
    h_lat_idx, h_lon_idx, h_weights = get_hko_idw_weights(str(GRID_CSV))
    
    run_rolling_oflow_short_horizon(raw_npz, raw_npz, OUTPUT_RAW_OFLOW, h_lat_idx, h_lon_idx, h_weights)
    run_rolling_oflow_short_horizon(raw_npz, syn_npz, OUTPUT_SYN_OFLOW, h_lat_idx, h_lon_idx, h_weights)