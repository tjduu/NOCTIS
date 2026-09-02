import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

# --- BYPASS OpenMP CONFLICT ON macOS --- # comment out if not on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from nocitis.forecast.models.fno import FNOForecaster   
from nocitis.forecast.datasets.dataloader_csi import CSIDailyAlignedDataset15minsslide
from nocitis.forecast.utils import get_hko_idw_weights

# ==========================================
# 2. IN-MEMORY INFERENCE & INTERPOLATION
# ==========================================
def run_24h_fast_interpolation(
    ckpt_path, npz_path, output_csv, lat_idx, lon_idx, weights, 
    batch_size=16, device='cpu'
):
    print(f"\n🚀 Processing: {os.path.basename(output_csv)}")
    
    ds = CSIDailyAlignedDataset15minsslide(
        npz_path=npz_path, split="test", fold_idx=4, n_folds=5, 
        pre_seq_length=96, aft_seq_length=96, target_channel=2, 
        normalize=True, utc_offset_hours=8, target_start_hour=5, 
        pad_to=64, sliding_window=False
    )
    
    time_utc = ds.npz_data["time"] if hasattr(ds, 'npz_data') else np.load(npz_path)["time"]
    dt_hk = pd.DatetimeIndex(time_utc + np.timedelta64(8, "h"))
    
    a, b = int(len(time_utc) * (2/3)), len(time_utc) - ds.total
    hourly_candidates = np.where(dt_hk.minute == 0)[0] 
    hourly_candidates = hourly_candidates[(hourly_candidates >= a) & (hourly_candidates + ds.total <= b)]
    
    valid_cumsum = np.concatenate(([0], np.cumsum(np.diff(time_utc).astype("timedelta64[m]").astype(np.int64) == 15)))
    ds.starts = np.array([s for s in hourly_candidates if valid_cumsum[s + ds.total - 1] - valid_cumsum[s] == ds.total - 1])
    
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    print(f"   -> Enqueued {len(ds.starts)} hourly initializations.")

    hparams = torch.load(ckpt_path, map_location=device, weights_only=False)['hyper_parameters']
    model = FNOForecaster.load_from_checkpoint(ckpt_path, config=hparams, strict=False, weights_only=False).to(device).eval()
    
    raw_data = np.load(npz_path)["data"]
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Inference & IDW")):
            x = batch[0].to(device)
            if x.dim() == 5: x = x.squeeze(2)
            
            try:
                pred = model(x, return_loss=False)
                pred = pred[0] if isinstance(pred, tuple) else pred
            except TypeError:
                pred = model(x)
                
            pred = pred.unsqueeze(2) if pred.dim() == 4 else pred
            pred_csi = pred.squeeze(2).cpu().numpy()[:, :, ds.top:ds.top+ds.H, ds.left:ds.left+ds.W]
            
            min_v, max_v = getattr(ds, 'min_val', 0.0), getattr(ds, 'max_val', 1.0)
            pred_csi = np.clip((pred_csi * (max_v - min_v)) + min_v, 0.0, 1.0)
            
            B = len(batch[0])
            start_indices = ds.starts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            true_ghi = np.stack([raw_data[s + ds.pre : s + ds.total, 0, :, :] for s in start_indices])
            true_csghi = np.stack([raw_data[s + ds.pre : s + ds.total, 1, :, :] for s in start_indices])
            pred_ghi = pred_csi * true_csghi 
            
            for b in range(B):
                s = start_indices[b]
                init_hkt = time_utc[s + ds.pre - 1] + np.timedelta64(8, "h")
                t_hkt = time_utc[s + ds.pre : s + ds.total] + np.timedelta64(8, "h")
                
                p_csi_val = np.sum(pred_csi[b][:, lat_idx[:, None], lon_idx] * weights, axis=(1,2))
                p_ghi_val = np.sum(pred_ghi[b][:, lat_idx[:, None], lon_idx] * weights, axis=(1,2))
                t_ghi_val = np.sum(true_ghi[b][:, lat_idx[:, None], lon_idx] * weights, axis=(1,2))
                c_ghi_val = np.sum(true_csghi[b][:, lat_idx[:, None], lon_idx] * weights, axis=(1,2))
                
                for t_step in range(ds.aft):
                    results.append({
                        "Init_Time_HKT": init_hkt,
                        "Time_HKT": t_hkt[t_step],
                        "Pred_CSI": p_csi_val[t_step],
                        "Truth_GHI": t_ghi_val[t_step],
                        "ClearSky_GHI": c_ghi_val[t_step],
                        "Pred_GHI": p_ghi_val[t_step],
                        "Latitude": 22.3019,
                        "Longitude": 114.1742
                    })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved precisely interpolated CSV to {output_csv}")

# ==========================================
# 3. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # --- 2. DYNAMIC PATHING ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    CAMS_DIR = PROJECT_ROOT / "data" / "cams"
    SYN_DIR = PROJECT_ROOT / "data" / "gum" / "synthetic"
    GRID_CSV = PROJECT_ROOT / "data" / "grid" / "cams_lon_lat_vectors.csv"
    
    RESULTS_FNO = PROJECT_ROOT / "results" / "fno"
    RESULTS_FNO.mkdir(parents=True, exist_ok=True)
    
    OUTPUT_RAW = RESULTS_FNO / "FNO_Raw_hko_station_interpolated_fold4_15min_slide.csv"
    OUTPUT_SYN = RESULTS_FNO / "FNO_Syn_hko_station_interpolated_fold4_15min_slide.csv"
    
    # --- 3. HARDWARE ACCELERATION ---
    if torch.cuda.is_available():
        accel_device = 'cuda'
    elif torch.backends.mps.is_available():
        accel_device = 'mps'
    else:
        accel_device = 'cpu'
    print(f"⚙️ Using inference device: {accel_device.upper()}")
    
    print("📍 Resolving HKO station to local grid...")
    h_lat_idx, h_lon_idx, h_weights = get_hko_idw_weights(str(GRID_CSV))
    
    # --- 4. CHECKPOINT LOCATIONS ---
    ckpt_base = PROJECT_ROOT / "forecast_checkpoint" / "fix_seed"
    
    raw_ckpt_pattern = str(ckpt_base / "fno_baseline_15min_slide" / "fold_4" / "*.ckpt")
    raw_ckpts = glob.glob(raw_ckpt_pattern)
    
    syn_ckpt_pattern = str(ckpt_base / "fno_syn_15min_slide" / "fold_4" / "*.ckpt")
    syn_ckpts = glob.glob(syn_ckpt_pattern)

    if raw_ckpts:
        raw_npz = str(CAMS_DIR / "ghi_grid2500_3ch_2021_2023_baseline_15mins_robust(stale)_csi.npz")
        run_24h_fast_interpolation(
            ckpt_path=raw_ckpts[0], npz_path=raw_npz, output_csv=str(OUTPUT_RAW),
            lat_idx=h_lat_idx, lon_idx=h_lon_idx, weights=h_weights, device=accel_device
        )
    else:
        print(f"⚠️ Skipping RAW model. No checkpoint found in {raw_ckpt_pattern}")

    if syn_ckpts:
        syn_npz = str(SYN_DIR / "15mins_synthetic_label.npz")
        run_24h_fast_interpolation(
            ckpt_path=syn_ckpts[0], npz_path=syn_npz, output_csv=str(OUTPUT_SYN),
            lat_idx=h_lat_idx, lon_idx=h_lon_idx, weights=h_weights, device=accel_device
        )
    else:
        print(f"⚠️ Skipping SYN model. No checkpoint found in {syn_ckpt_pattern}")
        
    print("\n🎉 All 24-hour predictions processed and saved to CSVs!")