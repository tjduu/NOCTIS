import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt

EPS = 1e-6
DEG2RAD = np.pi / 180.0
EARTH_KM_PER_DEG = 111.32

# ---------------- Time helpers ----------------
def to_utc_index(x) -> pd.DatetimeIndex:
    idx = pd.to_datetime(x, utc=True, errors="coerce")
    if not isinstance(idx.dtype, pd.DatetimeTZDtype):
        idx = idx.tz_localize("UTC")
    return pd.DatetimeIndex(idx)

# ---------------- Geometry helpers ----------------
def mu0_from_sza_deg(sza_deg: np.ndarray) -> np.ndarray:
    return np.clip(np.cos(sza_deg * DEG2RAD), 1e-3, 1.0).astype(np.float32)

def approx_vza(lon_deg: np.ndarray, lat_deg: np.ndarray, sub_lon: float = 140.7) -> np.ndarray:
    dlon = (lon_deg - sub_lon) * DEG2RAD
    latr = lat_deg * DEG2RAD
    cos_psi = np.clip(np.cos(latr) * np.cos(dlon), -1.0, 1.0)
    return (np.arccos(cos_psi) / DEG2RAD).astype(np.float32)

def deg_offsets_to_km(lon_s, lat_s, lon_t, lat_t):
    lat_c = 0.5 * (lat_s + lat_t)
    dlon_km = (lon_s - lon_t) * np.cos(lat_c * DEG2RAD) * EARTH_KM_PER_DEG
    dlat_km = (lat_s - lat_t) * EARTH_KM_PER_DEG
    dist_km = np.sqrt(dlon_km ** 2 + dlat_km ** 2)
    return dlon_km.astype(np.float32), dlat_km.astype(np.float32), dist_km.astype(np.float32)

# ---------------- Stats ----------------
def compute_band_stats_from_ptree(ptree_mmap, time_idx, r0,r1,c0,c1, max_frames=2000):
    T = len(time_idx)
    if T == 0:
        raise RuntimeError("No frames for stats.")
    rng = np.random.default_rng(42)
    sel = rng.choice(T, size=min(T, max_frames), replace=False)
    acc_sum  = np.zeros(6, dtype=np.float64)
    acc_sum2 = np.zeros(6, dtype=np.float64)
    acc_cnt  = 0
    for j in sel:
        x = ptree_mmap[int(time_idx[j]), :, r0:r1, c0:c1].astype(np.float32)
        x = np.nan_to_num(x, 0.0, 0.0, 0.0)
        acc_sum  += x.reshape(6, -1).mean(axis=1)
        acc_sum2 += (x.reshape(6, -1)**2).mean(axis=1)
        acc_cnt  += 1
    mean = (acc_sum / max(acc_cnt,1)).astype(np.float32)
    var  = (acc_sum2 / max(acc_cnt,1)) - mean**2
    std  = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)
    std[std < EPS] = 1.0
    return {"mean": mean, "std": std}

#---------------crop-------------------
def pad_to_32(x):
    # x: (B,C,H,W)
    h, w = x.shape[-2:]
    new_h = ((h + 31) // 32) * 32
    new_w = ((w + 31) // 32) * 32
    pad_h = new_h - h
    pad_w = new_w - w
    # pad format: (left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w

def crop_back(x, h, w):
    return x[..., :h, :w]

def _to_numpy_float32(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x.astype(np.float32)

def print_pre_norm_stats(X):
    """
    X : np.ndarray or torch.Tensor, shape (T, C, H, W)
    Prints per-channel raw min, max, mean, std.
    """
    X = _to_numpy_float32(X)

    T, C, H, W = X.shape
    print("Pre-normalization stats:")
    print("Shape:", X.shape)

    Xf = X.reshape(T, C, -1)

    for c in range(C):
        vals = Xf[:, c, :].reshape(-1)
        vals = vals[np.isfinite(vals)]
        print(
            f"Channel {c:02d}: "
            f"min={vals.min():.4f}, max={vals.max():.4f}, "
            f"mean={vals.mean():.4f}, std={vals.std():.4f}"
        )


def print_post_norm_stats_day(X, channel_mean, channel_std):
    """
    X            : np.ndarray or torch.Tensor, shape (T, C, H, W)
    channel_mean : (C,)
    channel_std  : (C,)
    Prints per-channel post-normalization min, max, mean, std.
    """

    X = _to_numpy_float32(X)
    channel_mean = _to_numpy_float32(channel_mean)
    channel_std  = _to_numpy_float32(channel_std)

    T, C, H, W = X.shape
    mean_b = channel_mean[None, :, None, None]
    std_b  = channel_std[None, :, None, None]

    X_norm = (X - mean_b) / std_b
    Xf = X_norm.reshape(T, C, -1)

    print("Post-normalization stats for DAY X:")
    print("Shape:", X_norm.shape)

    for c in range(C):
        vals = Xf[:, c, :].reshape(-1)
        vals = vals[np.isfinite(vals)]
        print(
            f"Channel {c:02d}: "
            f"min={vals.min():.4f}, max={vals.max():.4f}, "
            f"mean={vals.mean():.4f}, std={vals.std():.4f}"
        )

def print_post_norm_stats(X_norm):
    """
    X_norm : np.ndarray or torch.Tensor, shape (T, C, H, W)
    Prints per-channel min, max, mean, std after normalization.
    """
    X_norm = _to_numpy_float32(X_norm)
    T, C, H, W = X_norm.shape

    print("Post-normalization stats for X_norm:")
    print("Shape:", X_norm.shape)

    Xf = X_norm.reshape(T, C, -1)

    for c in range(C):
        vals = Xf[:, c, :].reshape(-1)
        vals = vals[np.isfinite(vals)]
        print(
            f"Channel {c:02d}: "
            f"min={vals.min():.4f}, max={vals.max():.4f}, "
            f"mean={vals.mean():.4f}, std={vals.std():.4f}"
        )


def print_raw_night_stats(X):
    """
    X : np.ndarray, shape (T, C, H, W)
    Prints per-channel raw min, max, mean, std for night data.
    """
    X = _to_numpy_float32(X)
    T, C, H, W = X.shape

    print("Night raw stats:")
    print("Shape:", X.shape)

    for c in range(C):
        ch = X[:, c, :, :]
        vals = ch[np.isfinite(ch)]
        print(
            f"Channel {c:02d}: "
            f"min={vals.min():.4f}, max={vals.max():.4f}, "
            f"mean={vals.mean():.4f}, std={vals.std():.4f}"
        )


def train_val_test_split_indices(T, train_ratio=0.8, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    idx_all = np.arange(T)
    np.random.shuffle(idx_all)
    n_train = int(T * train_ratio)
    n_val   = int(T * val_ratio)
    idx_train = idx_all[:n_train]
    idx_val   = idx_all[n_train:n_train+n_val]
    idx_test  = idx_all[n_train+n_val:]
    return idx_train, idx_val, idx_test


def compute_means(arr):
    # arr: (T,H,W) or (T,1,H,W)
    if arr.ndim == 4:
        arr = arr[:, 0]
    return arr.mean(axis=(1,2))

def make_labels(c0, c1, csi,
                tw_cs_thr=76.0,      # W/m^2 : twilight if clear-sky GHI below this
                clear_thr=0.80,      # CSI mean threshold for "clear"
                cloudy_thr=0.35):    # CSI mean threshold for "cloudy"
    """
    Returns labels:
      0=twilight, 1=clear, 2=cloudy, 3=mixed
    """
    c0m  = compute_means(c0)
    c1m  = compute_means(c1)
    csim = compute_means(csi)

    labels = np.full(len(c1m), 3, dtype=np.int64)  # default mixed

    tw = c1m < tw_cs_thr
    labels[tw] = 0

    daytime = ~tw
    labels[daytime & (csim >= clear_thr)] = 1
    labels[daytime & (csim <= cloudy_thr)] = 2

    return labels, {"c0_mean": c0m, "c1_mean": c1m, "csi_mean": csim}

def stratified_split_grouped_by_time(labels, cams_time_min, train=0.8, val=0.1, test=0.1, seed=42):
    assert abs(train + val + test - 1.0) < 1e-9

    labels = np.asarray(labels)
    groups = np.asarray(cams_time_min)  # group key: same CAMS time => same y
    idx = np.arange(len(labels))

    # train vs tmp (val+test) using 5 folds (~20% holdout)
    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    best = None
    target_tmp = val + test
    for tr, tmp in sgkf1.split(idx, labels, groups=groups):
        frac = len(tmp) / len(idx)
        score = abs(frac - target_tmp)
        if best is None or score < best[0]:
            best = (score, tr, tmp)
    _, tr_idx, tmp_idx = best

    # tmp -> val/test using 2 folds (~50/50, perfect for 0.1/0.1)
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed + 1)
    v_rel, te_rel = next(sgkf2.split(tmp_idx, labels[tmp_idx], groups=groups[tmp_idx]))

    idx_train = tr_idx
    idx_val   = tmp_idx[v_rel]
    idx_test  = tmp_idx[te_rel]

    # hard guarantee: no time overlap
    assert set(groups[idx_train]).isdisjoint(set(groups[idx_test]))
    assert set(groups[idx_train]).isdisjoint(set(groups[idx_val]))
    assert set(groups[idx_val]).isdisjoint(set(groups[idx_test]))

    return idx_train, idx_val, idx_test


def summarize_split_csi_mean(y, idx_train, idx_val, idx_test, bins=30, plot=True):
    """
    y: (T,H,W) or (T,1,H,W) CSI in [0,1]
    idx_*: arrays of global indices into y
    bins: histogram bins
    plot: whether to show histograms
    """
    # per-sample CSI mean
    csi_mean = compute_means(y)  # (T,)

    def _summary(name, idx):
        idx = np.asarray(idx, dtype=np.int64)
        v = csi_mean[idx]
        out = {
            "N": int(len(idx)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "p05": float(np.quantile(v, 0.05)),
            "p25": float(np.quantile(v, 0.25)),
            "p50": float(np.quantile(v, 0.50)),
            "p75": float(np.quantile(v, 0.75)),
            "p95": float(np.quantile(v, 0.95)),
            "max": float(np.max(v)),
        }
        return out, v

    train_stats, v_tr = _summary("train", idx_train)
    val_stats,   v_va = _summary("val", idx_val)
    test_stats,  v_te = _summary("test", idx_test)

    # print table-ish report
    def _line(name, s):
        return (f"{name:>5} | N={s['N']:6d} | mean={s['mean']:.4f} std={s['std']:.4f} | "
                f"p05={s['p05']:.4f} p50={s['p50']:.4f} p95={s['p95']:.4f} | "
                f"min={s['min']:.4f} max={s['max']:.4f}")

    print(_line("train", train_stats))
    print(_line("val",   val_stats))
    print(_line("test",  test_stats))

    # optional histogram plot
    if plot:
        all_min = float(min(v_tr.min(), v_va.min(), v_te.min()))
        all_max = float(max(v_tr.max(), v_va.max(), v_te.max()))
        rng = (all_min, all_max)

        plt.figure(figsize=(10,4))
        plt.hist(v_tr, bins=bins, range=rng, alpha=0.5, label="train", density=True)
        plt.hist(v_va, bins=bins, range=rng, alpha=0.5, label="val", density=True)
        plt.hist(v_te, bins=bins, range=rng, alpha=0.5, label="test", density=True)
        plt.title("Per-sample CSI mean distribution")
        plt.xlabel("mean(CSI) per sample")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "csi_mean": csi_mean,  # full vector (T,)
    }


def prepare_3ch_cams(in_path, out_path, clip_csi=(0.0, 1.0), eps=1e-6):
    print(f"📦 Expanding CAMS to 3 Channels...")
    cams = np.load(in_path)
    time_key = 'time_hourly' if 'time_hourly' in cams.files else 'time'
    t_cams = cams[time_key]
    data_cams = cams['data']  # Expected Shape: (T, 2, H, W)
    
    print(f"  • Original Shape: {data_cams.shape}")
    ghi = data_cams[:, 0, :, :]
    ghi_cs = data_cams[:, 1, :, :]
    ghi_cs_safe = np.where(ghi_cs < eps, eps, ghi_cs)
    csi = ghi / ghi_cs_safe    
    if clip_csi is not None:
        csi = np.clip(csi, clip_csi[0], clip_csi[1])
        
    csi_expanded = csi[:, None, :, :].astype(np.float32)
    data_3ch = np.concatenate([data_cams, csi_expanded], axis=1)
    
    np.savez_compressed(
        out_path,
        **{time_key: t_cams},  # Dynamically preserves original time key
        data=data_3ch
    )
    
    print(f"✅ Successfully created 3-channel baseline!")
    print(f"  • New Shape:      {data_3ch.shape}")
    print(f"  • Channel 0:      GHI")
    print(f"  • Channel 1:      csGHI")
    print(f"  • Channel 2:      Calculated Baseline CSI")
    print(f"  • Saved to:       {out_path}\n")


def clone_cams_15min_with_labels(
    path_syn, 
    path_cams_15min, 
    out_path, 
    csi_channel=2, 
    missing_label=-1.0, 
    eps=1e-6
):
    print(f" (Labeled Unmatched Mode: {missing_label})...")
    
    syn = np.load(path_syn)
    cams = np.load(path_cams_15min)
    t_syn = syn['time_hourly']       
    csi_syn = syn['csi']             
    
    time_key = 'time_hourly' if 'time_hourly' in cams.files else 'time'
    t_cams = cams[time_key]          
    data_cams = cams['data']         
    dt_syn = pd.to_datetime(t_syn, unit='m', utc=True)
    dt_cams = pd.to_datetime(t_cams, utc=True)
    
    if data_cams.shape[1] <= csi_channel:
        needed_channels = csi_channel + 1 - data_cams.shape[1]
        padding = np.zeros((data_cams.shape[0], needed_channels, data_cams.shape[2], data_cams.shape[3]), dtype=data_cams.dtype)
        data_out = np.concatenate([data_cams, padding], axis=1)
    else:
        data_out = data_cams.copy()
        
    # Channel 1 is csGHI. If the entire grid is <= eps, it is night.
    is_cams_night = np.all(data_out[:, 1, :, :] <= eps, axis=(1, 2))
    
    # Fill the CSI channel for all night frames with the missing label
    data_out[is_cams_night, csi_channel, :, :] = missing_label
    
    print(f"  • Painted {np.sum(is_cams_night)} CAMS night frames with label {missing_label}.")

    # Nearest-Neighbor Alignment
    syn_df = pd.DataFrame({'syn_idx': np.arange(len(dt_syn))}, index=dt_syn).sort_index()
    cams_df = pd.DataFrame({'cams_idx': np.arange(len(dt_cams))}, index=dt_cams).sort_index()
    
    merged = pd.merge_asof(
        cams_df, 
        syn_df, 
        left_index=True, 
        right_index=True, 
        direction='nearest', 
        tolerance=pd.Timedelta('6min') 
    )
    
    merged_clean = merged.dropna(subset=['syn_idx'])
    
    cams_idx = merged_clean['cams_idx'].astype(int).values
    syn_idx = merged_clean['syn_idx'].astype(int).values
    
    # Overwrite Matches with Real Synthetic Data
    data_out[cams_idx, csi_channel, :, :] = csi_syn[syn_idx, 0, :, :]

    np.savez_compressed(
        out_path,
        time=t_cams,     
        data=data_out    
    )
    
    print(f"✅ Saved 15-min grid to: {out_path}")
    print(f"  • Successfully matched & injected {len(merged_clean)} frames.")
    print(f"  • {np.sum(is_cams_night) - len(merged_clean)} frames failed to match and retained the {missing_label} label.")

def qc_check_X(X_tensor):
    print("=== QC Report for Feature Tensor X ===\n")
    
    print("--- 1. Basic Properties ---")
    print(f"Shape: {X_tensor.shape}")
    print(f"Data Type: {X_tensor.dtype}")
    print("\n--- 2. Mathematical Integrity ---")
    nan_count = np.isnan(X_tensor).sum()
    inf_count = np.isinf(X_tensor).sum()
    
    print(f"Total NaNs: {nan_count}")
    print(f"Total Infs: {inf_count}")
    
    if nan_count == 0 and inf_count == 0:
        print("✅ PASS: X is completely clean of NaNs and infinite values.")
    else:
        print("❌ FAIL: X contains invalid values. Check your interpolation/division steps.")
        
    print("\n--- 3. Normalized TBB Channels (0-8) ---")
    tbb_data = X_tensor[:, :9, :, :]
    tbb_min = np.nanmin(tbb_data)
    tbb_max = np.nanmax(tbb_data)
    tbb_mean = np.nanmean(tbb_data)
    
    print(f"Global Min:  {tbb_min:.4f}  (Expected: ~0.1 to 0.4 for very cold clouds)")
    print(f"Global Max:  {tbb_max:.4f}  (Expected: ~1.0 to 1.5 for hot ground)")
    print(f"Global Mean: {tbb_mean:.4f}")
    
    if tbb_max > 5.0 or tbb_min < 0.0:
        print("❌ WARNING: Values are way outside expected bounds. Did you forget to convert Celsius to Kelvin?")
    else:
        print("✅ PASS: (T_sat / T_surface)^4 normalization range looks physically sound.")

    if X_tensor.shape[1] > 9:
        print("\n--- 4. Auxiliary Geo-Channels (9-11) ---")
        aux_data = X_tensor[:, 9:, :, :]
        print(f"Global Min:  {np.nanmin(aux_data):.4f}")
        print(f"Global Max:  {np.nanmax(aux_data):.4f}")
        print(f"Global Mean: {np.nanmean(aux_data):.4f}")

def create_stale_baseline(input_path, output_path, safe_threshold=5.0):
    print(f"Loading data from: {input_path}")
    npz_data = np.load(input_path)
    
    data = npz_data['data'].copy()
    time = npz_data['time']
    csghi = data[:, 1, :, :]
    
    print("Identifying fully clean daytime grids...")
    # A grid is only "clean" if the absolute darkest pixel is above safe threshold
    min_csghi_per_step = np.min(csghi, axis=(1, 2))
    is_clean_day = (min_csghi_per_step > safe_threshold)
    
    # We need an initial valid CSI. We'll find the very first clean day in the dataset.
    first_clean_idx = np.where(is_clean_day)[0][0]
    last_clean_csi = data[first_clean_idx, 2, :, :].copy()
    
    print("Applying robust forward-fill...")
    overwritten_frames = 0
    
    for i in range(len(time)):
        if is_clean_day[i]:
            
            last_clean_csi = data[i, 2, :, :]
        else:
            # The grid is experiencing dusk, night, or dawn. 
            # Freeze the CSI to prevent slicing cuts and zero-division artifacts.
            data[i, 2, :, :] = last_clean_csi
            overwritten_frames += 1
            
    print(f"Total frames overwritten (night + twilight transitions): {overwritten_frames} / {len(time)}")
    
    print(f"Saving robust baseline to: {output_path}")
    np.savez_compressed(output_path, data=data, time=time)
    print("Save complete!")
    
    npz_data.close()