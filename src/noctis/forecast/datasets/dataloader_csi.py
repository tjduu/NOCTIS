import os
import numpy as np
import torch
from torch.utils.data import Dataset
from noctis.forecast.datasets.utils import create_loader
import pandas as pd

class CSIDailyAlignedDataset15mins(Dataset):
    def __init__(
        self,
        npz_path: str,
        split: str,
        pre_seq_length: int = 96,  # 24 hours * 4 frames/hr
        aft_seq_length: int = 16,  # 4 hours * 4 frames/hr
        target_channel: int = 2,
        fold_idx: int = 0,       
        n_folds: int = 5,        
        normalize: bool = False,
        utc_offset_hours: int = 8,
        target_start_hour: int = 5,
        require_contiguous: bool = True,
        use_augment: bool = False,
        aug_multiplier: int = 1,  
        pad_to: int | None = 64,
        **kwargs
    ):
        super().__init__()
        self.data_name = 'csi' 
        self.split = split
        
        self.use_augment = use_augment if split == 'train' else False
        self.multiplier = aug_multiplier if self.use_augment else 1
        self.pre = pre_seq_length
        self.aft = aft_seq_length
        self.total = self.pre + self.aft
        self.ch = target_channel
        self.pad_to = pad_to

        npz = np.load(npz_path)
        raw_data = npz["data"].astype(np.float32) 
        time_utc = npz["time"]
        
        # We only interpolate the Target Channel (CSI) to prevent NaNs
        csi_channel = raw_data[:, self.ch, :, :]
        missing_mask = (csi_channel == -1.0)
        
        if np.any(missing_mask):
            print(f"[{split}] Interpolating {np.sum(np.any(missing_mask, axis=(1,2)))} missing (-1.0) frames (Memory Safe Mode)...")
            time_idx = np.arange(csi_channel.shape[0])
            for y in range(csi_channel.shape[1]):
                for x in range(csi_channel.shape[2]):
                    pixel_timeline = csi_channel[:, y, x]
                    is_missing = (pixel_timeline == -1.0)
                    
                    if np.any(is_missing):
                        valid_idx = time_idx[~is_missing]
                        valid_data = pixel_timeline[~is_missing]
                        
                        # Use np.interp to fill the gaps in place
                        interp_values = np.interp(time_idx[is_missing], valid_idx, valid_data)
                        raw_data[is_missing, self.ch, y, x] = interp_values
            
        self.data = raw_data
        self.N, _, self.H, self.W = self.data.shape
        if self.pad_to:
            pad_h, pad_w = self.pad_to - self.H, self.pad_to - self.W
            self.top, self.left = pad_h // 2, pad_w // 2
            self.bottom, self.right = pad_h - self.top, pad_w - self.left

        hk = time_utc + np.timedelta64(int(utc_offset_hours), "h")
        dt_hk = pd.DatetimeIndex(hk)
        is_target_hour = (dt_hk.hour == target_start_hour) & (dt_hk.minute == 0)

        #ROBUST 2-YEAR CV / 1-YEAR TEST SPLIT
        test_start = int(self.N * (2 / 3)) 
        chunk_size = test_start // (n_folds + 1)
        buffer = self.total # 112-frame purge buffer to prevent leakage
        
        train_end = (fold_idx + 1) * chunk_size
        val_end   = train_end + chunk_size
        
        if split == "train":
            a, b = 0, train_end - buffer
        elif split == "val":
            a, b = train_end, val_end - buffer
        elif split == "test":
            a, b = test_start, self.N
        else:
            raise ValueError(f"Unknown split: {split}")

        # Extract Valid Sequences within Boundaries (Updated for 15-min)
        candidates = np.where(is_target_hour)[0]
        candidates = candidates[(candidates >= a) & (candidates + self.total <= b)]
        
        if require_contiguous:
            good = []
            for s in candidates:
                dif = (time_utc[s+1:s+self.total] - time_utc[s:s+self.total-1]).astype("timedelta64[m]").astype(np.int64)
                if np.all(dif == 15): 
                    good.append(s)
            self.starts = np.array(good)
        else:
            self.starts = candidates

        # 6. Normalization
        self.normalize = normalize
        self.min_val, self.max_val = 0.0, 1.0

    def __len__(self):
        return len(self.starts) * self.multiplier

    def _pad_seq(self, seq):
        if self.pad_to is None: return seq
        return np.pad(seq, ((0,0), (0,0), (self.top, self.bottom), (self.left, self.right)), mode='constant')

    def __getitem__(self, idx):
        real_idx = idx % len(self.starts)
        s = self.starts[real_idx]
        seq = self.data[s : s + self.total, self.ch : self.ch + 1]
        
        if self.normalize:
            seq = (seq - self.min_val) / (self.max_val - self.min_val)
            seq = np.clip(seq, 0.0, 1.0)
    
        x = self._pad_seq(seq[:self.pre])
        y = self._pad_seq(seq[self.pre:])

        # Augmentation (Train Only)
        if self.use_augment:
            if np.random.rand() > 0.5:
                x, y = np.flip(x, axis=-1).copy(), np.flip(y, axis=-1).copy()
            if np.random.rand() > 0.5:
                x, y = np.flip(x, axis=-2).copy(), np.flip(y, axis=-2).copy()
            rot = np.random.choice([0, 1, 2, 3])
            if rot > 0:
                x = np.rot90(x, k=rot, axes=(-2, -1)).copy()
                y = np.rot90(y, k=rot, axes=(-2, -1)).copy()
            if np.random.rand() > 0.3:  
                x[np.random.randint(0, self.pre)] = 0.0 
            if np.random.rand() > 0.5:
                x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
                
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class CSIDailyAlignedDataset15minsslide(Dataset):
    def __init__(
        self,
        npz_path: str,
        split: str,
        pre_seq_length: int = 96,  # 24 hours * 4 frames/hr
        aft_seq_length: int = 96,  
        target_channel: int = 2,
        fold_idx: int = 0,       
        n_folds: int = 5,        
        normalize: bool = False,
        utc_offset_hours: int = 8,
        target_start_hour: int = 5,
        require_contiguous: bool = True,
        use_augment: bool = False,
        aug_multiplier: int = 1,  
        pad_to: int | None = 64,
        sliding_window: bool = False,  
        stride: int = 1,               
        **kwargs
    ):
        super().__init__()
        self.data_name = 'csi' 
        self.split = split
        
        self.use_augment = use_augment if split == 'train' else False
        self.multiplier = aug_multiplier if self.use_augment else 1

        self.pre = pre_seq_length
        self.aft = aft_seq_length
        self.total = self.pre + self.aft
        self.ch = target_channel
        self.pad_to = pad_to
        self.sliding_window = sliding_window
        self.stride = stride

        
        npz = np.load(npz_path)
        raw_data = npz["data"].astype(np.float32) 
        time_utc = npz["time"]
        
        
        # DYNAMIC INTERPOLATION FOR -1.0 GAPS
        csi_channel = raw_data[:, self.ch, :, :]
        missing_mask = (csi_channel == -1.0)
        
        if np.any(missing_mask):
            print(f"[{split}] Interpolating {np.sum(np.any(missing_mask, axis=(1,2)))} missing (-1.0) frames (Memory Safe Mode)...")
            time_idx = np.arange(csi_channel.shape[0])
            for y in range(csi_channel.shape[1]):
                for x in range(csi_channel.shape[2]):
                    pixel_timeline = csi_channel[:, y, x]
                    is_missing = (pixel_timeline == -1.0)
                    if np.any(is_missing):
                        valid_idx = time_idx[~is_missing]
                        valid_data = pixel_timeline[~is_missing]
                        interp_values = np.interp(time_idx[is_missing], valid_idx, valid_data)
                        raw_data[is_missing, self.ch, y, x] = interp_values
            
        self.data = raw_data
        self.N, _, self.H, self.W = self.data.shape
        if self.pad_to:
            pad_h, pad_w = self.pad_to - self.H, self.pad_to - self.W
            self.top, self.left = pad_h // 2, pad_w // 2
            self.bottom, self.right = pad_h - self.top, pad_w - self.left
        test_start = int(self.N * (2 / 3)) 
        chunk_size = test_start // (n_folds + 1)
        buffer = self.total 
        train_end = (fold_idx + 1) * chunk_size
        val_end   = train_end + chunk_size
        
        if split == "train":
            a, b = 0, train_end - buffer
        elif split == "val":
            a, b = train_end, val_end - buffer
        elif split == "test":
            a, b = test_start, self.N
        else:
            raise ValueError(f"Unknown split: {split}")

        if self.sliding_window:
            candidates = np.arange(a, b - self.total + 1, self.stride)
        else:
            # Fixed Window (5 AM HKT)
            hk = time_utc + np.timedelta64(int(utc_offset_hours), "h")
            dt_hk = pd.DatetimeIndex(hk)
            is_target_hour = (dt_hk.hour == target_start_hour) & (dt_hk.minute == 0)
            candidates = np.where(is_target_hour)[0]
            candidates = candidates[(candidates >= a) & (candidates + self.total <= b)]

        # 5. Vectorized Contiguity Check
        if require_contiguous:
            all_diffs = np.diff(time_utc).astype("timedelta64[m]").astype(np.int64)
            valid_steps = (all_diffs == 15)
            # Create a cumulative sum of valid steps to check windows instantly
            valid_cumsum = np.concatenate(([0], np.cumsum(valid_steps)))
            
            good = []
            for s in candidates:
                # If the number of 15-min jumps exactly matches the window length - 1, it's contiguous
                if valid_cumsum[s + self.total - 1] - valid_cumsum[s] == self.total - 1:
                    good.append(s)
            self.starts = np.array(good)
        else:
            self.starts = candidates

        # 6. Normalization
        self.normalize = normalize
        self.min_val, self.max_val = 0.0, 1.0

        print(f"[{split.upper()}] Mode: {'Sliding Window' if self.sliding_window else 'Fixed Window'} | Total Samples: {len(self.starts) * self.multiplier}")

    def __len__(self):
        return len(self.starts) * self.multiplier

    def _pad_seq(self, seq):
        if self.pad_to is None: return seq
        return np.pad(seq, ((0,0), (0,0), (self.top, self.bottom), (self.left, self.right)), mode='constant')

    def __getitem__(self, idx):
        real_idx = idx % len(self.starts)
        s = self.starts[real_idx]
        seq = self.data[s : s + self.total, self.ch : self.ch + 1]
        
        if self.normalize:
            seq = (seq - self.min_val) / (self.max_val - self.min_val)
            seq = np.clip(seq, 0.0, 1.0)
    
        x = self._pad_seq(seq[:self.pre])
        y = self._pad_seq(seq[self.pre:])

        # Optional Augmentation
        if self.use_augment:
            if np.random.rand() > 0.5:
                x, y = np.flip(x, axis=-1).copy(), np.flip(y, axis=-1).copy()
            if np.random.rand() > 0.5:
                x, y = np.flip(x, axis=-2).copy(), np.flip(y, axis=-2).copy()
            rot = np.random.choice([0, 1, 2, 3])
            if rot > 0:
                x = np.rot90(x, k=rot, axes=(-2, -1)).copy()
                y = np.rot90(y, k=rot, axes=(-2, -1)).copy()
            if np.random.rand() > 0.3:  
                x[np.random.randint(0, self.pre)] = 0.0 
            if np.random.rand() > 0.5:
                x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
                
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def load_data_15mins_slide(
    batch_size, val_batch_size, data_root,
    train_npz_name="15mins_synthetic_2021_2023.npz", 
    eval_npz_name="15mins_synthetic_2021_2023.npz", 
    fold_idx=0, n_folds=5,
    num_workers=4, pre_seq_length=96, aft_seq_length=96,
    distributed=False, use_prefetcher=False, drop_last=False,
    target_channel=2, normalize=False, utc_offset_hours=8,
    target_start_hour=5, require_contiguous=True,
    use_augment=False, aug_multiplier=4, pad_to=64, 
    sliding_train_val=True, stride=1, **kwargs 
):
    train_npz_path = os.path.join(data_root, train_npz_name)
    eval_npz_path = os.path.join(data_root, eval_npz_name)

    # TRAIN: Sliding Window
    train_set = CSIDailyAlignedDataset15mins(
        train_npz_path, "train", pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
        target_channel=target_channel, fold_idx=fold_idx, n_folds=n_folds, normalize=normalize, 
        utc_offset_hours=utc_offset_hours, target_start_hour=target_start_hour, 
        require_contiguous=require_contiguous, use_augment=use_augment, aug_multiplier=aug_multiplier, 
        pad_to=pad_to, sliding_window=sliding_train_val, stride=stride, **kwargs
    )
    
    # VAL: Sliding Window (Matches Train environment)
    val_set = CSIDailyAlignedDataset15mins(
        eval_npz_path, "val", pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
        target_channel=target_channel, fold_idx=fold_idx, n_folds=n_folds, normalize=normalize, 
        utc_offset_hours=utc_offset_hours, target_start_hour=target_start_hour, 
        require_contiguous=require_contiguous, use_augment=False, pad_to=pad_to,
        sliding_window=sliding_train_val, stride=stride, **kwargs
    )
    
    # TEST: STRICTLY Fixed Window (Apples-to-apples with old models)
    test_set = CSIDailyAlignedDataset15mins(
        eval_npz_path, "test", pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
        target_channel=target_channel, fold_idx=fold_idx, n_folds=n_folds, normalize=normalize, 
        utc_offset_hours=utc_offset_hours, target_start_hour=target_start_hour, 
        require_contiguous=require_contiguous, use_augment=False, pad_to=pad_to,
        sliding_window=False, **kwargs
    )
    dl_train = create_loader(train_set, batch_size=batch_size, shuffle=True, is_training=True, pin_memory=True, drop_last=True, num_workers=num_workers, persistent_workers=True)
    dl_val = create_loader(val_set, batch_size=val_batch_size, shuffle=False, is_training=False, pin_memory=True, drop_last=drop_last, num_workers=num_workers)
    dl_test = create_loader(test_set, batch_size=val_batch_size, shuffle=False, is_training=False, pin_memory=True, drop_last=drop_last, num_workers=num_workers)
    
    return dl_train, dl_val, dl_test