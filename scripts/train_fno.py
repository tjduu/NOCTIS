import os
# --- MANDATORY FOR FNO DETERMINISM ---
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# --- BYPASS OpenMP CONFLICT ON macOS ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from nocitis.forecast.models.fno import FNOForecaster   

# CHANGE 1: Import the new sliding window loader
from nocitis.forecast.datasets.dataloader_csi import load_data_15mins_slide
torch.set_float32_matmul_precision('high')

class PersistenceDefeatingLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
    def forward(self, pred, target):
        return self.smooth_l1(pred, target)

def main():
    L.seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    save_base = PROJECT_ROOT / "forecast_checkpoint" / "fix_seed"
    
    training_runs = {
        "FNO_Baseline_Robust_15min_Slide": {
            "data_root": str(PROJECT_ROOT / "data" / "cams"),
            "dataset": "ghi_grid2500_3ch_2021_2023_baseline_15mins_robust(stale)_csi.npz", 
            "save_root": str(save_base / "fno_baseline_15min_slide")
        },
        "FNO_Proposed_Synthetic_15min_Slide": {
            "data_root": str(PROJECT_ROOT / "data" / "gum" / "synthetic"),
            "dataset": "15mins_synthetic_label.npz",
            "save_root": str(save_base / "fno_syn_15min_slide")
        }
    }

    # --- 3. BASE CONFIG FOR FNO ---
    base_config = {
        'dataname': 'csi_weather',
        'data_name': 'csi',
        'metrics': ['mae', 'mse', 'rmse'],
        'method': 'fno',
        'in_shape': [96, 1, 64, 64],
        'pre_seq_length': 96,
        'aft_seq_length': 96,
        'total_length': 192, 
        'modes': 32,
        'hidden_channels': 64,
        'in_channels': 96,   
        'out_channels': 96,  
        'opt': 'adamw',
        'lr': 5e-4,              
        'weight_decay': 0.1, 
        'opt_eps': 1e-8,
        'opt_betas': (0.9, 0.999),
        'sched': 'cosine',
        'epoch': 50,
        'min_lr': 1e-6,
        'warmup_lr': 1e-5,
        'warmup_epoch': 10,             
    }

    MethodClass = FNOForecaster
    n_folds = 5 
    target_fold = 4 

    # --- 4. THE AUTOMATED CV LOOP ---
    for run_name, setup in training_runs.items():
        print("\n" + "="*70)
        print(f"🚀 STARTING SLIDING-WINDOW TRAINING FOR: {run_name}")
        print(f"Data: {setup['dataset']}")
        print(f"Save Path: {setup['save_root']}")
        print("="*70 + "\n")

        for fold in [target_fold]:
            print(f"\n--- Running Fold {fold+1}/{n_folds} (Last Fold Only) ---")
            
            fold_save_dir = os.path.join(setup['save_root'], f"fold_{fold}")
            os.makedirs(fold_save_dir, exist_ok=True)
            
            try:
                # CHANGE 3: Call the new loader and pass sliding_train_val=True
                train_loader, val_loader, test_loader = load_data_15mins_slide(
                    batch_size=4, val_batch_size=8, data_root=setup['data_root'],
                    train_npz_name=setup['dataset'],
                    eval_npz_name=setup['dataset'], 
                    fold_idx=fold, n_folds=n_folds,
                    pre_seq_length=96, aft_seq_length=96, pad_to=64,
                    normalize=True,      
                    use_augment=False,
                    utc_offset_hours=8, 
                    target_start_hour=5,
                    aug_multiplier=4,
                    sliding_train_val=True, # Toggles the sliding window for Train/Val
                    stride=1                # 1 step = 15 minutes
                )

                base_config['steps_per_epoch'] = len(train_loader)

                model = MethodClass(config=base_config)
                model.criterion = PersistenceDefeatingLoss()

                logger = CSVLogger(save_dir=fold_save_dir, name="logs", version="version_0")

                trainer = L.Trainer(
                    max_epochs=base_config['epoch'],
                    accelerator="gpu",
                    devices=1,
                    logger=logger,
                    precision="32-true",           
                    deterministic=True,            
                    gradient_clip_val=1.0, 
                    # CHANGE 4: Hardcode logging frequency so it updates during massive epochs
                    log_every_n_steps=50, 
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=10, mode="min", check_on_train_epoch_end=False),
                        ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename='best-{epoch:02d}'),
                        LearningRateMonitor(logging_interval='epoch')
                    ],
                    default_root_dir=fold_save_dir
                )

                trainer.fit(model, train_loader, val_loader)

                del model, trainer, train_loader, val_loader, test_loader
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n❌ FATAL ERROR in {run_name} Fold {fold}: {e}\n")
                torch.cuda.empty_cache()

    print("\n🎉 ALL SLIDING WINDOW RUNS HAVE FINISHED! 🎉")

if __name__ == "__main__":
    main()