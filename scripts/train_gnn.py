import os
# --- MANDATORY FOR FNO DETERMINISM ---
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# --- BYPASS OpenMP CONFLICT ON macOS --- 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import lightning as L
from pathlib import Path
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from noctis.forecast.datasets.dataloader_csi import load_data_15mins_slide
from noctis.forecast.models.gnn import GNNForecaster
torch.set_float32_matmul_precision('high')

def main():
    L.seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    save_base = PROJECT_ROOT / "forecast_checkpoint" / "fix_seed"

    training_runs = {
        "GNN_Baseline_Robust_15min_Slide": {
            "data_root": str(PROJECT_ROOT / "data" / "cams"),
            "dataset": "ghi_grid2500_3ch_2021_2023_baseline_15mins_robust(stale)_csi.npz", 
            "save_root": str(save_base / "gnn_baseline_15min_slide")
        },
        "GNN_Proposed_Synthetic_15min_Slide": {
            "data_root": str(PROJECT_ROOT / "data" / "gum" / "synthetic"),
            "dataset": "15mins_synthetic_label.npz",
            "save_root": str(save_base / "gnn_syn_15min_slide")
        }
    }

    base_config = {
        'dataname': 'csi_weather',
        'data_name': 'csi',
        'metrics': ['mae', 'mse', 'rmse'],
        'method': 'gnn_hierarchical',
        'in_shape': [96, 1, 64, 64],
        'pre_seq_length': 96,
        'aft_seq_length': 96,
        'total_length': 192,
        
        # GNN specifics
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

    n_folds = 5 # 5-Fold Cross Validation

    # --- 4. THE AUTOMATED CV LOOP ---
    for run_name, setup in training_runs.items():
        print("\n" + "="*70)
        print(f"🚀 STARTING {n_folds}-FOLD CV FOR: {run_name}")
        print(f"Data: {setup['dataset']}")
        print(f"Save Path: {setup['save_root']}")
        print("="*70 + "\n")

        for fold in range(n_folds):
            print(f"\n--- Running Fold {fold+1}/{n_folds} ---")
            
            # Create isolated folder for this specific fold's checkpoints
            fold_save_dir = os.path.join(setup['save_root'], f"fold_{fold}")
            os.makedirs(fold_save_dir, exist_ok=True)
            
            try:
                # Load the specific dataset boundaries for this fold
                train_loader, val_loader, test_loader = load_data_15mins_slide(
                    batch_size=4, val_batch_size=4, data_root=setup['data_root'],
                    train_npz_name=setup['dataset'],
                    eval_npz_name=setup['dataset'], 
                    fold_idx=fold, n_folds=n_folds,
                    pre_seq_length=96, aft_seq_length=96, pad_to=64,
                    normalize=True,      
                    use_augment=False,
                    utc_offset_hours=8, 
                    target_start_hour=5,
                    aug_multiplier=4,
                    sliding_train_val=True,
                    stride = 1
                )

                # Dynamically update steps_per_epoch for the scheduler
                base_config['steps_per_epoch'] = len(train_loader)

                # Initialize fresh Hierarchical GNN
                model = GNNForecaster(config=base_config)

                # Setup isolated logging inside the fold folder
                logger = CSVLogger(
                    save_dir=fold_save_dir, 
                    name="logs", 
                    version="version_0"
                )

                # Initialize a fresh Trainer
                trainer = L.Trainer(
                    max_epochs=base_config['epoch'],
                    accelerator="gpu",
                    devices=1,
                    logger=logger,
                    precision="32-true", 
                    deterministic="warn", # warn for m1   # True for CUDA         
                    gradient_clip_val=1.0, 
                    log_every_n_steps=len(train_loader), 
                    callbacks=[
                        EarlyStopping(
                            monitor="val_loss", 
                            patience=10,          
                            mode="min",
                            check_on_train_epoch_end=False 
                        ),
                        ModelCheckpoint(
                            monitor="val_loss", 
                            save_top_k=1, 
                            mode="min",
                            filename='best-{epoch:02d}'
                        ),
                        LearningRateMonitor(logging_interval='epoch')
                    ],
                    default_root_dir=fold_save_dir
                )

                # TRAIN!
                trainer.fit(model, train_loader, val_loader)

                # CRITICAL MEMORY CLEANUP
                del model
                del trainer
                del train_loader
                del val_loader
                del test_loader
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n❌ FATAL ERROR in {run_name} Fold {fold}: {e}\n")
                torch.cuda.empty_cache()

    print("\n🎉 ALL GNN CROSS-VALIDATION RUNS HAVE FINISHED! 🎉")

if __name__ == "__main__":
    main()