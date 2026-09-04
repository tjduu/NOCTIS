# noctis: Nocturnal Optical Cloud Tracking via Infrared Synthesis

This repository provides the official implementation accompanying the paper on the **NOCTIS** (Nocturnal Optical Cloud Tracking via Infrared Synthesis) framework.

Satellite-based solar forecasting fundamentally relies on visible and near-infrared channels to retrieve the Clear-Sky Index ($\kappa$, CSI), creating a persistent "dawn barrier" due to the absence of nocturnal optical observations. NOCTIS overcomes this long-standing spectral boundary by synthesizing physically consistent nighttime CSI fields directly from longwave infrared brightness-temperature data. By uncovering a latent mapping between thermal structure and cloud optical state, NOCTIS restores atmospheric memory across the nocturnal gap, enabling continuous, model-agnostic solar forecasting across the entire diurnal divide.

## Key Features

* **Nighttime CSI Synthesis:** Synthesizes nocturnal cloud optical fields from infrared thermal observations to eliminate the dawn forecast spin-up barrier.
* **Model-Agnostic Compatibility:** Seamlessly integrates with both kinematic advection baselines (Optical Flow) and high-dimensional sequence models (Fourier Neural Operator).
* **Station-Level Evaluation:** Features an in-memory Inverse Distance Weighting (IDW) interpolation module to project grid-scale forecasts onto specific local station coordinates (e.g., King's Park Station).
* **Hardware-Agnostic Execution:** Fully native PyTorch Lightning pipeline with automatic hardware routing across NVIDIA GPUs (CUDA) and Apple Silicon (MPS).

## Repository Structure

```text
noctis/
├── data/                       # CAMS grid data and synthetic cloud labels
│   ├── cams/
│   ├── grid/
│   └── gum/
│   └── ....
├── forecast_checkpoint/        # Saved model weights and training logs
├── results/                    # Output CSV predictions and evaluation metrics
│   ├── fno/
│   └── optical_flow/
├── scripts/
│   ├── train_fno.py            # FNO training pipeline (sliding-window cross-validation)
│   ├── run_fno.py              # FNO inference and HKO station IDW interpolation
│   └── run_optical_flow.py     # Optical Flow baseline evaluation
├── src/
│   └── noctis/                # Core Python package (models, loss functions, dataloaders)
├── environment.yml             # Conda environment definition
└── README.md

```