# Project Structure Summary ✅

## Complete Directory Structure

```
HooperAI/
│
├── 📁 data/                          # Raw and processed data
│   ├── raw/                         # Original, unprocessed data
│   │   ├── videos/                  # Raw video files
│   │   └── images/                  # Raw image files
│   └── processed/                   # Processed/preprocessed data
│       ├── frames/                  # Extracted video frames
│       └── annotations/             # Manual annotations
│
├── 📁 datasets/                      # Training-ready datasets (YOLO format)
│   └── [dataset-name]/              # Individual dataset directories
│       ├── data.yaml                # Dataset configuration
│       ├── train/                   # Training split
│       │   ├── images/              # Training images
│       │   └── labels/              # Training labels
│       ├── valid/                   # Validation split
│       │   ├── images/              # Validation images
│       │   └── labels/              # Validation labels
│       └── test/                    # Test split (optional)
│
├── 📁 configs/                       # Configuration files
│   ├── datasets/                    # Dataset configurations
│   │   └── template.yaml           # Dataset config template
│   └── training/                    # Training configurations
│       └── default.yaml            # Default training parameters
│
├── 📁 models/                        # Trained model weights (gitignored)
│   └── [model files]                # .pt model files
│
├── 📁 runs/                          # Training outputs (gitignored)
│   └── detect/                      # Detection training runs
│
├── 📁 evaluations/                   # Evaluation results
│   ├── metrics/                     # Evaluation metrics (CSV, JSON)
│   ├── plots/                       # Visualizations
│   └── reports/                     # Evaluation reports
│
├── 📁 notebooks/                     # Jupyter notebooks
│   ├── exploration/                 # Data exploration
│   ├── analysis/                    # Analysis notebooks
│   └── experiments/                # Experimental notebooks
│
├── 📁 scripts/                       # Utility scripts
│   └── train_colab.py               # Colab training script
│
├── 📁 docs/                          # Documentation
│   ├── DATASET_STRUCTURE.md         # Dataset structure guide
│   └── [other docs]
│
└── 📁 archives/                      # Archived datasets/runs
    └── [archived files]
```

## Key Directories Explained

### 📊 Data Flow

1. **`data/raw/`** → Original videos/images
2. **`data/processed/`** → Processed frames/annotations
3. **`datasets/`** → YOLO-formatted training datasets
4. **`runs/`** → Training outputs and checkpoints
5. **`models/`** → Best model weights (optional)
6. **`evaluations/`** → Metrics, plots, reports

### 🎯 Usage Guide

#### Adding a New Dataset

1. Place raw data in `data/raw/videos/` or `data/raw/images/`
2. Process and annotate → `data/processed/`
3. Convert to YOLO format → `datasets/your-dataset-name/`
4. Create `data.yaml` using `configs/datasets/template.yaml`
5. Start training!

#### Training a Model

1. Use dataset config: `configs/datasets/your-dataset.yaml`
2. Use training config: `configs/training/default.yaml` (or create custom)
3. Run training script or use YOLO CLI
4. Results saved to `runs/detect/`

#### Evaluating Models

1. Run evaluation scripts
2. Metrics saved to `evaluations/metrics/`
3. Plots saved to `evaluations/plots/`
4. Reports saved to `evaluations/reports/`

## Configuration Files

### Dataset Config (`configs/datasets/template.yaml`)
- Dataset paths
- Class names and count
- Train/valid/test splits

### Training Config (`configs/training/default.yaml`)
- Model architecture
- Training parameters
- Augmentation settings
- Output settings

## Git Ignore Rules

The following are excluded from Git:
- `data/raw/` and `data/processed/` contents
- `datasets/*/train/`, `datasets/*/valid/`, `datasets/*/test/` contents
- `models/` directory
- `runs/` directory
- Large evaluation outputs

Directory structures are preserved via `.gitkeep` files.

## Next Steps

1. ✅ Structure created
2. 📝 Add your first dataset to `datasets/`
3. ⚙️ Configure dataset in `configs/datasets/`
4. 🚀 Start training!

See [DATASET_STRUCTURE.md](DATASET_STRUCTURE.md) for detailed guide.
