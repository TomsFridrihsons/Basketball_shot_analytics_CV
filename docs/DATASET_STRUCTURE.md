# Dataset Structure Guide

This document describes the dataset and project structure for HooperAI.

## 📁 Project Structure

```
HooperAI/
├── data/                    # Raw and processed data
│   ├── raw/                 # Original, unprocessed data
│   │   ├── videos/          # Raw video files
│   │   └── images/          # Raw image files
│   └── processed/           # Processed/preprocessed data
│       ├── frames/          # Extracted video frames
│       └── annotations/     # Manual annotations
│
├── datasets/                 # Training-ready datasets (YOLO format)
│   └── dataset-name/         # Individual dataset
│       ├── data.yaml        # Dataset configuration
│       ├── train/           # Training split
│       ├── valid/           # Validation split
│       └── test/            # Test split (optional)
│
├── configs/                  # Configuration files
│   ├── datasets/            # Dataset configurations
│   │   └── template.yaml    # Dataset config template
│   └── training/            # Training configurations
│       └── default.yaml     # Default training params
│
├── models/                   # Trained model weights (gitignored)
│   └── [model files]
│
├── runs/                     # Training outputs (gitignored)
│   └── detect/              # Detection training runs
│
├── evaluations/              # Evaluation results
│   ├── metrics/             # Evaluation metrics
│   ├── plots/               # Visualizations
│   └── reports/             # Evaluation reports
│
├── notebooks/               # Jupyter notebooks
│   ├── exploration/         # Data exploration
│   ├── analysis/           # Analysis notebooks
│   └── experiments/        # Experimental notebooks
│
├── scripts/                  # Utility scripts
│   └── train_colab.py      # Colab training script
│
├── docs/                     # Documentation
└── archives/                 # Archived datasets/runs
```

## 🔄 Data Flow

1. **Raw Data** → `data/raw/`
   - Original videos and images
   - Unprocessed, as received

2. **Processing** → `data/processed/`
   - Frame extraction
   - Initial annotations
   - Preprocessing

3. **Training Dataset** → `datasets/`
   - YOLO-formatted datasets
   - Train/valid/test splits
   - Ready for training

4. **Training** → `runs/detect/`
   - Training outputs
   - Model checkpoints
   - Training logs

5. **Models** → `models/` (optional)
   - Best model weights
   - Production-ready models

6. **Evaluation** → `evaluations/`
   - Metrics and analysis
   - Visualizations
   - Reports

## 📊 Dataset Format

### YOLO Dataset Structure

```
dataset-name/
├── data.yaml              # Dataset config
├── train/
│   ├── images/           # Training images
│   └── labels/           # Training labels (.txt)
├── valid/
│   ├── images/           # Validation images
│   └── labels/           # Validation labels
└── test/                 # Optional
    ├── images/
    └── labels/
```

### Label Format (YOLO)

Each image has a corresponding `.txt` file with normalized bounding boxes:

```
class_id center_x center_y width height
```

Example:
```
0 0.5 0.5 0.2 0.3
1 0.3 0.4 0.1 0.2
```

## 🎯 Best Practices

1. **Dataset Naming**: Use descriptive names (e.g., `basketball-v1`, `basketball-enhanced`)

2. **Splits**: Typical splits:
   - 80/10/10 (train/valid/test)
   - 70/15/15
   - Adjust based on dataset size

3. **Version Control**: 
   - Keep configs and structure in Git
   - Exclude large data files (use `.gitignore`)
   - Document dataset sources and processing

4. **Documentation**: 
   - Document each dataset in `datasets/README.md`
   - Include statistics (images, classes, splits)
   - Note data sources and processing steps

5. **Experiments**:
   - Use descriptive experiment names
   - Save configs for reproducibility
   - Track results in `evaluations/`

## 📝 Adding a New Dataset

1. Prepare data in `data/raw/`
2. Process and annotate → `data/processed/`
3. Convert to YOLO format → `datasets/your-dataset-name/`
4. Create `data.yaml` config (use template)
5. Update `datasets/README.md`
6. Start training!

## 🔗 Related Documentation

- [Dataset Archive Info](DATASET_ARCHIVE.md) - Archived datasets
- [Colab Setup](colab_setup.md) - Training in Colab
- [Training Configs](../configs/README.md) - Configuration guide
