# Configurations Directory

This directory contains configuration files for datasets, training, and model settings.

## Structure

```
configs/
├── datasets/              # Dataset configuration files
│   ├── template.yaml     # Dataset configuration template
│   └── *.yaml            # Specific dataset configs
├── training/              # Training configuration files
│   ├── default.yaml      # Default training parameters
│   └── *.yaml            # Specific training configs
└── README.md             # This file
```

## Configuration Files

### Dataset Configurations (`datasets/`)
- `template.yaml` - Template for creating new dataset configs
- Dataset-specific YAML files matching dataset names

### Training Configurations (`training/`)
- `default.yaml` - Default training parameters
- Experiment-specific configs (e.g., `experiment-001.yaml`)

## Usage

1. Copy template files and modify for your needs
2. Reference configs in training scripts
3. Keep configs version-controlled for reproducibility

## Example Usage

```python
from ultralytics import YOLO

# Load dataset config
model = YOLO('yolov8n.pt')
model.train(data='configs/datasets/basketball-v1.yaml', 
            epochs=100,
            imgsz=640)
```
