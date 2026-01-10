# Colab Training Setup Guide

This branch is configured for training models in Google Colab and saving training outputs.

## Branch Purpose

The `colab-training` branch is designed to:
- Train new models using Google Colab's GPU resources
- Save training outputs (weights, metrics, visualizations) to the repository
- Keep training runs organized and version-controlled

## Setup for Colab

### 1. Clone the Repository in Colab

```python
!git clone https://github.com/TomsFridrihsons/Basketball_shot_analytics_CV.git
!cd Basketball_shot_analytics_CV
!git checkout colab-training
```

### 2. Install Dependencies

```python
!pip install ultralytics
!pip install torch torchvision
```

### 3. Training Script Template

```python
from ultralytics import YOLO
import os

# Set up paths
os.chdir('/content/Basketball_shot_analytics_CV')

# Load base model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.

# Train on Hooper-2 dataset
results = model.train(
    data='Hooper-2/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='runs/train',
    name='basketball-colab-run1',
    save=True,
    plots=True
)

# Save best model to models directory
best_model_path = results.save_dir / 'weights' / 'best.pt'
os.makedirs('models', exist_ok=True)
import shutil
shutil.copy(best_model_path, 'models/basketball-colab-best.pt')
```

### 4. Save Training Results

After training, commit and push:

```python
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

!git add runs/train/basketball-colab-run1/
!git add models/basketball-colab-best.pt
!git commit -m "Add training run: basketball-colab-run1"
!git push origin colab-training
```

## Directory Structure

```
Basketball_shot_analytics_CV/
├── runs/
│   └── train/              # Training outputs (gitignored by default)
│       └── basketball-colab-run1/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.png
│           ├── confusion_matrix.png
│           └── ...
├── models/                  # Best models (can be committed)
│   └── basketball-colab-best.pt
└── Hooper-2/               # Training dataset
```

## Git Workflow

### Committing Training Results

1. **After training completes**, copy best model to `models/`:
   ```python
   import shutil
   shutil.copy('runs/train/run_name/weights/best.pt', 'models/model-name.pt')
   ```

2. **Commit training outputs** (optional - can be large):
   ```bash
   git add runs/train/your-run-name/
   git add models/your-model.pt
   git commit -m "Training run: description"
   git push origin colab-training
   ```

3. **Or just commit the best model**:
   ```bash
   git add models/your-model.pt
   git commit -m "Add trained model: description"
   git push origin colab-training
   ```

## Notes

- Training outputs in `runs/` are gitignored by default to keep repo size manageable
- Only commit best models to `models/` directory
- Use descriptive commit messages with training details (epochs, dataset, metrics)
- Consider using GitHub Releases for final model versions

## Merging to Main

When you have a good model trained:

```bash
git checkout main
git merge colab-training
git push origin main
```

Or create a Pull Request on GitHub for review.
