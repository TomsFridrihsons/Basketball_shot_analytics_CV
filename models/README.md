# Trained Models

This directory contains organized copies of trained model weights.

## Models

- `basketball-v1-best.pt` - Best model from basketball-v1 training run
- `basketball-v1-last.pt` - Last checkpoint from basketball-v1 training run
- `basketball-v3-best.pt` - Best model from basketball-v3 training run
- `basketball-v3-last.pt` - Last checkpoint from basketball-v3 training run

## Original Locations

These models were copied from:
- `runs/detect/basketball-v1/weights/`
- `runs/detect/basketball-v3/weights/`

## Usage

To use these models with YOLO/Ultralytics:

```python
from ultralytics import YOLO

# Load the best model
model = YOLO('models/basketball-v1-best.pt')
```
