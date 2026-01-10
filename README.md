# HooperAI 🏀

A basketball detection and analytics project using YOLO (You Only Look Once) object detection models. This project focuses on detecting basketballs, hoops/baskets, and players in video footage.

## 📋 Project Overview

HooperAI is a computer vision project that uses deep learning to detect and track basketball-related objects in videos. The project includes:

- **Object Detection**: Detects basketballs, hoops/baskets, and players
- **Trained Models**: Pre-trained YOLO models ready for inference
- **Datasets**: Two annotated datasets for training and validation

## 🗂️ Project Structure

```
HooperAI/
├── models/                    # Trained model weights
│   ├── basketball-v1-best.pt
│   ├── basketball-v1-last.pt
│   ├── basketball-v3-best.pt
│   └── basketball-v3-last.pt
├── Hooper-2/                 # Dataset 1: Ball, Hoop, Player detection
│   ├── data.yaml
│   ├── train/               # Training images and labels
│   └── valid/               # Validation images and labels
├── HooperAnalytic-1/         # Dataset 2: Ball, Basket, Player detection
│   ├── data.yaml
│   ├── train/               # Training images and labels
│   └── valid/               # Validation images and labels
├── runs/                     # Training runs (gitignored)
├── archives/                 # Archived training runs
└── new_vids/                 # Video files for processing
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Ultralytics YOLO

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/HooperAI.git
cd HooperAI
```

2. Create a virtual environment:
```bash
python -m venv .venv1
.venv1\Scripts\activate  # On Windows
# or
source .venv1/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install ultralytics
pip install torch torchvision
```

## 📊 Datasets

### Hooper-2 Dataset
- **Classes**: Ball, Hoop, Player
- **Training Images**: 84
- **Validation Images**: 13
- **Source**: [Roboflow Universe](https://universe.roboflow.com/basketball-dataset/hooper)

### HooperAnalytic-1 Dataset
- **Classes**: Ball, Basket, Player
- **Training Images**: 96
- **Validation Images**: 15
- **Source**: [Roboflow Universe](https://universe.roboflow.com/basketball-dataset/hooperanalytic)

## 🤖 Using Trained Models

### Load and Use a Model

```python
from ultralytics import YOLO

# Load the best model
model = YOLO('models/basketball-v1-best.pt')

# Run inference on an image
results = model('path/to/image.jpg')

# Run inference on a video
results = model('path/to/video.mp4')

# Display results
results[0].show()
```

### Training a New Model

```python
from ultralytics import YOLO

# Load a base model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.

# Train on Hooper-2 dataset
results = model.train(
    data='Hooper-2/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## 📈 Model Performance

The project includes several trained models:

- **basketball-v1**: Initial training run
- **basketball-v3**: Improved version
- **basketball-finetune**: Fine-tuned models

Check the `runs/detect/` directory for training metrics and visualizations.

## 🛠️ Development

### Adding New Features

1. Create feature branches from `main`
2. Make your changes
3. Test thoroughly
4. Submit a pull request

### Training New Models

Training outputs are automatically saved to `runs/detect/` (gitignored). To track important training runs, copy model weights to the `models/` directory.

## 📝 License

The datasets are licensed under CC BY 4.0. Please refer to the original dataset sources for specific licensing terms.

## 🙏 Acknowledgments

- Datasets from [Roboflow Universe](https://universe.roboflow.com/)
- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## 📧 Contact

For questions or contributions, please open an issue or submit a pull request.
