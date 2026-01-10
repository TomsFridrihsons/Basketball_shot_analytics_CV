# Post-Training Guide 🎯

This guide explains what to do after your model training is completed in Google Colab.

## ✅ What the Notebook Does Automatically

The notebook already handles these steps automatically:

1. **Training Results Visualization**
   - Displays training curves (loss, mAP, etc.)
   - Shows confusion matrix
   - Prints key metrics (mAP50, mAP50-95, Precision, Recall)

2. **Model Saving**
   - Saves best model to `models/basketball-detection-colab-yolo11s-best.pt`
   - Saves last checkpoint to `models/basketball-detection-colab-yolo11s-last.pt`
   - Full training outputs in `runs/detect/basketball-detection-colab-yolo11s/`

3. **Inference Testing** (optional cell)
   - Tests model on a sample validation image
   - Shows detection results

## 📋 Post-Training Checklist

### 1. Review Training Metrics

Check the displayed metrics:
- **mAP50**: Mean Average Precision at IoU=0.5 (should be > 0.5 for good model)
- **mAP50-95**: Mean Average Precision averaged over IoU 0.5-0.95
- **Precision**: How many detections are correct
- **Recall**: How many objects were found

**Good indicators:**
- ✅ mAP50 > 0.7 (excellent)
- ✅ mAP50 > 0.5 (good)
- ⚠️ mAP50 < 0.5 (may need more training/data)

### 2. Check Training Curves

Look at the `results.png` plot:
- **Loss curves**: Should decrease and stabilize
- **mAP curves**: Should increase over epochs
- **No overfitting**: Validation metrics should track training metrics

### 3. Save Model Weights

The notebook automatically saves models, but you may want to:

#### Option A: Download from Colab
```python
# Download best model to your computer
from google.colab import files
files.download('models/basketball-detection-colab-yolo11s-best.pt')
```

#### Option B: Save to Google Drive
If you mounted Drive, models are already saved. Check:
```
/content/drive/MyDrive/Basketball_shot_analytics_CV/models/
```

#### Option C: Commit to GitHub (if using Git)
```python
# In Colab, after training:
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

# Add model
!git add models/basketball-detection-colab-yolo11s-best.pt

# Commit
!git commit -m "Add trained YOLO11s model - mAP50: X.XX"

# Push (if you have write access)
# !git push origin main
```

### 4. Evaluate on Test Set (Optional)

If you have a test set, evaluate the model:

```python
from ultralytics import YOLO

# Load best model
model = YOLO('models/basketball-detection-colab-yolo11s-best.pt')

# Evaluate on test set
results = model.val(data=DATA_YAML, split='test')
print(f"Test mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
```

### 5. Test on Your Own Images/Videos

```python
from ultralytics import YOLO
from IPython.display import Image, display

# Load model
model = YOLO('models/basketball-detection-colab-yolo11s-best.pt')

# Test on image
results = model('path/to/your/image.jpg')

# Display results
for r in results:
    im_array = r.plot()
    display(Image.fromarray(im_array[..., ::-1]))
    
# Test on video
results = model('path/to/your/video.mp4', save=True)
```

### 6. Export Model for Deployment

Export to different formats:

```python
from ultralytics import YOLO

model = YOLO('models/basketball-detection-colab-yolo11s-best.pt')

# Export to ONNX (for deployment)
model.export(format='onnx')

# Export to TensorRT (for NVIDIA GPUs)
model.export(format='engine')

# Export to CoreML (for iOS)
model.export(format='coreml')
```

### 7. Save Training Logs and Config

```python
import shutil
from pathlib import Path

# Copy training configuration
training_config = Path(results.save_dir) / 'args.yaml'
shutil.copy(training_config, 'models/training-config.yaml')

# Copy training results CSV
results_csv = Path(results.save_dir) / 'results.csv'
if results_csv.exists():
    shutil.copy(results_csv, 'models/training-results.csv')
```

## 🔄 Next Steps Based on Results

### If Model Performs Well (mAP50 > 0.7)

1. ✅ **Save the model** - Download or backup
2. ✅ **Document results** - Note metrics and training parameters
3. ✅ **Test on real data** - Try on your actual use case
4. ✅ **Consider deployment** - Export and integrate into your application

### If Model Needs Improvement (mAP50 < 0.5)

1. **More Training**
   - Increase epochs (try 200-300)
   - Use larger model (yolov11m or yolov11l)

2. **More Data**
   - Add more training images
   - Improve annotation quality
   - Add data augmentation

3. **Hyperparameter Tuning**
   - Adjust learning rate
   - Change batch size
   - Modify augmentation settings

4. **Model Architecture**
   - Try larger model size
   - Use different YOLO version

## 📁 File Organization After Training

```
Basketball_shot_analytics_CV/
├── models/
│   ├── basketball-detection-colab-yolo11s-best.pt  # ✅ Best model
│   ├── basketball-detection-colab-yolo11s-last.pt  # Last checkpoint
│   └── training-config.yaml                         # Training config
│
├── runs/detect/
│   └── basketball-detection-colab-yolo11s/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── results.png              # Training curves
│       ├── confusion_matrix.png     # Confusion matrix
│       ├── args.yaml                # Training arguments
│       └── results.csv              # Detailed metrics
│
└── evaluations/                     # (Optional) Move results here
    ├── metrics/
    └── plots/
```

## 🚀 Deployment Options

### Option 1: Local Inference Script

```python
from ultralytics import YOLO

model = YOLO('models/basketball-detection-colab-yolo11s-best.pt')
results = model('path/to/image.jpg')
results[0].show()
```

### Option 2: Web Application

Use Flask/FastAPI to create a web service:

```python
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('models/basketball-detection-colab-yolo11s-best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle image upload and prediction
    pass
```

### Option 3: Mobile App

Export to CoreML or TensorFlow Lite for mobile deployment.

## 📝 Documentation

After training, document:

1. **Model Information**
   - Model name and version
   - Training date
   - Dataset version
   - Training parameters (epochs, batch size, etc.)

2. **Performance Metrics**
   - mAP50, mAP50-95
   - Precision, Recall
   - Per-class metrics

3. **Training Details**
   - Number of training images
   - Number of validation images
   - Training time
   - GPU used

4. **Usage Instructions**
   - How to load the model
   - Input format requirements
   - Output format

## 🔗 Related Documentation

- [Colab Quick Start](COLAB_QUICK_START.md) - How to open notebook
- [Colab Notebook Setup](COLAB_NOTEBOOK_SETUP.md) - Detailed setup guide
- [Dataset Structure](DATASET_STRUCTURE.md) - Dataset organization

---

**Remember**: Always save your best models and document your results! 🎯
