# Google Colab Notebook Setup Guide

This guide explains how to use the Colab notebook for training detection models.

## 📓 Notebook Location

The training notebook is located at:
```
notebooks/experiments/train_detection_model_colab.ipynb
```

## 🔐 Security: API Keys

### ⚠️ Important Security Notes

**NEVER commit API keys directly in code!** The notebook includes secure methods for handling API keys.

### Method 1: Colab Secrets (Recommended)

1. In Colab, click the **key icon** (🔑) in the left sidebar
2. Click **"Add new secret"**
3. Name: `ROBOFLOW_API_KEY`
4. Value: Your Roboflow API key
5. The notebook will automatically use this secret

### Method 2: Environment Variables

You can also set the API key as an environment variable in Colab:

```python
import os
os.environ['ROBOFLOW_API_KEY'] = 'your_api_key_here'
```

### Method 3: Direct Assignment (Not Recommended)

Only use direct assignment for personal notebooks that won't be shared:

```python
ROBOFLOW_API_KEY = "your_api_key_here"  # ⚠️ Remove before sharing!
```

## 📋 Setup Steps

### 1. Open Notebook in Colab

- Upload `notebooks/experiments/train_detection_model_colab.ipynb` to Google Colab
- Or clone the repository in Colab and open the notebook

### 2. Configure API Key

- Use Colab secrets (recommended) or set environment variable
- Update project configuration if needed

### 3. Run Cells Sequentially

The notebook is organized into sections:
1. Setup and Installation
2. Configuration
3. Download Dataset
4. Verify Dataset
5. Train Model
6. View Results
7. Save Model
8. Optional: Backup to Drive
9. Test Inference

### 4. Monitor Training

- Training progress is displayed in real-time
- Metrics are logged and visualized
- Results are saved automatically

## 🗂️ File Structure

```
notebooks/experiments/
└── train_detection_model_colab.ipynb  # Main training notebook

configs/
└── secrets.example.yaml                # Template for secrets (gitignored)

env.example                             # Environment variables template
```

## 🔒 Git Ignore Rules

The following files are excluded from Git to protect sensitive data:

- `.env` - Environment variables
- `.env.*` - All environment variable files
- `configs/secrets.yaml` - Secrets configuration
- `configs/api_keys.yaml` - API keys
- `*.key`, `*.secret` - Key files
- `secrets/` - Secrets directory

**Templates are included:**
- `env.example` - Environment variables template
- `configs/secrets.example.yaml` - Secrets template

## 📝 Configuration

### Roboflow Settings

Update these in the notebook configuration section:

```python
ROBOFLOW_WORKSPACE = "basketball-dataset"
ROBOFLOW_PROJECT = "hooper_dataset_3006_2235"
ROBOFLOW_VERSION = 3
```

### Training Settings

```python
MODEL_SIZE = "n"      # n, s, m, l, x
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
PROJECT_NAME = "basketball-detection-colab"
```

## 🚀 Usage

1. **Open notebook in Colab**
2. **Set API key** using Colab secrets
3. **Run all cells** sequentially
4. **Monitor training** progress
5. **Save model** weights
6. **Test inference** on sample images

## 📊 Outputs

After training, you'll have:

- **Model weights**: `models/{PROJECT_NAME}-best.pt`
- **Training outputs**: `runs/detect/{PROJECT_NAME}/`
  - `weights/` - Model checkpoints
  - `results.png` - Training curves
  - `confusion_matrix.png` - Confusion matrix
  - `args.yaml` - Training configuration

## 🔄 Version Control

### Committing Models

If you want to commit trained models:

```bash
git add models/your-model-best.pt
git commit -m "Add trained model: description"
git push
```

### Excluding Training Outputs

Training outputs in `runs/` are gitignored by default. To commit specific runs:

```bash
# Temporarily remove from .gitignore or use git add -f
git add -f runs/detect/your-run-name/
```

## 🛠️ Troubleshooting

### API Key Issues

- **Error**: "Invalid API key"
  - Check that API key is set correctly in Colab secrets
  - Verify API key is active in Roboflow dashboard

### Dataset Download Issues

- **Error**: "Dataset not found"
  - Verify workspace, project, and version names
  - Check API key has access to the dataset

### GPU Issues

- **No GPU available**: Colab may assign CPU-only runtime
  - Go to Runtime → Change runtime type → GPU
  - Free tier has limited GPU availability

### Memory Issues

- **Out of memory**: Reduce batch size
  - Set `BATCH_SIZE = 8` or lower
  - Use smaller model (`MODEL_SIZE = "n"`)

## 📚 Related Documentation

- [Colab Setup Guide](colab_setup.md) - General Colab setup
- [Dataset Structure](DATASET_STRUCTURE.md) - Dataset organization
- [Training Configs](../configs/README.md) - Configuration files

## ⚠️ Security Checklist

Before sharing your notebook:

- [ ] Remove any hardcoded API keys
- [ ] Use Colab secrets for sensitive data
- [ ] Remove personal information
- [ ] Clear outputs if they contain sensitive data
- [ ] Review all cells for exposed credentials

---

**Remember**: Never commit API keys or secrets to Git!
