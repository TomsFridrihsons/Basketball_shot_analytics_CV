# Google Colab Quick Start Guide 🚀

## Which File to Use?

**Use this file in Google Colab:**
```
notebooks/experiments/train_detection_model_colab.ipynb
```

## How to Open in Google Colab

### Method 1: Direct from GitHub (Recommended)

1. Go to your GitHub repository:
   ```
   https://github.com/TomsFridrihsons/Basketball_shot_analytics_CV
   ```

2. Navigate to the notebook:
   ```
   notebooks/experiments/train_detection_model_colab.ipynb
   ```

3. Click on the notebook file

4. Click **"Open in Colab"** button (if available) or:
   - Copy the raw file URL
   - Go to https://colab.research.google.com/
   - Click "File" → "Open notebook"
   - Paste the GitHub URL

### Method 2: Upload to Colab

1. Go to https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Upload `train_detection_model_colab.ipynb` from your local machine

### Method 3: Clone Repository in Colab

1. Open a new Colab notebook
2. Run this cell:
   ```python
   !git clone https://github.com/TomsFridrihsons/Basketball_shot_analytics_CV.git
   %cd Basketball_shot_analytics_CV
   ```
3. Navigate to: `notebooks/experiments/train_detection_model_colab.ipynb`
4. Open the notebook

## ✅ Verification Checklist

- [x] Notebook is in GitHub: `notebooks/experiments/train_detection_model_colab.ipynb`
- [x] Notebook is tracked in Git (not ignored)
- [x] API keys are secure (no hardcoded values)
- [x] All dependencies are included
- [x] Roboflow integration is configured

## 📋 Quick Setup Steps

1. **Open notebook in Colab** (use Method 1 above)

2. **Set API Key**:
   - Click 🔑 icon in Colab sidebar
   - Add secret: `ROBOFLOW_API_KEY` = `your_api_key`
   - Or enter when prompted in the notebook

3. **Run all cells** sequentially:
   - Cell 1: Mount Drive (optional)
   - Cell 2: Clone repo (if using Method 3)
   - Cell 3: Install packages
   - Cell 4: Configure API key
   - Cell 5: Download dataset
   - Cell 6: Verify dataset
   - Cell 7: Train model
   - Cell 8: View results
   - Cell 9: Save model
   - Cell 10: Test inference

## 🔗 Direct Links

- **GitHub Repository**: https://github.com/TomsFridrihsons/Basketball_shot_analytics_CV
- **Notebook Path**: `notebooks/experiments/train_detection_model_colab.ipynb`
- **Raw Notebook URL**: 
  ```
  https://github.com/TomsFridrihsons/Basketball_shot_analytics_CV/blob/main/notebooks/experiments/train_detection_model_colab.ipynb
  ```

## 📝 Notes

- The notebook is **already in GitHub** and ready to use
- It's **not ignored** by `.gitignore` (only `.ipynb_checkpoints` are ignored)
- API keys are **secure** - no hardcoded values
- All your Roboflow settings are **pre-configured**

## 🆘 Troubleshooting

### Can't find the notebook?
- Check: `notebooks/experiments/train_detection_model_colab.ipynb`
- Verify it's in the `main` branch

### Notebook not opening in Colab?
- Use the raw GitHub URL
- Or download and upload manually

### API key issues?
- Use Colab secrets (recommended)
- Or enter when prompted in the notebook

---

**Ready to train!** 🎯 Just open the notebook and follow the cells.
