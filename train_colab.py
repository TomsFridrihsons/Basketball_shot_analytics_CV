"""
Training script for Google Colab
Usage: Run this in a Colab notebook cell
"""

from ultralytics import YOLO
import os
from pathlib import Path

def train_basketball_model(
    dataset='Hooper-2',  # or 'HooperAnalytic-1'
    model_size='n',      # n, s, m, l, x
    epochs=100,
    imgsz=640,
    batch=16,
    run_name=None
):
    """
    Train a YOLO model on basketball dataset
    
    Args:
        dataset: Dataset name ('Hooper-2' or 'HooperAnalytic-1')
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        run_name: Custom run name (auto-generated if None)
    """
    
    # Set working directory
    if os.path.exists('/content/Basketball_shot_analytics_CV'):
        os.chdir('/content/Basketball_shot_analytics_CV')
    elif os.path.exists('Basketball_shot_analytics_CV'):
        os.chdir('Basketball_shot_analytics_CV')
    
    # Load base model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Set run name
    if run_name is None:
        run_name = f'basketball-{dataset.lower()}-{model_size}-colab'
    
    # Train model
    print(f"Starting training: {run_name}")
    print(f"Dataset: {dataset}")
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
    
    results = model.train(
        data=f'{dataset}/data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='runs/train',
        name=run_name,
        save=True,
        plots=True,
        val=True,
        patience=50,  # Early stopping patience
    )
    
    # Save best model
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_model_path.exists():
        import shutil
        output_path = models_dir / f'{run_name}-best.pt'
        shutil.copy(best_model_path, output_path)
        print(f"\n✅ Best model saved to: {output_path}")
        return str(output_path)
    else:
        print("\n⚠️ Best model not found")
        return None

# Example usage in Colab:
# from train_colab import train_basketball_model
# train_basketball_model(dataset='Hooper-2', model_size='s', epochs=100)
