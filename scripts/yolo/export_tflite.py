"""
export_tflite.py  –  Export the YOLO detector to INT8 TFLite (for the mobile app)
"""

import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import YOLO_MODEL, YOLO_DATASET_DIR  # noqa: E402

model = YOLO(str(YOLO_MODEL))

# Full INT8 quantization; the dataset yaml supplies calibration images
model.export(format="tflite", int8=True, data=str(YOLO_DATASET_DIR / "data.yaml"))
