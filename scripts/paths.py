"""
paths.py  –  Shared project paths
=================================
Every script imports its default paths from here, so the repo works from any
clone location. Paths resolve relative to the repository root.

Scripts in sub-folders make this module importable with:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from paths import MODEL_DIR, ...
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── data folders ─────────────────────────────────────────────────────────────
DATA_DIR         = ROOT / "data"
MODEL_DIR        = DATA_DIR / "model"
RAW_VIDEO_DIR    = DATA_DIR / "raw" / "videos"
YOLO_DATASET_DIR = DATA_DIR / "raw" / "model_dataset"
ANNOTATIONS_DIR  = DATA_DIR / "processed" / "annotations"
PREDICTIONS_DIR  = DATA_DIR / "processed" / "predictions"
EVAL_DIR         = DATA_DIR / "processed" / "eval"
EVALUATION_DIR   = DATA_DIR / "processed" / "evaluation"
DETECTIONS_DIR   = DATA_DIR / "processed" / "detections"   # cached per-frame YOLO boxes
ARCHIVE_MODEL_DIR = ROOT / "archives" / "models"

# ── default models ───────────────────────────────────────────────────────────
YOLO_MODEL = MODEL_DIR / "yolo26m-new-best.pt"
RIM_RULE_CONFIG = MODEL_DIR / "basketball_shot_rim_rule_config.pkl"   # default made/missed predictor


def lstm_artifacts(name: str, best: bool = True) -> dict:
    """Return the model / scaler / config paths saved by lstm_train.py for `name`."""
    model_file = f"{name}_best.keras" if best else f"{name}.keras"
    return {
        "model":  MODEL_DIR / model_file,
        "scaler": MODEL_DIR / f"{name}_scaler.pkl",
        "config": MODEL_DIR / f"{name}_config.pkl",
    }
