"""
lstm_infer.py  –  Basketball Shot Prediction: Real-Time Inference
=================================================================
Loads the model produced by lstm_train.py and runs a full
YOLO + LSTM inference pipeline on a video file.

Combines the best from both branches:
  - Dev branch:  YOLO integration, shot-detection state machine,
                 confidence levels, per-shot stats, output saving,
                 weight reconstruction, optimal threshold
  - Main branch: clean config loading, RobustScaler, per-shot CSV,
                 comprehensive statistics JSON, summary .txt

Usage:
    python scripts/lstm/lstm_infer.py
    (edit the paths in the CONFIGURATION block below)
"""

import os
import json
import pickle
import zipfile
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ─── paths ───────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (YOLO_MODEL as _YOLO, RAW_VIDEO_DIR, PREDICTIONS_DIR,  # noqa: E402
                   ANNOTATIONS_DIR, DETECTIONS_DIR, RIM_RULE_CONFIG, lstm_artifacts)
import shot_features as sf  # noqa: E402

_LSTM = lstm_artifacts("basketball_shot_lstm_v6")
YOLO_MODEL  = str(_YOLO)
LSTM_MODEL  = str(_LSTM["model"])     # ignored when CONFIG_PATH is a rim-rule config
SCALER_PATH = str(_LSTM["scaler"])
CONFIG_PATH = str(RIM_RULE_CONFIG)   # default predictor; use _LSTM["config"] for the LSTM
VIDEO_PATH  = str(RAW_VIDEO_DIR / "20250711_171647.mp4")
OUTPUT_DIR  = str(PREDICTIONS_DIR)

YOLO_CONF        = 0.75   # YOLO detection confidence threshold
DISPLAY          = True   # show OpenCV window
SAVE_VIDEO       = True   # write annotated video to output folder
FEEDBACK_CSV     = str(ANNOTATIONS_DIR / "feedback_labels.csv")
USE_DET_CACHE    = True   # reuse cached YOLO boxes on headless runs (display/save off)
# ─────────────────────────────────────────────────────────────────────────────


# ── Custom serialisable layers (must match lstm_train.py) ────────────────────
@keras.utils.register_keras_serializable()
class MaskedAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1, activation='tanh')

    def call(self, x, mask=None):
        scores = self.score_dense(x)
        scores = tf.squeeze(scores, axis=-1)
        if mask is not None:
            mask_float = tf.cast(mask, tf.float32)
            scores = scores * mask_float + (1.0 - mask_float) * (-1e9)
        weights = tf.nn.softmax(scores, axis=-1)
        weights = tf.expand_dims(weights, axis=-1)
        return tf.reduce_sum(x * weights, axis=1)

    def compute_mask(self, x, mask=None):
        return None   # time axis is reduced away; nothing to propagate

    def get_config(self):
        return super().get_config()


@keras.utils.register_keras_serializable()
class SumAlongAxis(layers.Layer):
    def __init__(self, axis: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.reduce_sum(x, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


# ── Loss (needed for model.compile in weight-reconstruction path) ─────────────

def focal_loss(gamma: float = 2.0, alpha: float = 0.3):
    def _loss(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce     = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        weight  = tf.pow(1 - p_t, gamma)
        alpha_w = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_w * weight * bce)
    return _loss


# ── Model reconstruction helper ───────────────────────────────────────────────
def _rebuild_model(max_len: int, n_features: int, cfg: dict) -> keras.Model:
    """
    Rebuild architecture from config dict.
    Mirrors lstm_train.py exactly by removing BatchNorm from the LSTM stack
    to resolve weight dimension mismatch bugs.
    """
    units = cfg.get('lstm_units', 128)
    drop = cfg.get('dropout_rate', 0.4)
    l2 = cfg.get('l2_reg', 0.001)
    bidir = cfg.get('use_bidirectional', True)
    attention = cfg.get('use_attention', True)
    reg = regularizers.l2(l2)

    inputs = layers.Input(shape=(max_len, n_features), name="sequence_input")
    x = layers.Masking(mask_value=0.0)(inputs)

    # ── LSTM Block 1 (BatchNorm Removed to match training) ───────────────────
    lstm1 = layers.LSTM(units, return_sequences=True, kernel_regularizer=reg, name="lstm_1")
    x = layers.Bidirectional(lstm1, name="bi_lstm_1")(x) if bidir else lstm1(x)
    x = layers.Dropout(drop)(x)

    # ── LSTM Block 2 (BatchNorm Removed to match training) ───────────────────
    lstm2 = layers.LSTM(units // 2, return_sequences=True, kernel_regularizer=reg, name="lstm_2")
    x = layers.Bidirectional(lstm2, name="bi_lstm_2")(x) if bidir else lstm2(x)
    x = layers.Dropout(drop)(x)

    # ── Attention or Final LSTM Stack ────────────────────────────────────────
    if attention:
        x = MaskedAttention(name="masked_attention")(x)
    else:
        lstm3 = layers.LSTM(units // 4, return_sequences=False, kernel_regularizer=reg, name="lstm_3")
        x = layers.Bidirectional(lstm3, name="bi_lstm_3")(x) if bidir else lstm3(x)
        x = layers.Dropout(drop)(x)

    # ── Dense Head (BatchNorm kept here as per training configuration) ───────
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(drop)(x)

    x = layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(drop / 2)(x)

    outputs = layers.Dense(1, activation='sigmoid', name="shot_result")(x)

    model = keras.Model(inputs, outputs, name="ShotPredictionLSTM_v3")

    # Compile with dummy metrics just to initialize internal model state
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.get('learning_rate', 0.001)),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model

def load_lstm(model_path: str, config: dict) -> keras.Model:
    """
    Load LSTM model.  Tries direct keras load first; falls back to
    weight reconstruction (handles Lambda/custom-layer serialisation issues).
    """
    max_len    = config['max_len']
    n_features = config['n_features']
    model_cfg  = config.get('model_config', {})

    # ── attempt 1: direct load ────────────────────────────────────────────────
    try:
        keras.config.enable_unsafe_deserialization()
        model = keras.models.load_model(
            model_path,
            custom_objects={'SumAlongAxis': SumAlongAxis,
                            'MaskedAttention': MaskedAttention,
                            'focal_loss_fixed': focal_loss()},
        )
        print("  ✅ Direct load succeeded")
        return model
    except Exception as e:
        print(f"  ⚠️  Direct load failed ({e}), attempting weight reconstruction…")

    # ── attempt 2: rebuild + load weights from .keras zip ─────────────────────
    model = _rebuild_model(max_len, n_features, model_cfg)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(model_path, 'r') as zf:
            zf.extractall(tmp)
        weights_h5 = os.path.join(tmp, 'model.weights.h5')
        if not os.path.exists(weights_h5):
            available = os.listdir(tmp)
            raise FileNotFoundError(
                f"model.weights.h5 not found in archive. Contents: {available}"
            )
        try:
            model.load_weights(weights_h5)
            print("  ✅ Weight reconstruction succeeded")
        except Exception:
            model.load_weights(weights_h5, skip_mismatch=True)
            print("  ✅ Weight reconstruction succeeded (skip_mismatch)")

    return model


# ── Main inference class ──────────────────────────────────────────────────────

class RealTimeShotPredictor:
    """
    YOLO + LSTM real-time basketball shot predictor.

    Shot detection uses a two-stage state machine (possible → confirmed)
    to avoid false triggers from incidental ball elevation.
    """

    FEATURE_COLS = [
        'ball_x', 'ball_y', 'ball_w', 'ball_h',
        'ball_velocity_x', 'ball_velocity_y',
        'player_x', 'player_y', 'player_w', 'player_h',
        'basket_x', 'basket_y', 'basket_w', 'basket_h',
        'ball_player_dist', 'ball_basket_dist', 'ball_basket_angle',
        'ball_above_player',
    ]

    # YOLO class indices (adjust if your model differs)
    CLS_BALL   = 0
    CLS_BASKET = 1
    CLS_PLAYER = 2

    def __init__(
        self,
        yolo_path:    str,
        lstm_path:    str,
        scaler_path:  str,
        config_path:  str,
        output_dir:   str,
        feedback_csv: str = None,
    ):
        print("=" * 60)
        print("INITIALISING SHOT PREDICTOR")
        print("=" * 60)

        # ── YOLO ─────────────────────────────────────────────────────────────
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
        print("\nLoading YOLO …")
        self.yolo = YOLO(yolo_path)
        self.yolo_path = yolo_path
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.half   = self.device != 'cpu'      # FP16 on GPU: ~2x faster, same boxes
        print(f"  ✅ {yolo_path}  (device: {'cuda' if self.half else 'cpu'})")
        if not self.half:
            print("  ⚠️  CUDA not available – YOLO runs on CPU (slow). "
                  "Install a CUDA build of PyTorch that matches your driver.")

        # ── Config ───────────────────────────────────────────────────────────
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)

        self.max_seq_len       = self.config.get('max_len', 68)
        self.feature_cols      = self.config.get('feature_cols', self.FEATURE_COLS)
        self.optimal_threshold = self.config.get('optimal_threshold', 0.5)
        # v2 models (lstm_train.py v6+) build features with shot_features.py;
        # older models take the raw pixel columns directly.
        self.feature_version   = self.config.get('feature_version', 1)

        # Rim-pass rule configs (lstm_train.py) need no network or scaler
        self.model_type = self.config.get('model_type', 'lstm')
        if self.model_type == 'rim_rule':
            self.lstm = self.scaler = None
            self.rim_threshold = self.config['rim_threshold']
            print(f"\n  ✅ Rim-pass rule (made if rim offset < {self.rim_threshold:.2f} basket widths)")
        else:
            # ─── 2. LOAD LSTM MODEL ───────────────────────────────────────────────
            print("Loading LSTM …")

            def focal_loss(gamma=2.0, alpha=0.25):
                def _loss(y_true, y_pred):
                    y_true = tf.cast(y_true, tf.float32)
                    bce = keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
                    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
                    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
                    modulating_factor = tf.pow(1.0 - p_t, gamma)
                    return tf.reduce_mean(alpha_factor * modulating_factor * bce)

                return _loss

            try:
                # Clean direct loading: Forces Keras to parse the model structure
                # exactly as it was compiled during training.
                # FIXED: Assigned directly to self.lstm to match downstream inference calls
                self.lstm = keras.models.load_model(
                    lstm_path,
                    custom_objects={
                        "MaskedAttention": MaskedAttention,
                        "_loss": focal_loss()
                    },
                    compile=False
                )
                print(f"   ✅ Model loaded cleanly via serialization from: {lstm_path}")

            except Exception as e:
                print(f"   ⚠️ Direct structural load failed: {e}")
                print("   Falling back to manual layer-by-layer reconstruction...")

                # Legacy fallback if your file was saved using old weights format
                self.lstm = _rebuild_model(self.max_seq_len, len(self.feature_cols),
                                           self.config.get('model_config', {}))

                # Attempt to safely load weights onto the rebuilt structure
                try:
                    self.lstm.load_weights(lstm_path, skip_mismatch=True)
                    print("   ✅ Weights mounted successfully via legacy reconstruction path.")
                except Exception as weight_err:
                    print(f"   ❌ Fatal error loading weights to structure: {weight_err}")
                    raise weight_err
            # ── Scaler ───────────────────────────────────────────────────────────
            if scaler_path and os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                parts = self.scaler.values() if isinstance(self.scaler, dict) else [self.scaler]
                fitted = all(hasattr(s, 'center_') or hasattr(s, 'mean_') for s in parts)
                print(f"\n  ✅ Scaler loaded ({'fitted' if fitted else 'NOT FITTED – predictions will be off'})")
            else:
                self.scaler = None
                print("\n  ⚠️  No scaler found – predictions will be inaccurate!")

        # ── Output ───────────────────────────────────────────────────────────
        self.output_base = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # ── Feedback (human-labelled training data) ───────────────────────────
        self.feedback_csv = feedback_csv
        self._feedback_rows: list = []   # accumulated this session

        # ── State ────────────────────────────────────────────────────────────
        self._reset_shot_state()
        self._reset_stats()

        # Last known basket position (fallback when not detected)
        self._last_basket = None

        # Display sizing
        self.DISPLAY_W   = 1280
        self.DISPLAY_H   = 900

        print(f"\n{'=' * 60}")
        print("READY")
        print(f"  Threshold : {self.optimal_threshold}")
        print(f"  Max seq   : {self.max_seq_len}")
        print(f"{'=' * 60}")

    # ── State management ──────────────────────────────────────────────────────

    def _reset_shot_state(self):
        self.possible_shot     = False
        self.actual_shot       = False
        self.poss_frame_count  = 0
        self.pre_data          = []    # frames recorded during 'possible' stage
        self.shot_data         = []    # frames recorded during confirmed shot
        self.frame_idx         = 0
        self.shot_start_frame  = None
        self.last_pred         = None
        self.last_prob         = None

    def _reset_stats(self):
        self.stats = {
            'total_shots':    0,
            'predicted_made': 0,
            'predicted_miss': 0,
            'high_conf':      0,
            'low_conf':       0,
            'shots':          [],
            'start_time':     None,
            'end_time':       None,
            'video_info':     {},
            'model_info': {
                'max_seq_len':       self.max_seq_len,
                'n_features':        len(self.feature_cols),
                'optimal_threshold': self.optimal_threshold,
            },
        }

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(self, ball_boxes, player_boxes, basket_boxes, fps: float):
        """Extract the 18-feature vector for one frame."""

        # Ball
        if ball_boxes:
            bx1, by1, bx2, by2 = ball_boxes[0]
            ball_x = (bx1 + bx2) / 2;  ball_y = (by1 + by2) / 2
            ball_w = bx2 - bx1;         ball_h = by2 - by1
        else:
            ball_x = ball_y = ball_w = ball_h = None

        # Player closest to ball
        player_x = player_y = player_w = player_h = player_top = None
        if player_boxes and ball_x is not None:
            cp = min(player_boxes, key=lambda b: np.hypot(
                (b[0]+b[2])/2 - ball_x, (b[1]+b[3])/2 - ball_y))
            player_x = (cp[0]+cp[2])/2;  player_y = (cp[1]+cp[3])/2
            player_w = cp[2]-cp[0];       player_h = cp[3]-cp[1]
            player_top = cp[1]

        # Basket (with last-known fallback)
        if basket_boxes:
            bkx1, bky1, bkx2, bky2 = basket_boxes[0]
            basket_x = (bkx1+bkx2)/2;  basket_y = (bky1+bky2)/2
            basket_w = bkx2-bkx1;       basket_h = bky2-bky1
            self._last_basket = (basket_x, basket_y, basket_w, basket_h)
        elif self._last_basket:
            basket_x, basket_y, basket_w, basket_h = self._last_basket
        else:
            basket_x = basket_y = basket_w = basket_h = None

        # Derived
        ball_player_dist = ball_basket_dist = ball_basket_angle = ball_above_player = None
        if ball_x is not None and player_x is not None:
            ball_player_dist  = np.hypot(ball_x - player_x, ball_y - player_y)
            ball_above_player = 1 if ball_y < player_top else 0
        if ball_x is not None and basket_x is not None:
            ball_basket_dist  = np.hypot(ball_x - basket_x, ball_y - basket_y)
            ball_basket_angle = np.arctan2(basket_y - ball_y, basket_x - ball_x)

        # Velocity (from previous frame in buffer)
        ball_vx = ball_vy = None
        prev = (self.shot_data or self.pre_data or [None])[-1]
        if prev is not None and ball_x is not None and prev[0] is not None:
            dt = 1.0 / fps if fps > 0 else 1/30
            ball_vx = (ball_x - prev[0]) / dt
            ball_vy = (ball_y - prev[1]) / dt

        return [
            ball_x, ball_y, ball_w, ball_h,
            ball_vx, ball_vy,
            player_x, player_y, player_w, player_h,
            basket_x, basket_y, basket_w, basket_h,
            ball_player_dist, ball_basket_dist, ball_basket_angle,
            ball_above_player,
        ]

    # ── Shot detection state machine ──────────────────────────────────────────

    def _detect_shot(self, ball_boxes, player_boxes):
        """
        Two-stage detection:
          Stage 1 (possible): ball rises above all player tops.
          Stage 2 (confirmed): ball stays elevated for ≥ 10 frames.
        Returns (possible_started, confirmed_started, shot_ended).
        """
        poss_started = conf_started = ended = False

        if not ball_boxes or not player_boxes:
            if self.possible_shot:
                self.poss_frame_count += 1
                if self.poss_frame_count > 25:
                    self.possible_shot    = False
                    self.poss_frame_count = 0
                    self.pre_data         = []
                    print("[POSSIBLE SHOT CANCELLED – no detections]")
            return poss_started, conf_started, ended

        ball  = ball_boxes[0]
        b_bot = ball[3]
        b_ctr = (ball[1] + ball[3]) / 2
        p_top = min(p[1] for p in player_boxes)

        above = b_bot < p_top
        below = b_ctr > p_top

        if not self.possible_shot and not self.actual_shot and above:
            self.possible_shot    = True
            self.poss_frame_count = 0
            self.pre_data         = []
            self.shot_start_frame = self.frame_idx
            poss_started          = True
            print("\n[POSSIBLE SHOT – monitoring …]")

        if self.possible_shot and not self.actual_shot:
            self.poss_frame_count += 1
            if above:
                if self.poss_frame_count >= sf.CONFIRM_FRAMES:
                    self.actual_shot   = True
                    self.possible_shot = False
                    self.shot_data     = self.pre_data[:]
                    self.pre_data      = []
                    conf_started       = True
                    print(f"[SHOT CONFIRMED – {len(self.shot_data)} pre-frames]")
            else:
                print(f"[POSSIBLE SHOT CANCELLED – only {self.poss_frame_count} frames]")
                self.possible_shot    = False
                self.poss_frame_count = 0
                self.pre_data         = []

        if self.actual_shot and below:
            self.actual_shot = False
            ended            = True

        return poss_started, conf_started, ended

    # ── Human feedback overlay ────────────────────────────────────────────────

    def _ask_feedback(self, frame, pred: int, prob: float, shot_number: int) -> int | None:
        """
        Freeze the video and show a keyboard prompt so the user can correct
        the prediction.  Returns the ground-truth label (0/1) or None if
        the user skips.

        Keys:
          M / 1  → shot was MADE
          X / 0  → shot was MISSED
          S      → skip (keep model prediction, no label saved)
        """
        overlay = frame.copy()
        h, w    = overlay.shape[:2]

        # Dark panel
        cv2.rectangle(overlay, (0, h - 160), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, overlay)

        # Predicted result colour strip
        col = (0, 210, 0) if pred == 1 else (0, 0, 220)
        cv2.rectangle(overlay, (0, h - 160), (w, h - 110), col, -1)
        cv2.putText(overlay,
                    f"Shot #{shot_number}  –  Model says: {'MADE' if pred else 'MISSED'}  ({prob:.1%})",
                    (14, h - 122), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Instruction line
        cv2.putText(overlay,
                    "Was it correct?   M = MADE    X = MISSED    S = Skip",
                    (14, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

        cv2.putText(overlay,
                    "(video paused – waiting for your input)",
                    (14, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

        # Scale to display size
        scale   = min(self.DISPLAY_W / w, self.DISPLAY_H / h)
        display = cv2.resize(overlay, (int(w * scale), int(h * scale)))
        cv2.imshow('Shot Prediction', display)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('m'), ord('M'), ord('1')):
                return 1
            elif key in (ord('x'), ord('X'), ord('0')):
                return 0
            elif key in (ord('s'), ord('S')):
                return None   # user chose to skip

    def _save_feedback_row(self, shot_number: int, label: int,
                           pred: int, prob: float, video_path: str):
        """
        Append one labeled shot to the in-memory feedback list and persist
        the raw frame features so they can be used as training data.
        The shot_data at call time still holds the frame feature vectors.
        """
        cols = self.feature_cols
        for frame_idx, feat in enumerate(self.shot_data):
            row = {c: v for c, v in zip(cols, feat)}
            row['shot_number']    = shot_number
            row['frame_within_shot'] = frame_idx
            row['shot_result']    = label          # human ground truth
            row['model_pred']     = pred
            row['model_prob']     = round(prob, 4)
            row['source_video']   = os.path.basename(video_path)
            self._feedback_rows.append(row)

    def _flush_feedback(self):
        """Write all accumulated feedback rows to the feedback CSV."""
        if not self._feedback_rows or not self.feedback_csv:
            return

        os.makedirs(os.path.dirname(self.feedback_csv), exist_ok=True)
        df_new = pd.DataFrame(self._feedback_rows)

        if os.path.exists(self.feedback_csv):
            df_existing = pd.read_csv(self.feedback_csv)
            # Re-number shot_number so it never collides with existing data
            if not df_existing.empty and 'shot_number' in df_existing.columns:
                offset = int(df_existing['shot_number'].max()) + 1
                df_new['shot_number'] += offset
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_out = df_new

        df_out.to_csv(self.feedback_csv, index=False)
        correct   = sum(1 for r in self._feedback_rows
                        if r['shot_result'] == r['model_pred'])
        total_lab = len({r['shot_number'] for r in self._feedback_rows})
        print(f"\n  ✅ Feedback saved → {self.feedback_csv}")
        print(f"     Labeled shots : {total_lab}")
        print(f"     Model correct : {correct / total_lab * 100:.0f}%"
              if total_lab else "")
        self._feedback_rows = []

    # ── LSTM prediction ───────────────────────────────────────────────────────

    def _predict(self):
        """Run LSTM on current shot_data buffer. Returns (label, probability)."""
        if len(self.shot_data) < 5:
            print(f"[WARNING] Only {len(self.shot_data)} frames – skipping prediction")
            return None, None

        if self.model_type == 'rim_rule':
            offset = sf.rim_pass_offset(sf.shot_to_features(np.array(self.shot_data, dtype=np.float64)))
            # Map the offset to a 0–1 score; 0.5 exactly at the threshold
            prob  = float(1.0 / (1.0 + np.exp((offset - self.rim_threshold) / 0.1)))
            label = 1 if prob >= self.optimal_threshold else 0
            print(f"  Rim offset : {offset:.3f}  (threshold {self.rim_threshold:.2f})")
            print(f"  Prediction : {'MADE' if label else 'MISSED'}")
            return label, prob

        if self.model_type == 'lstm_hybrid':
            # v7: [scaled frame sequence, scaled shot-geometry summary, rule logit]
            feats  = sf.shot_to_features(np.array(self.shot_data, dtype=np.float64))
            seq    = self.scaler['seq'].transform(feats).astype(np.float32)
            summ   = self.scaler['summary'].transform(sf.shot_summary(feats)[None]).astype(np.float32)
            rule   = np.clip((self.scaler['rim_threshold'] - sf.rim_pass_offset(feats))
                             / self.config.get('rule_temp', 0.1), -6, 6)
            rule   = np.array([[rule]], dtype=np.float32)
            prob   = float(self.lstm([sf.pad_tail([seq], self.max_seq_len), summ, rule],
                                     training=False).numpy()[0][0])
            label  = 1 if prob >= self.optimal_threshold else 0
            print(f"  Raw prob   : {prob:.4f}  (threshold {self.optimal_threshold:.3f})")
            print(f"  Prediction : {'MADE' if label else 'MISSED'}")
            return label, prob

        if self.feature_version >= 2:
            # Same transform as training: features → scale real frames → tail-pad
            seq = sf.shot_to_features(np.array(self.shot_data, dtype=np.float64))
            if self.scaler is not None:
                seq = self.scaler.transform(seq).astype(np.float32)
            padded = sf.pad_tail([seq], self.max_seq_len)
        else:
            seq = np.array(self.shot_data, dtype=np.float32)
            seq = np.nan_to_num(seq, nan=0.0)

            n_feat = len(self.feature_cols)
            padded = np.zeros((1, self.max_seq_len, n_feat), dtype=np.float32)
            L      = min(len(seq), self.max_seq_len)
            padded[0, :L, :] = seq[:L]

            # Scale
            if self.scaler is not None:
                is_fitted = hasattr(self.scaler, 'center_') or hasattr(self.scaler, 'mean_')
                if is_fitted:
                    flat   = padded.reshape(-1, n_feat)
                    flat   = self.scaler.transform(flat)
                    padded = flat.reshape(1, self.max_seq_len, n_feat)

        # Direct call: model.predict() has ~50 ms of per-call overhead
        prob  = float(self.lstm(padded, training=False).numpy()[0][0])
        label = 1 if prob >= self.optimal_threshold else 0

        print(f"  Raw prob   : {prob:.4f}")
        print(f"  Threshold  : {self.optimal_threshold:.3f}")
        print(f"  Prediction : {'MADE' if label else 'MISSED'}")
        return label, prob

    # ── Confidence categorisation ─────────────────────────────────────────────

    def _confidence(self, prob: float):
        dist = abs(prob - self.optimal_threshold)
        if   dist > 0.30: return 'HIGH'
        elif dist > 0.15: return 'MEDIUM'
        else:             return 'LOW'

    # ── Output helpers ────────────────────────────────────────────────────────

    def _make_output_folder(self, video_path: str):
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        name        = os.path.splitext(os.path.basename(video_path))[0]
        out_folder  = os.path.join(self.output_base, f"{ts}_{name}")
        os.makedirs(out_folder, exist_ok=True)
        return out_folder, ts

    def _det_cache_path(self, video_path: str) -> Path:
        """Cache key: video name + size, YOLO weights, confidence threshold."""
        video = Path(video_path)
        size  = video.stat().st_size if video.exists() else 0
        name  = f"{video.stem}_{size}_{Path(self.yolo_path).stem}_conf{YOLO_CONF:.2f}.npz"
        return DETECTIONS_DIR / name

    def _save_outputs(self, out_folder: str, video_path: str):
        """Save statistics.json, shots.csv, summary.txt."""
        total = self.stats['total_shots']
        made  = self.stats['predicted_made']
        miss  = self.stats['predicted_miss']
        pct   = (made / total * 100) if total > 0 else 0.0

        # Processing duration
        dur = None
        if self.stats['start_time'] and self.stats['end_time']:
            s   = datetime.fromisoformat(self.stats['start_time'])
            e   = datetime.fromisoformat(self.stats['end_time'])
            dur = round((e - s).total_seconds(), 2)

        self.stats['summary'] = {
            'total_shots':            total,
            'predicted_made':         made,
            'predicted_miss':         miss,
            'predicted_make_pct':     round(pct, 2),
            'high_conf_predictions':  self.stats['high_conf'],
            'low_conf_predictions':   self.stats['low_conf'],
            'input_video':            video_path,
            'threshold_used':         self.optimal_threshold,
            'processing_seconds':     dur,
        }

        # JSON
        json_path = os.path.join(out_folder, "statistics.json")
        with open(json_path, 'w') as f:
            json.dump(self.stats, f, indent=4, default=str)

        # CSV
        if self.stats['shots']:
            csv_path = os.path.join(out_folder, "shots.csv")
            pd.DataFrame(self.stats['shots']).to_csv(csv_path, index=False)
            print(f"  Shots CSV → {csv_path}")

        # TXT summary
        txt_path = os.path.join(out_folder, "summary.txt")
        with open(txt_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("BASKETBALL SHOT PREDICTION – SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input video        : {os.path.basename(video_path)}\n")
            f.write(f"Processing time    : {dur}s\n")
            f.write(f"Threshold used     : {self.optimal_threshold:.3f}\n\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total shots        : {total}\n")
            f.write(f"Predicted made     : {made}\n")
            f.write(f"Predicted missed   : {miss}\n")
            f.write(f"Make percentage    : {pct:.1f}%\n")
            f.write(f"High confidence    : {self.stats['high_conf']}\n")
            f.write(f"Low confidence     : {self.stats['low_conf']}\n")

        print(f"  Stats JSON → {json_path}")
        print(f"  Summary    → {txt_path}")
        return json_path, txt_path

    # ── Video processing ──────────────────────────────────────────────────────

    def process_video(self, video_path: str, display: bool = True,
                      save_video: bool = True, verbose: bool = True,
                      progress_cb=None):
        """Main entry point. Processes one video file end-to-end.

        progress_cb(frame_idx, total_frames) is called every 30 frames (used by
        scripts/app/server.py). Raw frames of each predicted shot are kept in
        self.shot_frames[shot_number] so a UI can turn corrections into training rows.
        """
        self._reset_stats()
        self._reset_shot_state()
        self.shot_frames = {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.stats['video_info'] = {
            'path': video_path, 'fps': fps,
            'total_frames': total_frames, 'width': fw, 'height': fh,
            'duration_s': round(total_frames / fps if fps else 0, 2),
        }

        out_folder, _ = self._make_output_folder(video_path)

        writer = None
        if save_video:
            vid_out = os.path.join(out_folder, "processed_video.mp4")
            writer  = cv2.VideoWriter(vid_out, cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps, (fw, fh))

        self.stats['start_time'] = datetime.now().isoformat()

        print(f"\n{'=' * 60}")
        print("PROCESSING VIDEO")
        print(f"  {video_path}")
        print(f"  {fps:.1f} fps  |  {total_frames} frames  |  output → {out_folder}")
        print(f"  Threshold : {self.optimal_threshold:.3f}")
        if display:
            print("  [q] quit   [r] reset")

        last_pct = -1

        # Headless runs (e.g. lstm_eval.py) skip YOLO and video decoding entirely
        # by replaying cached boxes, so a new LSTM is evaluated in seconds.
        headless   = not display and not save_video
        cache_path = self._det_cache_path(video_path)
        cached     = None
        if headless and USE_DET_CACHE and cache_path.exists():
            cached = np.load(cache_path)['boxes']   # rows: frame, cls, x1, y1, x2, y2
            n_cached = int(cached[:, 0].max()) if len(cached) else total_frames
            total_frames = max(total_frames, n_cached)
            # frame_rows[k-1] = boxes of frame k (cache rows are sorted by frame)
            frame_rows = np.split(cached, np.searchsorted(cached[:, 0], np.arange(1, total_frames + 1)))[1:]
            print(f"  Using cached detections → {cache_path}")
        new_cache   = [] if cached is None else None
        reached_end = False

        while True:
            if cached is None:
                ret, frame = cap.read()
                if not ret:
                    reached_end = True
                    break
            elif self.frame_idx >= total_frames:
                break

            self.frame_idx += 1

            # Progress
            if progress_cb and self.frame_idx % 30 == 0:
                progress_cb(self.frame_idx, total_frames)
            pct = int(self.frame_idx / total_frames * 100)
            if pct % 10 == 0 and pct != last_pct:
                print(f"  [{pct}%]")
                last_pct = pct

            # YOLO detection (boxes come sorted by confidence)
            if cached is None:
                results  = self.yolo(frame, conf=YOLO_CONF, verbose=False,
                                     device=self.device, half=self.half)[0]
                cls_arr  = results.boxes.cls.cpu().numpy()
                xyxy_arr = results.boxes.xyxy.cpu().numpy()
                new_cache.append(np.column_stack(
                    [np.full(len(cls_arr), self.frame_idx), cls_arr, xyxy_arr]))
            else:
                rows = frame_rows[self.frame_idx - 1]
                cls_arr, xyxy_arr = rows[:, 1], rows[:, 2:6]

            ball_boxes   = list(xyxy_arr[cls_arr == self.CLS_BALL])
            basket_boxes = list(xyxy_arr[cls_arr == self.CLS_BASKET])
            player_boxes = list(xyxy_arr[cls_arr == self.CLS_PLAYER])

            # Shot detection
            poss_start, conf_start, ended = self._detect_shot(ball_boxes, player_boxes)

            if poss_start:
                self.pre_data = []
            if conf_start:
                self.last_pred = self.last_prob = None

            # Feature collection
            if self.possible_shot or self.actual_shot:
                feats = self._extract_features(ball_boxes, player_boxes,
                                               basket_boxes, fps)
                if self.possible_shot:
                    self.pre_data.append(feats)
                else:
                    self.shot_data.append(feats)

            # Prediction at shot end
            if ended and self.shot_data:
                pred, prob = self._predict()
                self.last_pred = pred
                self.last_prob = prob

                if pred is not None:
                    result_txt  = "MADE" if pred == 1 else "MISSED"
                    conf_level  = self._confidence(prob)
                    if verbose:
                        print(f"  [SHOT #{self.stats['total_shots']+1}] "
                              f"{result_txt}  prob={prob:.2%}  conf={conf_level}")

                    self.stats['total_shots'] += 1
                    self.stats['predicted_made' if pred else 'predicted_miss'] += 1
                    self.stats['high_conf' if conf_level == 'HIGH' else 'low_conf'] += 1

                    self.stats['shots'].append({
                        'shot_number':     self.stats['total_shots'],
                        'prediction':      int(pred),
                        'prediction_text': result_txt,
                        'probability':     round(prob, 4),
                        'confidence':      conf_level,
                        'start_frame':     self.shot_start_frame,
                        'end_frame':       self.frame_idx,
                        'n_frames':        len(self.shot_data),
                    })
                    self.shot_frames[self.stats['total_shots']] = list(self.shot_data)

                    # ── Human feedback ────────────────────────────────────────
                    if display and self.feedback_csv:
                        true_label = self._ask_feedback(
                            annotated, pred, prob, self.stats['total_shots'])
                        if true_label is not None:
                            agreement = "✅ correct" if true_label == pred else "❌ wrong"
                            print(f"     Label: {'MADE' if true_label else 'MISSED'}"
                                  f"  ({agreement})")
                            self._save_feedback_row(
                                self.stats['total_shots'], true_label,
                                pred, prob, video_path)
                        else:
                            print("     [skipped]")

                self.shot_data = []

            if headless:
                continue   # nothing to draw or show

            # ── Overlay rendering ─────────────────────────────────────────────
            annotated = results.plot()

            if self.possible_shot:
                cv2.rectangle(annotated, (10, 10), (680, 58), (0, 165, 255), -1)
                cv2.putText(annotated,
                            f"POSSIBLE SHOT ({self.poss_frame_count}/{sf.CONFIRM_FRAMES})",
                            (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)

            elif self.actual_shot:
                cv2.rectangle(annotated, (10, 10), (600, 58), (0, 200, 0), -1)
                cv2.putText(annotated,
                            f"TRACKING  ({len(self.shot_data)} frames)",
                            (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)

            elif self.last_pred is not None:
                col  = (0, 210, 0) if self.last_pred == 1 else (0, 0, 220)
                txt  = "MADE" if self.last_pred == 1 else "MISSED"
                conf = self._confidence(self.last_prob)
                cv2.rectangle(annotated, (10, 10), (500, 58), col, -1)
                cv2.putText(annotated,
                            f"{txt}  ({self.last_prob:.1%})  [{conf}]",
                            (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

            # Counter bar (bottom)
            t = self.stats['total_shots']
            m = self.stats['predicted_made']
            pct_label = f"{m/t*100:.0f}%" if t else "–"
            cv2.rectangle(annotated, (10, fh-42), (480, fh-6), (0,0,0), -1)
            cv2.putText(annotated,
                        f"Shots: {t}  |  Made: {m}  ({pct_label})",
                        (18, fh-16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

            if writer:
                writer.write(annotated)

            if display:
                scale   = min(self.DISPLAY_W / fw, self.DISPLAY_H / fh)
                resized = cv2.resize(annotated, (int(fw*scale), int(fh*scale)))
                cv2.imshow('Shot Prediction', resized)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    idx = self.frame_idx          # keep frame numbering intact
                    self._reset_shot_state()
                    self.frame_idx = idx

        # Cache detections only after a complete pass (not after pressing q)
        if new_cache is not None and reached_end and USE_DET_CACHE:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            boxes = np.concatenate(new_cache) if new_cache else np.zeros((0, 6))
            np.savez_compressed(cache_path, boxes=boxes.astype(np.float32))
            print(f"  Detections cached → {cache_path}")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        self.stats['end_time'] = datetime.now().isoformat()
        self._flush_feedback()
        self._save_outputs(out_folder, video_path)

        print(f"\n{'=' * 60}")
        print("DONE")
        print(f"  Shots detected : {self.stats['total_shots']}")
        print(f"  Made           : {self.stats['predicted_made']}")
        print(f"  Missed         : {self.stats['predicted_miss']}")
        print(f"  Output folder  : {out_folder}")
        print(f"{'=' * 60}")

        return out_folder, self.stats


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BASKETBALL SHOT PREDICTION – INFERENCE")
    print("=" * 60)

    print("\nChecking files …")
    ok = True
    needed = [("YOLO", YOLO_MODEL), ("Config", CONFIG_PATH), ("Video", VIDEO_PATH)]
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'rb') as f:
            if pickle.load(f).get('model_type', 'lstm') == 'lstm':
                needed += [("LSTM", LSTM_MODEL), ("Scaler", SCALER_PATH)]
    for label, path in needed:
        exists = os.path.exists(path)
        print(f"  {'✅' if exists else '❌'}  {label}: {path}")
        if not exists:
            ok = False

    if not ok:
        print("\nERROR: one or more required files missing – check paths above.")
        raise SystemExit(1)

    predictor = RealTimeShotPredictor(
        yolo_path    = YOLO_MODEL,
        lstm_path    = LSTM_MODEL,
        scaler_path  = SCALER_PATH,
        config_path  = CONFIG_PATH,
        output_dir   = OUTPUT_DIR,
        feedback_csv = FEEDBACK_CSV,
    )

    predictor.process_video(
        video_path  = VIDEO_PATH,
        display     = DISPLAY,
        save_video  = SAVE_VIDEO,
        verbose     = True,
    )