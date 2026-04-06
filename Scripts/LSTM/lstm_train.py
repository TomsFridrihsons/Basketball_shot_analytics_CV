"""
lstm_train.py  –  Basketball Shot Prediction: Unified Training Script
=====================================================================
Combines the best from both branches:
  - Main branch:  data augmentation (3x), full config export, evaluation metrics
  - Dev branch:   Bidirectional LSTM, attention mechanism, masking layer,
                  focal loss, L2 regularisation, RobustScaler

Output artefacts (all written to MODEL_SAVE_DIR):
  basketball_shot_lstm.keras   – saved model
  basketball_shot_lstm_scaler.pkl
  basketball_shot_lstm_config.pkl
  basketball_shot_lstm_config.json
  training_history.png
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler          # outlier-resistant (dev branch)
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score,
                             f1_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─── paths ───────────────────────────────────────────────────────────────────
DATA_PATH      = r"C:\Users\fridr\Documents\HooperAI\data\processed\annotations\basketball_shot_data_labeled_3103.csv"
MODEL_SAVE_DIR = r"C:\Users\fridr\Documents\HooperAI\data\model"
MODEL_NAME     = "basketball_shot_lstm"
# ─────────────────────────────────────────────────────────────────────────────


# ── Custom layers (serialisable – needed for .keras format) ──────────────────

@keras.utils.register_keras_serializable()
class SumAlongAxis(layers.Layer):
    """Replaces Lambda(lambda x: tf.reduce_sum(x, axis=1)) for safe serialisation."""
    def __init__(self, axis: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.reduce_sum(x, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


# ── Loss function ─────────────────────────────────────────────────────────────

def focal_loss(gamma: float = 2.0, alpha: float = 0.3):
    """
    Focal loss – downweights easy examples so the model focuses on hard ones.
    Crucial when made/missed shots are imbalanced (dev branch insight).
    """
    def _loss(y_true, y_pred):
        y_true   = tf.cast(y_true, tf.float32)
        y_pred   = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce      = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t      = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        weight   = tf.pow(1 - p_t, gamma)
        alpha_w  = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_w * weight * bce)
    return _loss


# ── Main class ────────────────────────────────────────────────────────────────

class ShotPredictionLSTM:

    FEATURE_COLS = [
        'ball_x', 'ball_y', 'ball_w', 'ball_h',
        'ball_velocity_x', 'ball_velocity_y',
        'player_x', 'player_y', 'player_w', 'player_h',
        'basket_x', 'basket_y', 'basket_w', 'basket_h',
        'ball_player_dist', 'ball_basket_dist', 'ball_basket_angle',
        'ball_above_player',
    ]

    def __init__(
        self,
        data_path:        str,
        model_save_dir:   str,
        model_name:       str  = "basketball_shot_lstm",
        # architecture
        lstm_units:       int   = 128,
        dropout_rate:     float = 0.4,
        l2_reg:           float = 0.001,
        use_bidirectional: bool = True,
        use_attention:    bool  = True,
        # focal loss
        focal_gamma:      float = 2.0,
        focal_alpha:      float = 0.3,
        # training
        test_size:        float = 0.2,
        random_state:     int   = 42,
        augment:          bool  = True,
        epochs:           int   = 200,
        batch_size:       int   = 8,
        patience:         int   = 30,
    ):
        self.data_path        = data_path
        self.model_save_dir   = model_save_dir
        self.model_name       = model_name

        self.lstm_units       = lstm_units
        self.dropout_rate     = dropout_rate
        self.l2_reg           = l2_reg
        self.use_bidirectional = use_bidirectional
        self.use_attention    = use_attention

        self.focal_gamma      = focal_gamma
        self.focal_alpha      = focal_alpha

        self.test_size        = test_size
        self.random_state     = random_state
        self.augment          = augment
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.patience         = patience

        # populated during pipeline
        self.scaler    = RobustScaler()   # outlier-resistant (dev branch)
        self.model     = None
        self.max_len   = None
        self.history   = None
        self.seq_stats = {}
        self.eval_metrics: dict = {}

        os.makedirs(model_save_dir, exist_ok=True)

    # ── 1. Data loading ───────────────────────────────────────────────────────

    def load_data(self):
        print("\n" + "=" * 60)
        print("STEP 1 – LOADING DATA")
        print("=" * 60)

        df = pd.read_csv(self.data_path)
        print(f"  Frames  : {len(df):,}")
        print(f"  Shots   : {df['shot_number'].nunique()}")

        shot_results = df.groupby('shot_number')['shot_result'].first()
        print(f"  Made    : {(shot_results == 1).sum()}")
        print(f"  Missed  : {(shot_results == 0).sum()}")

        X_raw, y, shot_ids = [], [], []
        for shot_num in df['shot_number'].unique():
            shot_df  = df[df['shot_number'] == shot_num]
            features = shot_df[self.FEATURE_COLS].fillna(0).values
            label    = int(shot_df['shot_result'].iloc[0])
            X_raw.append(features)
            y.append(label)
            shot_ids.append(shot_num)

        lengths = [len(s) for s in X_raw]
        self.seq_stats = {
            'min_len':      int(min(lengths)),
            'max_len':      int(max(lengths)),
            'mean_len':     float(np.mean(lengths)),
            'total_shots':  len(X_raw),
            'made_shots':   int(sum(y)),
            'missed_shots': int(len(y) - sum(y)),
        }
        print(f"\n  Seq length  min/mean/max : "
              f"{self.seq_stats['min_len']} / "
              f"{self.seq_stats['mean_len']:.1f} / "
              f"{self.seq_stats['max_len']}")

        self._X_raw = X_raw
        self._y     = np.array(y)
        return X_raw, self._y

    # ── 2. Padding ────────────────────────────────────────────────────────────

    def _pad(self, X_raw, max_len=None):
        """Zero-pad sequences to max_len (pad value = 0 matches Masking layer)."""
        if max_len is None:
            max_len = max(len(s) for s in X_raw)
        n_feat  = X_raw[0].shape[1]
        X_pad   = np.zeros((len(X_raw), max_len, n_feat), dtype=np.float32)
        for i, seq in enumerate(X_raw):
            L = min(len(seq), max_len)
            X_pad[i, :L, :] = seq[:L]
        return X_pad, max_len

    # ── 3. Augmentation (main branch) ────────────────────────────────────────

    def _augment(self, X_pad, y):
        """
        3× expansion per shot:
          - original
          - gaussian noise (σ = 0.02)
          - random time-shift (1–3 frames)
        """
        Xa, ya = [], []
        for seq, label in zip(X_pad, y):
            Xa.append(seq);  ya.append(label)
            noise = np.random.normal(0, 0.02, seq.shape).astype(np.float32)
            Xa.append(seq + noise); ya.append(label)
            shift = np.random.randint(1, 4)
            Xa.append(np.roll(seq, shift, axis=0)); ya.append(label)
        return np.array(Xa, dtype=np.float32), np.array(ya)

    # ── 4. Train / test split ─────────────────────────────────────────────────

    def prepare_split(self):
        print("\n" + "=" * 60)
        print("STEP 2 – PREPARING TRAIN / TEST SPLIT")
        print("=" * 60)

        X_pad, self.max_len = self._pad(self._X_raw)

        if self.augment:
            print("  Applying 3× augmentation (noise + time-shift) …")
            X_pad, y = self._augment(X_pad, self._y)
            print(f"  Augmented size : {len(X_pad)} sequences")
        else:
            y = self._y

        # Scale with RobustScaler – fits on flat array, reshapes back
        n, t, f = X_pad.shape
        X_flat  = X_pad.reshape(-1, f)
        X_flat  = self.scaler.fit_transform(X_flat)
        X_scaled = X_flat.reshape(n, t, f)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y,
            test_size    = self.test_size,
            random_state = self.random_state,
            stratify     = y,
        )

        print(f"  Train : {len(self.X_train)}   Test : {len(self.X_test)}")
        print(f"  Seq length : {self.max_len}   Features : {f}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    # ── 5. Model architecture (dev branch) ───────────────────────────────────

    def build_model(self):
        print("\n" + "=" * 60)
        print("STEP 3 – BUILDING MODEL")
        print("=" * 60)

        n_feat  = len(self.FEATURE_COLS)
        reg     = regularizers.l2(self.l2_reg)
        units   = self.lstm_units

        inputs = layers.Input(shape=(self.max_len, n_feat), name="sequence_input")

        # Masking: tells LSTM to ignore zero-padded timesteps (dev branch)
        x = layers.Masking(mask_value=0.0)(inputs)

        # ── LSTM stack ──────────────────────────────────────────────────────
        # Layer 1
        lstm1 = layers.LSTM(units, return_sequences=True,
                             kernel_regularizer=reg, name="lstm_1")
        x = (layers.Bidirectional(lstm1, name="bi_lstm_1")(x)
             if self.use_bidirectional else lstm1(x))
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Layer 2
        lstm2 = layers.LSTM(units // 2, return_sequences=True,
                             kernel_regularizer=reg, name="lstm_2")
        x = (layers.Bidirectional(lstm2, name="bi_lstm_2")(x)
             if self.use_bidirectional else lstm2(x))
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # ── Attention or final LSTM ──────────────────────────────────────────
        if self.use_attention:
            # Attention units = output dim of previous layer
            att_units = (units // 2) * (2 if self.use_bidirectional else 1)
            att  = layers.Dense(1, activation='tanh')(x)        # (B, T, 1)
            att  = layers.Flatten()(att)                         # (B, T)
            att  = layers.Activation('softmax')(att)             # (B, T)
            att  = layers.RepeatVector(att_units)(att)           # (B, att_units, T)
            att  = layers.Permute([2, 1])(att)                   # (B, T, att_units)
            x    = layers.Multiply()([x, att])
            x    = SumAlongAxis(axis=1)(x)                       # (B, att_units)
        else:
            lstm3 = layers.LSTM(units // 4, return_sequences=False,
                                 kernel_regularizer=reg, name="lstm_3")
            x = (layers.Bidirectional(lstm3, name="bi_lstm_3")(x)
                 if self.use_bidirectional else lstm3(x))
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # ── Dense head ───────────────────────────────────────────────────────
        x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)
        x = layers.Dropout(self.dropout_rate / 2)(x)

        outputs = layers.Dense(1, activation='sigmoid', name="shot_result")(x)

        self.model = keras.Model(inputs, outputs, name="ShotPredictionLSTM")
        self.model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=0.001),
            loss      = focal_loss(self.focal_gamma, self.focal_alpha),
            metrics   = [
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ],
        )

        dir_tag = "Bidirectional" if self.use_bidirectional else "Unidirectional"
        att_tag = "+ Attention"   if self.use_attention      else ""
        print(f"  Architecture : {dir_tag} LSTM {att_tag}")
        print(f"  Loss         : Focal (γ={self.focal_gamma}, α={self.focal_alpha})")
        print(f"  L2 reg       : {self.l2_reg}")
        self.model.summary()
        return self.model

    # ── 6. Training ──────────────────────────────────────────────────────────

    def train(self):
        print("\n" + "=" * 60)
        print("STEP 4 – TRAINING")
        print("=" * 60)

        ckpt_path = os.path.join(self.model_save_dir, f"{self.model_name}_best.keras")

        callbacks = [
            EarlyStopping(
                monitor              = 'val_loss',
                patience             = self.patience,
                restore_best_weights = True,
                verbose              = 1,
            ),
            ModelCheckpoint(
                filepath        = ckpt_path,
                monitor         = 'val_accuracy',
                save_best_only  = True,
                verbose         = 1,
            ),
        ]

        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_split = 0.2,
            epochs           = self.epochs,
            batch_size       = self.batch_size,
            callbacks        = callbacks,
            verbose          = 1,
        )

        h = self.history.history
        print(f"\n  Epochs trained      : {len(h['loss'])}")
        print(f"  Best val accuracy   : {max(h['val_accuracy']):.4f}")
        print(f"  Best val loss       : {min(h['val_loss']):.4f}")
        return self.history

    # ── 7. Evaluation (main branch) ──────────────────────────────────────────

    def evaluate(self):
        print("\n" + "=" * 60)
        print("STEP 5 – EVALUATION")
        print("=" * 60)

        proba = self.model.predict(self.X_test, verbose=0).flatten()
        pred  = (proba > 0.5).astype(int)

        acc = accuracy_score(self.y_test, pred)
        cm  = confusion_matrix(self.y_test, pred)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = f1_score(self.y_test, pred, zero_division=0)

        # Optimal threshold search
        thresholds   = np.linspace(0.1, 0.9, 81)
        best_thresh  = 0.5
        best_f1      = 0.0
        for t in thresholds:
            p = (proba > t).astype(int)
            s = f1_score(self.y_test, p, zero_division=0)
            if s > best_f1:
                best_f1, best_thresh = s, t

        print(f"\n  Test accuracy    : {acc:.4f}")
        print(f"  Precision        : {precision:.4f}")
        print(f"  Recall           : {recall:.4f}")
        print(f"  F1 score         : {f1:.4f}")
        print(f"  Optimal threshold: {best_thresh:.2f}  (F1 = {best_f1:.4f})")
        print("\n  Classification report:")
        print(classification_report(self.y_test, pred,
                                    target_names=['Missed', 'Made']))
        print(f"\n  Confusion matrix:\n    TN={tn}  FP={fp}\n    FN={fn}  TP={tp}")

        self.eval_metrics = {
            'test_accuracy':      float(acc),
            'precision':          float(precision),
            'recall':             float(recall),
            'f1_score':           float(f1),
            'optimal_threshold':  float(best_thresh),
            'optimal_f1':         float(best_f1),
            'true_negatives':     int(tn),
            'false_positives':    int(fp),
            'false_negatives':    int(fn),
            'true_positives':     int(tp),
        }
        return acc, pred, proba

    # ── 8. Plot (main branch) ────────────────────────────────────────────────

    def plot_history(self, save_path: str = None):
        if self.history is None:
            print("No history – train first.")
            return
        if save_path is None:
            save_path = os.path.join(self.model_save_dir, "training_history.png")

        h   = self.history.history
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training History", fontsize=16)

        pairs = [
            ('accuracy',  'val_accuracy',  'Accuracy'),
            ('loss',      'val_loss',       'Loss'),
            ('precision', 'val_precision',  'Precision'),
            ('recall',    'val_recall',     'Recall'),
        ]
        for ax, (tr, va, title) in zip(axes.flat, pairs):
            ax.plot(h[tr], label='Train')
            ax.plot(h[va], label='Validation')
            ax.set_title(title); ax.set_xlabel('Epoch')
            ax.legend(); ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n  Plot saved → {save_path}")
        plt.show()

    # ── 9. Save model + config (main branch structure, dev format) ────────────

    def save(self):
        print("\n" + "=" * 60)
        print("STEP 6 – SAVING")
        print("=" * 60)

        base        = os.path.join(self.model_save_dir, self.model_name)
        model_path  = base + ".keras"
        scaler_path = base + "_scaler.pkl"
        cfg_pkl     = base + "_config.pkl"
        cfg_json    = base + "_config.json"

        # Model
        self.model.save(model_path)
        print(f"  Model  → {model_path}")

        # Scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  Scaler → {scaler_path}")

        # Config (comprehensive – consumed by lstm_infer.py)
        h = self.history.history if self.history else {}
        config = {
            # inference essentials
            'max_len':            self.max_len,
            'feature_cols':       self.FEATURE_COLS,
            'n_features':         len(self.FEATURE_COLS),
            'optimal_threshold':  self.eval_metrics.get('optimal_threshold', 0.5),

            # architecture (so infer script can rebuild identically)
            'model_config': {
                'lstm_units':       self.lstm_units,
                'dropout_rate':     self.dropout_rate,
                'l2_reg':           self.l2_reg,
                'use_bidirectional': self.use_bidirectional,
                'use_attention':    self.use_attention,
                'focal_gamma':      self.focal_gamma,
                'focal_alpha':      self.focal_alpha,
                'learning_rate':    0.001,
                'optimizer':        'Adam',
                'loss':             'focal_loss',
            },

            # data
            'seq_stats': self.seq_stats,
            'training_config': {
                'test_size':      self.test_size,
                'random_state':   self.random_state,
                'augment':        self.augment,
                'n_train':        len(self.X_train),
                'n_test':         len(self.X_test),
                'epochs_run':     len(h.get('loss', [])),
                'batch_size':     self.batch_size,
                'patience':       self.patience,
                'best_val_acc':   float(max(h['val_accuracy'])) if h else None,
                'best_val_loss':  float(min(h['val_loss']))     if h else None,
            },

            # evaluation
            'eval_metrics': self.eval_metrics,

            # paths
            'model_path':  model_path,
            'scaler_path': scaler_path,
        }

        with open(cfg_pkl, 'wb') as f:
            pickle.dump(config, f)
        with open(cfg_json, 'w') as f:
            json.dump(config, f, indent=4)

        print(f"  Config → {cfg_json}")
        print(f"  Config → {cfg_pkl}")

        print("\n  ── Config summary ──────────────────────────────────")
        print(f"  max_len          : {config['max_len']}")
        print(f"  n_features       : {config['n_features']}")
        print(f"  optimal_threshold: {config['optimal_threshold']:.3f}")
        if self.eval_metrics:
            print(f"  test_accuracy    : {self.eval_metrics['test_accuracy']:.4f}")
            print(f"  f1_score         : {self.eval_metrics['f1_score']:.4f}")

        return config

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(self):
        self.load_data()
        self.prepare_split()
        self.build_model()
        self.train()
        self.evaluate()
        self.plot_history()
        return self.save()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BASKETBALL SHOT PREDICTION – TRAINING")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: data file not found → {DATA_PATH}")
        raise SystemExit(1)

    trainer = ShotPredictionLSTM(
        data_path        = DATA_PATH,
        model_save_dir   = MODEL_SAVE_DIR,
        model_name       = MODEL_NAME,
        # architecture
        lstm_units       = 128,
        dropout_rate     = 0.4,
        l2_reg           = 0.001,
        use_bidirectional = True,
        use_attention    = True,
        # focal loss
        focal_gamma      = 2.0,
        focal_alpha      = 0.3,
        # training
        test_size        = 0.2,
        random_state     = 42,
        augment          = True,
        epochs           = 200,
        batch_size       = 8,
        patience         = 30,
    )

    config = trainer.run()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model   : {config['model_path']}")
    print(f"  Scaler  : {config['scaler_path']}")
    print(f"  F1      : {config['eval_metrics']['f1_score']:.4f}")
    print(f"  Acc     : {config['eval_metrics']['test_accuracy']:.4f}")
    print(f"  Thresh  : {config['optimal_threshold']:.3f}")
    print("=" * 60)