"""
lstm_train.py  –  Basketball Shot Prediction: Training + Honest Evaluation
=========================================================================
Fits the made/missed predictors on labeled shot CSVs (one CSV per video):
  - rim-pass rule (default in lstm_infer.py): made if the ball drops through
    rim height within `rim_threshold` basket widths of the rim centre.
    One parameter, ~93% on unseen videos.
  - LSTM v7 (residual on the rule): logit = a·rule_logit + b + correction, where
    the correction comes from a BiLSTM + attention over the per-frame features and
    the shot-level rim geometry (shot_features.shot_summary). The correction starts
    at zero and is L2-penalised, so with little data the model stays close to the
    rule and learns departures as labeled videos are added. 3 seeds averaged.

Evaluation and data handling (fixes over v5):
  - No leakage: shots are split into train / val / test FIRST; augmentation is
    applied to the training shots only.
  - Thresholds are fit on training data only, never on the test set.
  - Cross-validation:
      video – leave-one-video-out (each CSV is one video): how well the model
              works on footage it has never seen. This is the number that matters.
      shot  – stratified 5-fold over shots (same videos in train and test).
  - Features v2 (scripts/shot_features.py): basket-relative, scale-free, gap
    interpolation, smoothed velocity; same code is used by lstm_infer.py.
  - Sequences keep the LAST 64 frames (the end of the shot decides the outcome)
    instead of the first 260; scaler is fit on real frames only, then padded.
  - Small model for ~200 shots; augmentation: noise, time shift, left↔right mirror.

Usage:
    python scripts/lstm/lstm_train.py                 # CV (video + shot) then train + save
    python scripts/lstm/lstm_train.py --cv video      # only leave-one-video-out
    python scripts/lstm/lstm_train.py --cv video --no-save
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# ─── paths ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ANNOTATIONS_DIR, MODEL_DIR  # noqa: E402
import shot_features as sf  # noqa: E402

DATA_PATHS = [
    str(ANNOTATIONS_DIR / "basketball_shot_data_labeled_04_05.csv"),
    str(ANNOTATIONS_DIR / "basketball_shot_data_labeled_04_05_02.csv"),
    str(ANNOTATIONS_DIR / "basketball_shot_data_labeled_04_05_03.csv"),
    str(ANNOTATIONS_DIR / "basketball_shot_data_labeled_3103.csv"),
]
MODEL_SAVE_DIR = str(MODEL_DIR)
MODEL_NAME     = "basketball_shot_lstm_v7"
# ─────────────────────────────────────────────────────────────────────────────

SEED       = 42
N_SEEDS    = 3      # ensemble members (averaged)
LSTM_UNITS = 24
DROPOUT    = 0.4
LR         = 1e-3
EPOCHS     = 200
PATIENCE   = 20
BATCH_SIZE = 32
VAL_FRAC   = 0.15
CORR_L2    = 1e-2   # penalty on the LSTM correction's output weights


# ── Custom layer (must match lstm_infer.py) ──────────────────────────────────

@keras.utils.register_keras_serializable(package="Custom")
class MaskedAttention(layers.Layer):
    """Scalar score per timestep, padded frames get -inf before softmax, weighted sum."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1, activation='tanh')

    def call(self, x, mask=None):
        scores = tf.squeeze(self.score_dense(x), axis=-1)          # (B, T)
        if mask is not None:
            mask_float = tf.cast(mask, tf.float32)
            scores = scores * mask_float + (1.0 - mask_float) * (-1e9)
        weights = tf.expand_dims(tf.nn.softmax(scores, axis=-1), -1)  # (B, T, 1)
        return tf.reduce_sum(x * weights, axis=1)                     # (B, F)

    def compute_mask(self, x, mask=None):
        return None   # time axis is reduced away; nothing to propagate

    def get_config(self):
        return super().get_config()


# ── Data ─────────────────────────────────────────────────────────────────────

def load_shots(paths):
    """One dict per shot: raw (T, 18) frames, label, group (= source video/CSV)."""
    shots = []
    for path in paths:
        head = open(path).read(500)
        df = pd.read_csv(path, sep=';' if head.count(';') > head.count(',') else ',')
        stem = Path(path).stem
        for _, shot_df in df.groupby('shot_number', sort=False):
            # app_labeled.csv (scripts/app) mixes videos; group those by video
            group = (f"{stem}:{shot_df['source_video'].iloc[0]}"
                     if 'source_video' in shot_df else stem)
            shots.append({
                'raw':   shot_df[sf.RAW_COLS].to_numpy(dtype=np.float64),
                'label': int(shot_df['shot_result'].iloc[0]),
                'group': group,
            })
        n = sum(s['group'].split(':')[0] == stem for s in shots)
        print(f"  {stem}: {n} shots")
    labels = np.array([s['label'] for s in shots])
    print(f"  Total: {len(shots)} shots  (made {labels.sum()}, missed {len(labels) - labels.sum()})")
    return shots


def featurize(shots):
    return [sf.shot_to_features(s['raw']) for s in shots]


def augment(seqs, labels, rng, mirror: bool = True):
    out_x, out_y = [], []
    for seq, y in zip(seqs, labels):
        variants = [seq, seq + rng.normal(0, 0.02, seq.shape).astype(np.float32)]
        shift = rng.integers(1, 4)
        if len(seq) - shift >= 8:
            variants.append(seq[shift:])
        if mirror:
            variants.append(sf.mirror(seq))
        out_x += variants
        out_y += [y] * len(variants)
    return out_x, np.array(out_y)


# ── Model (v7: sequence branch + shot-geometry branch) ───────────────────────

RULE_TEMP = 0.1   # rule logit = (rim_threshold - rim_offset) / RULE_TEMP, clipped


def build_member(max_len: int, n_feat: int, n_sum: int, seed: int) -> keras.Model:
    """Residual model: logit = a * rule_logit + b + correction(sequence, summary).

    The correction's output weights start at zero and carry an L2 penalty, so with
    little data the model behaves like the rim-pass rule and only departs from it
    where the frame sequence gives consistent evidence.
    """
    init = keras.initializers.GlorotUniform(seed=seed)
    seq_in  = layers.Input(shape=(max_len, n_feat), name="sequence_input")
    sum_in  = layers.Input(shape=(n_sum,), name="summary_input")
    rule_in = layers.Input(shape=(1,), name="rule_input")

    x = layers.Masking(mask_value=0.0)(seq_in)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True,
                                         kernel_initializer=init))(x)
    x = MaskedAttention()(x)
    x = layers.Dropout(DROPOUT, seed=seed)(x)
    x = layers.Dense(8, activation='relu', kernel_initializer=init)(x)
    s = layers.Dense(8, activation='relu', kernel_initializer=init)(sum_in)
    h = layers.Concatenate()([x, s])
    correction = layers.Dense(1, kernel_initializer='zeros',
                              kernel_regularizer=keras.regularizers.l2(CORR_L2),
                              name="correction")(h)

    base = layers.Dense(1, kernel_initializer='ones', bias_initializer='zeros',
                        name="rule_calibration")(rule_in)
    out = layers.Activation('sigmoid', name="prob")(layers.Add()([base, correction]))
    model = keras.Model([seq_in, sum_in, rule_in], out)
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def ensemble(members) -> keras.Model:
    """Average the members' outputs in one model (one file for lstm_infer.py)."""
    ins = [layers.Input(shape=i.shape[1:], name=n)
           for i, n in zip(members[0].inputs, ("sequence_input", "summary_input", "rule_input"))]
    outs = [m(ins) for m in members]
    out = layers.Average(name="shot_result")(outs) if len(outs) > 1 else outs[0]
    return keras.Model(ins, out, name="ShotPredictionLSTM_v7")


def rule_logit(seqs, rim_threshold):
    off = np.array([sf.rim_pass_offset(s) for s in seqs], dtype=np.float32)
    return np.clip((rim_threshold - off) / RULE_TEMP, -6, 6)[:, None]


def _inputs(scalers, seqs, max_len):
    X_seq = sf.pad_tail([scalers['seq'].transform(s).astype(np.float32) for s in seqs], max_len)
    X_sum = scalers['summary'].transform(np.stack([sf.shot_summary(s) for s in seqs])).astype(np.float32)
    return [X_seq, X_sum, rule_logit(seqs, scalers['rim_threshold'])]


def fit(train_seqs, y_train, val_seqs, y_val, max_len=sf.MAX_LEN, verbose=0):
    """Fit the rule threshold on the training shots, augment train only, fit
    scalers on training data, train N_SEEDS residual members (early stopping on
    val) and average them; decision threshold 0.5."""
    rim_t = _best_rim_threshold(np.array([sf.rim_pass_offset(s) for s in train_seqs]),
                                np.asarray(y_train))
    rng = np.random.default_rng(SEED)
    aug_x, aug_y = augment(train_seqs, y_train, rng)

    scalers = {
        'seq':           StandardScaler().fit(np.concatenate(aug_x)),
        'summary':       StandardScaler().fit(np.stack([sf.shot_summary(s) for s in aug_x])),
        'rim_threshold': rim_t,
    }
    X_tr, X_val = _inputs(scalers, aug_x, max_len), _inputs(scalers, val_seqs, max_len)

    classes = np.unique(aug_y)
    class_weight = dict(zip(classes, compute_class_weight('balanced', classes=classes, y=aug_y)))

    members, histories = [], []
    for k in range(N_SEEDS):
        keras.utils.set_random_seed(SEED + k)
        m = build_member(max_len, X_tr[0].shape[2], X_tr[1].shape[1], SEED + k)
        histories.append(m.fit(
            X_tr, aug_y, validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight, verbose=verbose,
            callbacks=[EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                     restore_best_weights=True)],
        ))
        members.append(m)

    # The output is a calibrated logistic around the rule; tuning the cut-off on a
    # ~25-shot val set (by F1) only pushed it toward "made" and added false positives.
    return ensemble(members), scalers, 0.5, histories


def predict(model, scalers, seqs, max_len=sf.MAX_LEN):
    return model.predict(_inputs(scalers, seqs, max_len), verbose=0).ravel()


# ── Metrics ──────────────────────────────────────────────────────────────────

def metrics(y, prob, threshold):
    pred = (prob > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        'n': int(len(y)),
        'accuracy':  float((tp + tn) / len(y)),
        'precision': float(prec),
        'recall':    float(rec),
        'f1':        float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def _fmt(m):
    return (f"acc {m['accuracy']:.3f}  f1 {m['f1']:.3f}  "
            f"prec {m['precision']:.3f}  rec {m['recall']:.3f}  "
            f"(TN {m['tn']} FP {m['fp']} FN {m['fn']} TP {m['tp']}, n={m['n']})")


# ── Evaluation protocols ─────────────────────────────────────────────────────

def _val_split(idx, labels):
    tr, val = train_test_split(idx, test_size=VAL_FRAC, random_state=SEED, stratify=labels[idx])
    return tr, val


def _folds(shots, mode):
    labels = np.array([s['label'] for s in shots])
    groups = np.array([s['group'] for s in shots])
    idx = np.arange(len(shots))
    if mode == 'video':
        return [(idx[groups != g], idx[groups == g], g) for g in dict.fromkeys(groups)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return [(tr, te, f"fold {k + 1}") for k, (tr, te) in enumerate(skf.split(idx, labels))]


def _rim_offsets(shots):
    return np.array([sf.rim_pass_offset(sf.shot_to_features(s['raw'])) for s in shots])


def _best_rim_threshold(offset, labels):
    grid = np.linspace(0.1, 1.5, 29)
    return float(max(grid, key=lambda g: ((offset < g) == labels).mean()))


def save_rim_rule(shots, cv_results: dict):
    """Fit the rule's single threshold on all shots and save a config lstm_infer.py can load."""
    offset, labels = _rim_offsets(shots), np.array([s['label'] for s in shots])
    t = _best_rim_threshold(offset, labels)
    config = {
        'model_type':        'rim_rule',
        'feature_version':   sf.FEATURE_VERSION,
        'rim_threshold':     t,
        'optimal_threshold': 0.5,     # lstm_infer maps offset → score with 0.5 at rim_threshold
        'max_len':           sf.MAX_LEN,
        'feature_cols':      sf.FEATURE_COLS,
        'train_metrics':     metrics(labels, (offset < t).astype(float), 0.5),
        'cv_metrics':        {k: v for k, v in cv_results.items() if 'rim_rule' in k},
        'data_paths':        [Path(p).name for p in DATA_PATHS],
    }
    base = os.path.join(MODEL_SAVE_DIR, "basketball_shot_rim_rule_config")
    with open(base + ".pkl", 'wb') as f:
        pickle.dump(config, f)
    with open(base + ".json", 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\n── Rim-pass rule: threshold {t:.2f} basket widths → {base}.pkl")
    return config


def rim_rule_cv(shots, mode: str):
    """Baseline without learning: made if the ball drops through rim height within
    `t` basket widths of the rim centre; `t` is picked on the training fold."""
    offset = _rim_offsets(shots)
    labels = np.array([s['label'] for s in shots])

    print(f"\n── CV: {mode} (rim-pass rule baseline) " + "─" * 20)
    all_y, all_pred = [], []
    for train_idx, test_idx, name in _folds(shots, mode):
        t = _best_rim_threshold(offset[train_idx], labels[train_idx])
        pred = (offset[test_idx] < t).astype(float)
        print(f"  {name:<45} thr {t:.2f}  {'':>11}{_fmt(metrics(labels[test_idx], pred, 0.5))}")
        all_y.append(labels[test_idx])
        all_pred.append(pred)
    pooled = metrics(np.concatenate(all_y), np.concatenate(all_pred), 0.5)
    print(f"  {'POOLED':<45} {'':>23}{_fmt(pooled)}")
    return pooled


def cross_validate(shots, mode: str):
    """mode='video': leave-one-video-out; mode='shot': stratified 5-fold over shots.

    Returns pooled metrics over all held-out predictions.
    """
    seqs = featurize(shots)
    labels = np.array([s['label'] for s in shots])
    folds = _folds(shots, mode)

    print(f"\n── CV: {mode} (LSTM v7 residual, {N_SEEDS}-seed ensemble) " + "─" * 12)
    all_y, all_prob, all_thr = [], [], []
    for train_idx, test_idx, name in folds:
        tr, val = _val_split(train_idx, labels)
        model, scalers, thr, hists = fit([seqs[i] for i in tr], labels[tr],
                                         [seqs[i] for i in val], labels[val])
        prob = predict(model, scalers, [seqs[i] for i in test_idx])
        m = metrics(labels[test_idx], prob, thr)
        epochs = max(len(h.history['loss']) for h in hists)
        print(f"  {name:<45} thr {thr:.2f}  epochs {epochs:>3}  {_fmt(m)}")
        all_y.append(labels[test_idx])
        all_prob.append(prob)
        all_thr.append(np.full(len(test_idx), thr))
        keras.backend.clear_session()

    y, prob, thr = map(np.concatenate, (all_y, all_prob, all_thr))
    pooled = metrics(y, (prob > thr).astype(float), 0.5)   # per-fold thresholds applied
    print(f"  {'POOLED':<45} {'':>23}{_fmt(pooled)}")
    return pooled


# ── Final model ──────────────────────────────────────────────────────────────

def train_final(shots, cv_results: dict, save: bool = True):
    seqs = featurize(shots)
    labels = np.array([s['label'] for s in shots])
    tr, val = _val_split(np.arange(len(shots)), labels)

    print(f"\n── Final model: train {len(tr)} shots, val {len(val)} shots " + "─" * 15)
    model, scalers, thr, histories = fit([seqs[i] for i in tr], labels[tr],
                                         [seqs[i] for i in val], labels[val], verbose=0)
    val_m = metrics(labels[val], predict(model, scalers, [seqs[i] for i in val]), thr)
    history = histories[0]
    print(f"  epochs {[len(h.history['loss']) for h in histories]}  threshold {thr:.2f}")
    print(f"  val: {_fmt(val_m)}")
    model.summary()

    if not save:
        return None

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    base = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    model.save(base + "_best.keras")
    with open(base + "_scaler.pkl", 'wb') as f:
        pickle.dump(scalers, f)      # {'seq', 'summary': StandardScaler, 'rim_threshold': float}

    lengths = [len(s) for s in seqs]
    config = {
        'model_type':        'lstm_hybrid',
        'feature_version':   sf.FEATURE_VERSION,
        'max_len':           sf.MAX_LEN,
        'feature_cols':      sf.FEATURE_COLS,
        'raw_cols':          sf.RAW_COLS,
        'n_features':        len(sf.FEATURE_COLS),
        'summary_cols':      sf.SUMMARY_COLS,
        'rule_temp':         RULE_TEMP,
        'optimal_threshold': thr,
        'model_config': {'lstm_units': LSTM_UNITS, 'dropout_rate': DROPOUT, 'n_seeds': N_SEEDS,
                         'learning_rate': LR, 'loss': 'binary_crossentropy'},
        'seq_stats': {'min_len': int(min(lengths)), 'max_len': int(max(lengths)),
                      'mean_len': float(np.mean(lengths)), 'total_shots': len(seqs),
                      'made_shots': int(labels.sum()), 'missed_shots': int(len(labels) - labels.sum())},
        'training_config': {'data_paths': [Path(p).name for p in DATA_PATHS],
                            'n_train': len(tr), 'n_val': len(val),
                            'epochs_run': len(history.history['loss']), 'batch_size': BATCH_SIZE},
        'val_metrics': val_m,
        'cv_metrics':  cv_results,
    }
    with open(base + "_config.pkl", 'wb') as f:
        pickle.dump(config, f)
    with open(base + "_config.json", 'w') as f:
        json.dump(config, f, indent=4)

    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ('loss', 'accuracy')):
        ax.plot(h[key], label='train')
        ax.plot(h[f'val_{key}'], label='val')
        ax.set_title(key)
        ax.set_xlabel('epoch')
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(base + "_history.png", dpi=150)

    print(f"\n  Saved → {base}_best.keras / _scaler.pkl / _config.pkl / _config.json / _history.png")
    return config


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit the rim-pass rule and train the shot-outcome LSTM")
    ap.add_argument("--cv", choices=["both", "video", "shot", "none"], default="both")
    ap.add_argument("--no-save", action="store_true", help="skip training the final model")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    for p in DATA_PATHS:
        if not os.path.exists(p):
            raise SystemExit(f"ERROR: data file not found → {p}")

    print("=" * 60)
    print("BASKETBALL SHOT PREDICTION – TRAINING v7")
    print("=" * 60)
    shots = load_shots(DATA_PATHS)

    modes = {"both": ["video", "shot"], "none": []}.get(args.cv, [args.cv])
    cv_results = {}
    for mode in modes:
        cv_results[f"{mode}_rim_rule"] = rim_rule_cv(shots, mode)
        cv_results[mode] = cross_validate(shots, mode)

    if not args.no_save:
        save_rim_rule(shots, cv_results)
        train_final(shots, cv_results)
