"""
eval_confusion.py  –  Model vs Ground-Truth Evaluation
=======================================================
Runs lstm_infer.RealTimeShotPredictor on a video, then matches each
detected shot to a ground-truth shot in the labeled CSV using frame
overlap (±FRAME_TOL frames on start/end).

Three outcome categories
──────────────────────────────────────────────────────────────────────
  MATCHED      – model shot overlaps a GT shot
                   → compared label-vs-prediction → feeds confusion matrix
  GHOST        – model fired but no GT shot nearby
                   → "phantom" detection, real-world false alarm
  MISSED_GT    – GT shot exists but model never triggered
                   → model didn't detect the shot at all

The extended matrix is 3-row:
  rows  = ground truth  (GT MISS / GT MADE / Ghost = no GT)
  cols  = model output  (Pred MISS / Pred MADE)

Outputs
──────────────────────────────────────────────────────────────────────
  confusion_matrix.png   – standard 2×2 (matched shots only)
  extended_matrix.png    – 3-row plot including ghost shots
  match_log.csv          – every shot: matched + ghost + missed GT
  eval_summary.json      – all metrics
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
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score,
)

sys.path.insert(0, str(Path(__file__).parent))
import lstm_infer as infer

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
YOLO_MODEL  = r"C:\Users\fridr\Documents\HooperAI\data\model\yolo26m_best.pt"
LSTM_MODEL  = r"C:\Users\fridr\Documents\HooperAI\data\model\basketball_shot_lstm.keras"
SCALER_PATH = r"C:\Users\fridr\Documents\HooperAI\data\model\basketball_shot_lstm_scaler.pkl"
CONFIG_PATH = r"C:\Users\fridr\Documents\HooperAI\data\model\basketball_shot_lstm_config.pkl"
VIDEO_PATH  = r"C:\Users\fridr\Documents\HooperAI\data\raw\videos\20250711_171647.mp4"
GT_CSV      = r"C:\Users\fridr\Documents\HooperAI\data\processed\annotations\basketball_shot_data_labeled_3103.csv"
OUTPUT_DIR  = r"C:\Users\fridr\Documents\HooperAI\data\processed\eval"

FRAME_TOL   = 5      # ±frames allowed when matching shot boundaries
YOLO_CONF   = 0.75
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. Ground truth ───────────────────────────────────────────────────────────

def load_ground_truth(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    gt = (df.groupby("shot_number")
            .agg(
                start_frame=("frame_count", "min"),
                end_frame=("frame_count", "max"),
                shot_result=("shot_result", "first"),
            )
            .reset_index())
    gt["matched"] = False
    return gt


# ── 2. Inference ──────────────────────────────────────────────────────────────

def run_inference(video_path: str, output_dir: str,
                  yolo_conf: float = YOLO_CONF) -> list:
    os.makedirs(output_dir, exist_ok=True)
    infer.YOLO_CONF = yolo_conf   # patch module global – only reliable way

    predictor = infer.RealTimeShotPredictor(
        yolo_path   = YOLO_MODEL,
        lstm_path   = LSTM_MODEL,
        scaler_path = SCALER_PATH,
        config_path = CONFIG_PATH,
        output_dir  = output_dir,
    )
    _, stats = predictor.process_video(
        video_path = video_path,
        display    = False,
        save_video = False,
        verbose    = True,
    )
    return stats.get("shots", [])


# ── 3. Shot matching ──────────────────────────────────────────────────────────

def frames_overlap(m_start, m_end, gt_start, gt_end, tol: int) -> bool:
    return max(m_start - tol, gt_start) <= min(m_end + tol, gt_end)


def match_shots(model_shots: list, gt: pd.DataFrame, tol: int = FRAME_TOL):
    """
    Returns
    -------
    y_true, y_pred   – for matched shots (feeds 2x2 confusion matrix)
    matched_log      – list of dicts, one per matched pair
    ghost_shots      – model shots with no GT counterpart
    missed_gt        – GT shots the model never triggered on
    """
    gt = gt.copy()
    gt["matched"] = False

    y_true, y_pred = [], []
    matched_log    = []
    ghost_shots    = []

    for shot in model_shots:
        m_start = shot["start_frame"]
        m_end   = shot["end_frame"]
        m_pred  = shot["prediction"]

        candidates = gt[
            gt.apply(lambda r: frames_overlap(m_start, m_end,
                                              r.start_frame, r.end_frame, tol), axis=1)
            & ~gt["matched"]
        ]

        if candidates.empty:
            ghost_shots.append({
                "category":    "GHOST",
                "model_shot":  shot["shot_number"],
                "gt_shot":     None,
                "m_start":     m_start,
                "m_end":       m_end,
                "gt_start":    None,
                "gt_end":      None,
                "gt_label":    None,
                "pred_label":  int(m_pred),
                "probability": shot["probability"],
                "confidence":  shot["confidence"],
                "correct":     None,
                "note":        "Model fired – no GT shot within tolerance",
            })
            continue

        def boundary_dist(row):
            return abs(row.start_frame - m_start) + abs(row.end_frame - m_end)

        best_idx = candidates.apply(boundary_dist, axis=1).idxmin()
        best     = gt.loc[best_idx]
        gt.at[best_idx, "matched"] = True

        y_true.append(int(best.shot_result))
        y_pred.append(int(m_pred))
        matched_log.append({
            "category":    "MATCHED",
            "model_shot":  shot["shot_number"],
            "gt_shot":     int(best.shot_number),
            "m_start":     m_start,
            "m_end":       m_end,
            "gt_start":    int(best.start_frame),
            "gt_end":      int(best.end_frame),
            "gt_label":    int(best.shot_result),
            "pred_label":  int(m_pred),
            "probability": shot["probability"],
            "confidence":  shot["confidence"],
            "correct":     int(best.shot_result) == int(m_pred),
            "note":        "",
        })

    missed_gt = []
    for _, row in gt[~gt["matched"]].iterrows():
        missed_gt.append({
            "category":    "MISSED_GT",
            "model_shot":  None,
            "gt_shot":     int(row.shot_number),
            "m_start":     None,
            "m_end":       None,
            "gt_start":    int(row.start_frame),
            "gt_end":      int(row.end_frame),
            "gt_label":    int(row.shot_result),
            "pred_label":  None,
            "probability": None,
            "confidence":  None,
            "correct":     False,
            "note":        "GT shot – model never detected",
        })

    return y_true, y_pred, matched_log, ghost_shots, missed_gt


# ── 4. Plots ──────────────────────────────────────────────────────────────────

def plot_standard_cm(cm: np.ndarray, output_path: str,
                     n_matched: int, acc: float, f1: float):
    """Standard 2x2 confusion matrix for matched shots."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#181c27")

    cm_norm = cm.astype(float) / max(cm.sum(), 1)
    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)

    labels = ["Missed (0)", "Made (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color="#e8e8e8", fontsize=12)
    ax.set_yticklabels(labels, color="#e8e8e8", fontsize=12)
    ax.set_xlabel("Predicted", color="#9ca3af", fontsize=12, labelpad=10)
    ax.set_ylabel("Ground truth", color="#9ca3af", fontsize=12, labelpad=10)

    cell_tag = {(0,0):"TN", (0,1):"FP", (1,0):"FN", (1,1):"TP"}
    cell_col = {(0,0):"#22c55e", (0,1):"#ef4444",
                (1,0):"#ef4444", (1,1):"#22c55e"}
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)\n{cell_tag[(i,j)]}",
                    ha="center", va="center",
                    color=cell_col[(i,j)], fontsize=14, fontweight="bold")

    for sp in ax.spines.values():
        sp.set_edgecolor("#252a38")
    ax.tick_params(colors="#6b7280")

    fig.suptitle("Shot Prediction – Confusion Matrix (matched shots only)",
                 color="#f97316", fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(f"Matched: {n_matched}   Acc: {acc*100:.1f}%   F1: {f1:.3f}",
                 color="#9ca3af", fontsize=10, pad=10)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="#6b7280")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#6b7280")
    cb.set_label("Proportion", color="#6b7280", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Confusion matrix (2x2)  -> {output_path}")


def plot_extended_matrix(cm: np.ndarray,
                         ghost_miss: int, ghost_made: int,
                         missed_gt_miss: int, missed_gt_made: int,
                         output_path: str):
    """
    Extended 3-row matrix:
      Row 0  GT MISSED  – matched shots where GT label = 0
      Row 1  GT MADE    – matched shots where GT label = 1
      Row 2  GHOST      – model fired but no real shot existed

    Columns: Pred MISSED | Pred MADE
    """
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    data = np.array([
        [tn,         fp        ],
        [fn,         tp        ],
        [ghost_miss, ghost_made],
    ], dtype=float)

    row_labels = ["GT: Missed (0)", "GT: Made (1)", "Ghost\n(no GT shot)"]
    col_labels = ["Pred: Missed", "Pred: Made"]

    row_bg = [
        ["#1a3326", "#3d1c1c"],
        ["#3d1c1c", "#1a3326"],
        ["#2a2718", "#2a2718"],
    ]
    text_col = [
        ["#22c55e", "#ef4444"],
        ["#ef4444", "#22c55e"],
        ["#eab308", "#eab308"],
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.3, 3)
    ax.axis("off")

    cell_w, cell_h = 1.0, 1.0

    for r in range(3):
        for c in range(2):
            val  = int(data[r, c])
            rect = plt.Rectangle(
                (c * cell_w, (2 - r) * cell_h),
                cell_w, cell_h,
                linewidth=1.5, edgecolor="#252a38",
                facecolor=row_bg[r][c],
            )
            ax.add_patch(rect)
            ax.text(
                c * cell_w + cell_w / 2,
                (2 - r) * cell_h + cell_h / 2,
                str(val),
                ha="center", va="center",
                color=text_col[r][c],
                fontsize=24, fontweight="bold",
            )

    # Row labels (left side)
    for r, lbl in enumerate(row_labels):
        ax.text(-0.06, (2 - r) * cell_h + cell_h / 2, lbl,
                ha="right", va="center",
                color="#e8e8e8", fontsize=11)

    # Column labels (top)
    for c, lbl in enumerate(col_labels):
        ax.text(c * cell_w + cell_w / 2, 3.06, lbl,
                ha="center", va="bottom",
                color="#e8e8e8", fontsize=11, fontweight="bold")

    # Dashed divider between real and ghost rows
    ax.plot([0, 2], [1, 1], color="#f97316", linewidth=1.5,
            linestyle="--", alpha=0.7)

    # Annotations on right
    ax.text(2.05, 2.5, "real shots\n(matched to GT)",
            color="#9ca3af", fontsize=9, va="center")
    ax.text(2.05, 0.5, "ghost shots\n(no real shot existed)",
            color="#eab308", fontsize=9, va="center")

    # Missed GT note at bottom
    total_missed = missed_gt_miss + missed_gt_made
    if total_missed > 0:
        ax.text(1.0, -0.15,
                f"Real shots model never detected: {total_missed}  "
                f"({missed_gt_miss} missed, {missed_gt_made} made)",
                ha="center", va="top", color="#6b7280", fontsize=9)

    fig.suptitle("Extended Shot Matrix  (matched + ghost + undetected GT)",
                 color="#f97316", fontsize=13, fontweight="bold", y=0.97)

    plt.tight_layout(rect=[0.14, 0.06, 0.86, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Extended matrix (3-row) -> {output_path}")


# ── 5. Main evaluate ──────────────────────────────────────────────────────────

def evaluate(video_path: str, gt_csv: str, output_dir: str,
             tol: int = FRAME_TOL):
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Ground truth ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 – LOADING GROUND TRUTH")
    print("=" * 60)
    gt = load_ground_truth(gt_csv)
    print(f"  GT shots   : {len(gt)}")
    print(f"  GT made    : {(gt.shot_result==1).sum()}")
    print(f"  GT missed  : {(gt.shot_result==0).sum()}")
    print(f"  Frame range: {gt.start_frame.min()} – {gt.end_frame.max()}")

    # Video / GT sanity check
    import cv2 as _cv2
    _cap       = _cv2.VideoCapture(video_path)
    vid_frames = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    vid_fps    = _cap.get(_cv2.CAP_PROP_FPS)
    _cap.release()
    gt_max = int(gt.end_frame.max())

    print(f"\n  Video : {vid_frames} frames  "
          f"({vid_frames/max(vid_fps,1):.0f}s @ {vid_fps:.0f}fps)")
    if gt_max > vid_frames:
        in_range = gt[gt.start_frame <= vid_frames]
        print(f"  WARNING: GT extends to frame {gt_max} but video has {vid_frames}.")
        print(f"  Check that VIDEO_PATH matches this CSV.")
        print(f"  GT shots within video: {len(in_range)}/{len(gt)}")
    else:
        print(f"  OK: Video covers all GT shots")

    # ── Step 2: Inference ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 – RUNNING INFERENCE")
    print("=" * 60)
    model_shots = run_inference(video_path, output_dir, yolo_conf=YOLO_CONF)

    print(f"\n  Model detected : {len(model_shots)} shots")
    if model_shots:
        preds = [s["prediction"] for s in model_shots]
        print(f"  Pred made      : {sum(preds)}")
        print(f"  Pred missed    : {len(preds)-sum(preds)}")

    # Overfitting warning
    try:
        with open(CONFIG_PATH, "rb") as _f:
            _cfg = pickle.load(_f)
        _m = _cfg.get("eval_metrics", {})
        if _m.get("test_accuracy", 0) >= 0.99 or _m.get("f1_score", 0) >= 0.99:
            print(f"\n  WARNING: training metrics show "
                  f"{_m.get('test_accuracy',0)*100:.0f}% acc / "
                  f"F1={_m.get('f1_score',0):.2f} — likely overfit. "
                  f"Eval below reflects real-world performance.")
    except Exception:
        pass

    # ── Step 3: Match ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 3 – MATCHING  (tolerance +/-{tol} frames)")
    print("=" * 60)
    y_true, y_pred, matched_log, ghost_shots, missed_gt = \
        match_shots(model_shots, gt, tol)

    n_matched = len(y_true)
    n_ghost   = len(ghost_shots)
    n_miss_gt = len(missed_gt)

    print(f"\n  Matched pairs    : {n_matched}")
    print(f"  Ghost shots      : {n_ghost}  <- model fired, no real shot nearby")
    print(f"  Missed GT shots  : {n_miss_gt}  <- real shot, model never detected")

    if n_matched == 0:
        print("\nERROR: No shots matched. Check VIDEO_PATH vs CSV.")
        return

    # ── Step 4: Metrics ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 – RESULTS")
    print("=" * 60)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    ghost_miss     = sum(1 for g in ghost_shots if g["pred_label"] == 0)
    ghost_made     = sum(1 for g in ghost_shots if g["pred_label"] == 1)
    missed_gt_miss = sum(1 for r in missed_gt  if r["gt_label"]   == 0)
    missed_gt_made = sum(1 for r in missed_gt  if r["gt_label"]   == 1)

    # Console tables
    W = 54
    print(f"\n  {'MATCHED SHOTS':=^{W}}")
    print(f"  Total matched  : {n_matched}")
    print(f"  Accuracy       : {acc*100:.1f}%")
    print(f"  Precision      : {precision*100:.1f}%")
    print(f"  Recall         : {recall*100:.1f}%")
    print(f"  F1             : {f1:.3f}")
    print(f"\n  {'':20}  {'Pred MISS':>10}  {'Pred MADE':>10}")
    print(f"  {'GT MISSED':20}  {tn:>10}  (TN)  {fp:>5}  (FP)")
    print(f"  {'GT MADE':20}  {fn:>10}  (FN)  {tp:>5}  (TP)")

    print(f"\n  {'GHOST SHOTS (no real shot existed)':=^{W}}")
    print(f"  Total ghost    : {n_ghost}")
    print(f"  {'':20}  {'Pred MISS':>10}  {'Pred MADE':>10}")
    print(f"  {'No GT shot':20}  {ghost_miss:>10}        {ghost_made:>10}")

    print(f"\n  {'REAL SHOTS MODEL NEVER DETECTED':=^{W}}")
    print(f"  Total          : {n_miss_gt}")
    print(f"  GT label MISSED: {missed_gt_miss}")
    print(f"  GT label MADE  : {missed_gt_made}")

    print(f"\n  Classification report (matched shots only):")
    print(classification_report(y_true, y_pred,
                                target_names=["Missed", "Made"],
                                zero_division=0))

    # Ghost detail
    if ghost_shots:
        print(f"  Ghost shot detail ({n_ghost} total):")
        print(f"  {'Mdl#':>4}  {'Start':>7}  {'End':>7}  {'Pred':^6}  {'Prob':^6}  Conf")
        print(f"  {'----':>4}  {'-------':>7}  {'-------':>7}  {'------':^6}  {'------':^6}  ------")
        for g in sorted(ghost_shots, key=lambda x: x["m_start"]):
            pred_txt = "MADE" if g["pred_label"] == 1 else "MISS"
            print(f"  {g['model_shot']:>4}  {g['m_start']:>7}  {g['m_end']:>7}"
                  f"  {pred_txt:^6}  {g['probability']:.3f}   {g['confidence']}")

    # Missed GT detail
    if missed_gt:
        print(f"\n  Missed GT detail ({n_miss_gt} total):")
        print(f"  {'GT#':>4}  {'Start':>7}  {'End':>7}  {'Label':^6}")
        print(f"  {'----':>4}  {'-------':>7}  {'-------':>7}  {'------':^6}")
        for r in sorted(missed_gt, key=lambda x: x["gt_start"]):
            lbl = "MADE" if r["gt_label"] == 1 else "MISS"
            print(f"  {r['gt_shot']:>4}  {r['gt_start']:>7}  {r['gt_end']:>7}  {lbl:^6}")

    # ── Full match log CSV ────────────────────────────────────────────────────
    all_rows = matched_log + ghost_shots + missed_gt
    all_rows.sort(key=lambda r: (
        r["m_start"]  if r["m_start"]  is not None else
        r["gt_start"] if r["gt_start"] is not None else 0
    ))
    log_df  = pd.DataFrame(all_rows)
    log_csv = os.path.join(output_dir, "match_log.csv")
    log_df.to_csv(log_csv, index=False)
    print(f"\n  Full match log ({len(all_rows)} rows) -> {log_csv}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_standard_cm(
        cm,
        os.path.join(output_dir, "confusion_matrix.png"),
        n_matched, acc, f1,
    )
    plot_extended_matrix(
        cm, ghost_miss, ghost_made,
        missed_gt_miss, missed_gt_made,
        os.path.join(output_dir, "extended_matrix.png"),
    )

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "video":             video_path,
        "gt_csv":            gt_csv,
        "frame_tolerance":   tol,
        "gt_shots_total":    len(gt),
        "model_shots_total": len(model_shots),
        "matched_shots":     n_matched,
        "ghost_shots":       n_ghost,
        "ghost_pred_miss":   ghost_miss,
        "ghost_pred_made":   ghost_made,
        "missed_gt_total":   n_miss_gt,
        "missed_gt_miss":    missed_gt_miss,
        "missed_gt_made":    missed_gt_made,
        "accuracy":          round(float(acc), 4),
        "precision":         round(float(precision), 4),
        "recall":            round(float(recall), 4),
        "f1_score":          round(float(f1), 4),
        "confusion_matrix":  {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        },
    }
    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"  Summary JSON           -> {summary_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate lstm_infer against labeled ground-truth CSV"
    )
    parser.add_argument("--video",  default=VIDEO_PATH)
    parser.add_argument("--gt",     default=GT_CSV)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--tol",    default=FRAME_TOL, type=int,
                        help=f"Frame boundary tolerance (default {FRAME_TOL})")
    parser.add_argument("--yolo",   default=YOLO_MODEL)
    parser.add_argument("--lstm",   default=LSTM_MODEL)
    parser.add_argument("--scaler", default=SCALER_PATH)
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()

    infer.YOLO_MODEL  = args.yolo
    infer.LSTM_MODEL  = args.lstm
    infer.SCALER_PATH = args.scaler
    infer.CONFIG_PATH = args.config
    infer.YOLO_CONF   = YOLO_CONF

    evaluate(
        video_path = args.video,
        gt_csv     = args.gt,
        output_dir = args.output,
        tol        = args.tol,
    )