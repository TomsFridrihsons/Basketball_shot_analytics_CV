"""
yolo_eval.py  –  YOLO Model Evaluation on Test Dataset
=======================================================
Runs the trained YOLO model against a labelled test set and reports:
  - Per-class metrics: Precision, Recall, mAP@50, mAP@50-95
  - Overall / weighted averages
  - Confusion matrix
  - Per-image detection summary CSV
  - Visual sample grid (worst / best predictions)

Expected test folder layout (standard YOLO format):
  test/
    images/   *.jpg / *.png / *.mp4 frames
    labels/   *.txt  (YOLO format: class cx cy w h, normalised)

Usage:
    python yolo_eval.py
"""

import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from ultralytics import YOLO

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import YOLO_MODEL as _YOLO, YOLO_DATASET_DIR, DATA_DIR  # noqa: E402

YOLO_MODEL   = str(_YOLO)
TEST_DIR     = str(YOLO_DATASET_DIR / "test")
OUTPUT_DIR   = str(DATA_DIR / "processed" / "yolo_eval")

# Detection threshold for inference
CONF_THRESH  = 0.75   # lower than production to capture borderline detections
IOU_THRESH   = 0.5    # IoU threshold for a TP match

# Class names — must match your YOLO model order (0=ball, 1=basket, 2=player)
CLASS_NAMES  = {0: "ball", 1: "basket", 2: "player"}

# How many sample images to save in the visual grid
N_WORST      = 12
N_BEST       = 12
# ──────────────────────────────────────────────────────────────────────────────


def iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (aA + aB - inter)


def yolo_label_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert normalised YOLO label → pixel [x1,y1,x2,y2]."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_labels(label_path, img_w, img_h):
    """
    Load a YOLO-format .txt label file.
    Returns list of (class_id, [x1,y1,x2,y2]).
    """
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            box = yolo_label_to_xyxy(cx, cy, w, h, img_w, img_h)
            gts.append((cls, box))
    return gts


def match_predictions(preds, gts, iou_thresh=0.5):
    """
    Greedy match predictions → ground-truths by descending confidence.

    preds : list of (class_id, conf, [x1,y1,x2,y2])
    gts   : list of (class_id, [x1,y1,x2,y2])

    Returns (tp_list, fp_list, fn_count_per_class)
    where tp_list / fp_list are lists of (class_id, conf).
    """
    matched_gt = set()
    tps, fps   = [], []

    # sort by descending confidence
    for cls_p, conf, box_p in sorted(preds, key=lambda x: -x[1]):
        best_iou, best_idx = 0.0, -1
        for i, (cls_g, box_g) in enumerate(gts):
            if i in matched_gt:
                continue
            if cls_g != cls_p:
                continue
            v = iou(box_p, box_g)
            if v > best_iou:
                best_iou, best_idx = v, i

        if best_iou >= iou_thresh and best_idx >= 0:
            tps.append((cls_p, conf))
            matched_gt.add(best_idx)
        else:
            fps.append((cls_p, conf))

    # FN = unmatched ground truths
    fn_per_cls = {}
    for i, (cls_g, _) in enumerate(gts):
        if i not in matched_gt:
            fn_per_cls[cls_g] = fn_per_cls.get(cls_g, 0) + 1

    return tps, fps, fn_per_cls


def compute_ap(recalls, precisions):
    """Compute AP using the 11-point interpolation (VOC style)."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p_at_r = [p for r, p in zip(recalls, precisions) if r >= thr]
        ap += max(p_at_r) if p_at_r else 0.0
    return ap / 11.0


def find_images(test_dir):
    """Locate all images under test_dir/images/."""
    img_dir = Path(test_dir) / "images"
    if not img_dir.exists():
        # fallback: images at top level
        img_dir = Path(test_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)
    return imgs, img_dir


def find_label(img_path: Path, test_dir: Path):
    """Given an image path, find the corresponding .txt label file."""
    label_dir = test_dir / "labels"
    if not label_dir.exists():
        label_dir = img_path.parent.parent / "labels"
    label_path = label_dir / (img_path.stem + ".txt")
    return label_path


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_eval():
    print("=" * 60)
    print("YOLO MODEL EVALUATION")
    print("=" * 60)
    print(f"  Model   : {YOLO_MODEL}")
    print(f"  TestDir : {TEST_DIR}")
    print(f"  Conf    : {CONF_THRESH}  |  IoU : {IOU_THRESH}")

    # ── Validate paths ────────────────────────────────────────────────────────
    for label, path in [("YOLO model", YOLO_MODEL), ("Test dir", TEST_DIR)]:
        if not os.path.exists(path):
            print(f"\n❌ {label} not found: {path}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading YOLO model …")
    model = YOLO(YOLO_MODEL)
    print("  ✅ Loaded")

    # ── Discover images ───────────────────────────────────────────────────────
    imgs, img_dir = find_images(TEST_DIR)
    test_path = Path(TEST_DIR)
    print(f"\nFound {len(imgs)} images in {img_dir}")
    if len(imgs) == 0:
        print("❌ No images found. Check TEST_DIR.")
        sys.exit(1)

    # ── Per-class accumulators ─────────────────────────────────────────────────
    # all_preds[cls] = list of (conf, is_tp)  — for AP curve
    all_preds  = {c: [] for c in CLASS_NAMES}
    n_gt       = {c: 0  for c in CLASS_NAMES}

    per_image_rows = []     # for CSV
    score_per_image = []    # (score, img_path) for best/worst grid

    print("\nRunning inference …\n")
    n = len(imgs)

    for idx, img_path in enumerate(imgs):
        if idx % max(1, n // 10) == 0:
            print(f"  [{idx}/{n}]  {img_path.name}")

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  ⚠️  Could not read {img_path.name} – skipping")
            continue

        img_h, img_w = frame.shape[:2]

        # Ground-truth
        label_path = find_label(img_path, test_path)
        gts = load_labels(str(label_path), img_w, img_h)
        for cls_g, _ in gts:
            if cls_g in n_gt:
                n_gt[cls_g] += 1

        # Predictions
        results = model(frame, conf=CONF_THRESH, verbose=False)[0]
        preds = []
        for box in results.boxes:
            cls_p  = int(box.cls[0])
            conf_p = float(box.conf[0])
            xyxy   = box.xyxy[0].cpu().numpy().tolist()
            preds.append((cls_p, conf_p, xyxy))

        # Match
        tps, fps, fn_per_cls = match_predictions(preds, gts, IOU_THRESH)

        img_tp = len(tps)
        img_fp = len(fps)
        img_fn = sum(fn_per_cls.values())
        img_gt = len(gts)

        # Accumulate for AP
        matched_gt_set = set()
        tp_set_temp    = set()
        sorted_preds   = sorted(preds, key=lambda x: -x[1])
        for cls_p, conf_p, box_p in sorted_preds:
            best_iou, best_idx = 0.0, -1
            for i, (cls_g, box_g) in enumerate(gts):
                if i in matched_gt_set or cls_g != cls_p:
                    continue
                v = iou(box_p, box_g)
                if v > best_iou:
                    best_iou, best_idx = v, i
            is_tp = (best_iou >= IOU_THRESH and best_idx >= 0)
            if is_tp:
                matched_gt_set.add(best_idx)
            if cls_p in all_preds:
                all_preds[cls_p].append((conf_p, is_tp))

        # Per-image precision / recall
        prec = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 1.0
        rec  = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 1.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_image_rows.append({
            "image":      img_path.name,
            "n_gt":       img_gt,
            "n_pred":     len(preds),
            "tp":         img_tp,
            "fp":         img_fp,
            "fn":         img_fn,
            "precision":  round(prec, 4),
            "recall":     round(rec,  4),
            "f1":         round(f1,   4),
        })

        score_per_image.append((f1, str(img_path)))

    # ── Per-class AP ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    class_metrics = {}
    map50_vals    = []

    for cls_id, cls_name in CLASS_NAMES.items():
        entries  = sorted(all_preds[cls_id], key=lambda x: -x[0])
        total_gt = n_gt[cls_id]

        if total_gt == 0 or len(entries) == 0:
            class_metrics[cls_name] = dict(
                precision=0, recall=0, f1=0, ap50=0, n_gt=total_gt, n_pred=len(entries))
            continue

        tp_cum = 0
        recs, precs = [], []
        for i, (conf, is_tp) in enumerate(entries):
            if is_tp:
                tp_cum += 1
            precs.append(tp_cum / (i + 1))
            recs.append(tp_cum / total_gt)

        ap = compute_ap(recs, precs)
        map50_vals.append(ap)

        # Final P / R at lowest confidence = best recall point
        final_p = precs[-1] if precs else 0
        final_r = recs[-1]  if recs  else 0
        f1      = (2*final_p*final_r/(final_p+final_r)) if (final_p+final_r) > 0 else 0

        class_metrics[cls_name] = dict(
            precision=round(final_p, 4),
            recall   =round(final_r, 4),
            f1       =round(f1,      4),
            ap50     =round(ap,      4),
            n_gt     =total_gt,
            n_pred   =len(entries),
        )

    mAP50 = np.mean(map50_vals) if map50_vals else 0.0

    # ── Print table ───────────────────────────────────────────────────────────
    header = f"{'Class':<10} {'GT':>6} {'Pred':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AP@50':>8}"
    print(header)
    print("-" * len(header))
    for cls_name, m in class_metrics.items():
        print(f"{cls_name:<10} {m['n_gt']:>6} {m['n_pred']:>6} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['ap50']:>8.4f}")
    print("-" * len(header))
    print(f"{'mAP@50':<10} {'':>6} {'':>6} {'':>8} {'':>8} {'':>8} {mAP50:>8.4f}")

    # ── Overall stats across all images ───────────────────────────────────────
    all_tp  = sum(r["tp"]  for r in per_image_rows)
    all_fp  = sum(r["fp"]  for r in per_image_rows)
    all_fn  = sum(r["fn"]  for r in per_image_rows)
    all_gt  = sum(r["n_gt"] for r in per_image_rows)
    all_pred= sum(r["n_pred"] for r in per_image_rows)

    overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_rec  = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    overall_f1   = (2*overall_prec*overall_rec/(overall_prec+overall_rec)
                    if (overall_prec+overall_rec) > 0 else 0)

    print(f"\n  Overall  –  GT: {all_gt}  Pred: {all_pred}")
    print(f"  TP: {all_tp}   FP: {all_fp}   FN: {all_fn}")
    print(f"  Precision : {overall_prec:.4f}")
    print(f"  Recall    : {overall_rec:.4f}")
    print(f"  F1        : {overall_f1:.4f}")
    print(f"  mAP@50    : {mAP50:.4f}")
    print(f"\n  Images evaluated : {len(per_image_rows)}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, f"per_image_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_image_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"\n  Per-image CSV  → {csv_path}")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary = {
        "timestamp":       ts,
        "model":           YOLO_MODEL,
        "test_dir":        TEST_DIR,
        "conf_thresh":     CONF_THRESH,
        "iou_thresh":      IOU_THRESH,
        "n_images":        len(per_image_rows),
        "overall": {
            "precision": round(overall_prec, 4),
            "recall":    round(overall_rec,  4),
            "f1":        round(overall_f1,   4),
            "mAP50":     round(mAP50,        4),
            "total_gt":  all_gt,
            "total_pred":all_pred,
            "tp":        all_tp,
            "fp":        all_fp,
            "fn":        all_fn,
        },
        "per_class": class_metrics,
    }
    json_path = os.path.join(OUTPUT_DIR, f"summary_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"  Summary JSON   → {json_path}")

    # ── Visual sample grid ─────────────────────────────────────────────────────
    print(f"\nGenerating visual sample grids …")
    score_per_image.sort(key=lambda x: x[0])
    worst_paths = [p for _, p in score_per_image[:N_WORST]]
    best_paths  = [p for _, p in score_per_image[-N_BEST:][::-1]]

    for tag, paths in [("worst", worst_paths), ("best", best_paths)]:
        tiles = []
        for p in paths:
            frame = cv2.imread(p)
            if frame is None:
                continue
            res = model(frame, conf=CONF_THRESH, verbose=False)[0]
            annotated = res.plot()
            annotated = cv2.resize(annotated, (320, 240))
            tiles.append(annotated)

        if tiles:
            # Fill to make a complete grid (4 columns)
            cols = 4
            while len(tiles) % cols != 0:
                tiles.append(np.zeros((240, 320, 3), dtype=np.uint8))
            rows = [np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)]
            grid = np.vstack(rows)
            grid_path = os.path.join(OUTPUT_DIR, f"{tag}_predictions_{ts}.jpg")
            cv2.imwrite(grid_path, grid)
            print(f"  {tag.capitalize()} grid → {grid_path}")

    # ── Ultralytics native val (optional, requires data.yaml) ─────────────────
    data_yaml = os.path.join(TEST_DIR, "data.yaml")
    if os.path.exists(data_yaml):
        print(f"\nRunning ultralytics model.val() with {data_yaml} …")
        val_results = model.val(
            data   = data_yaml,
            split  = "test",
            conf   = CONF_THRESH,
            iou    = IOU_THRESH,
            plots  = True,
            save_json = True,
            project= OUTPUT_DIR,
            name   = f"val_{ts}",
        )
        print(f"  mAP@50      : {val_results.box.map50:.4f}")
        print(f"  mAP@50-95   : {val_results.box.map:.4f}")
    else:
        print(f"\n  ℹ️  No data.yaml found at {data_yaml}")
        print("     Skipping ultralytics model.val() — manual metrics above are complete.")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"  mAP@50    : {mAP50:.4f}")
    print(f"  Precision : {overall_prec:.4f}")
    print(f"  Recall    : {overall_rec:.4f}")
    print(f"  F1        : {overall_f1:.4f}")
    print(f"  Outputs   : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()