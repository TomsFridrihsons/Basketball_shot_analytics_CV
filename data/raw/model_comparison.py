"""
model_comparison.py
───────────────────
Salīdzina divus YOLO modeļus uz viena video.

Metrikas (bez GT anotācijām)
─────────────────────────────
  Detektēšanas līmenis (uz video segmentu):
    • Precision, Recall, F1  — M1 pret M2 kā pseudo-GT (IoU ≥ IOU_THRESH)
    • mAP@0.5 un mAP@0.5:0.95  — vidējā precizitāte pa klasi un IoU slieksni
    • Per-klases AP  — atsevišķi ball / basket / player
    • Confidence sadalījums  — vidējā un mediānas ticamība

  Ātruma līmenis:
    • Vidējais secinājuma laiks (ms) un fps
    • Min / max / std latency

  Vizualizācija (PNG faili):
    • metrics_summary.png   — joslu diagramma pa klasi un modeli
    • pr_curves.png         — Precision-Recall līknes pa klasi
    • conf_distribution.png — confidence histogrammas

Palaišana:
    python model_comparison.py
"""

import time
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURĀCIJA
# ══════════════════════════════════════════════════════════════════════════════
VIDEO  = r"C:\Users\fridr\Documents\HooperAI\data\raw\videos\20250711_171647.mp4"
MODEL1 = r"C:\Users\fridr\Documents\HooperAI\archieves\model\best_yolo11.pt"
MODEL2 = r"C:\Users\fridr\Documents\HooperAI\data\model\yolo26m_best.pt"
OUTPUT = r"C:\Users\fridr\Documents\HooperAI\data\processed\evaluation\20250711_171647_comparison.mp4"
METRICS_DIR = r"C:\Users\fridr\Documents\HooperAI\data\processed\evaluation\metrics"

CONF        = 0.75    # YOLO confidence slieksnis
SHOW        = True    # rādīt OpenCV logu
IOU_THRESH  = 0.5     # IoU slieksnis matched/unmatched nolēmumam
MAP_THRESHS = np.arange(0.50, 1.00, 0.05).tolist()  # 0.50 … 0.95 (COCO stils)

# Video segments (0.0–1.0 no kopējā garuma)
SEG_START = 0.0
SEG_END   = 0.10
# ══════════════════════════════════════════════════════════════════════════════


# ── krāsu palete (BGR) ───────────────────────────────────────────────────────
PALETTE = [
    (56, 229, 255), (255, 56, 56), (255, 157, 151), (255, 215, 0),
    (0, 255, 127),  (255, 99, 71),  (0, 191, 255),  (255, 20, 147),
    (124, 252, 0),  (255, 140, 0),
]
# Matplotlib krāsas modeļiem
PLT_COLORS = {"m1": "#f97316", "m2": "#3b82f6"}   # oranžs / zils


def color_for(class_id: int) -> tuple:
    return PALETTE[class_id % len(PALETTE)]


# ── IoU palīgfunkcijas ───────────────────────────────────────────────────────

def box_iou(b1, b2) -> float:
    """IoU starp divām kastēm (x1,y1,x2,y2)."""
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


def match_boxes(pred_boxes, gt_boxes, iou_thr: float) -> tuple:
    """
    Atrod maksimālo bipartīto saskaņojumu starp pred un gt pēc IoU.
    Atgriež (tp_count, matched_pred_indices, matched_gt_indices).
    """
    if not pred_boxes or not gt_boxes:
        return 0, [], []

    # Sakārtot pred pēc confidence (dilstošā secībā)
    pred_sorted = sorted(enumerate(pred_boxes),
                         key=lambda x: x[1][4], reverse=True)
    gt_matched = [False] * len(gt_boxes)
    tp_idx_pred, tp_idx_gt = [], []

    for pred_i, (orig_pi, pb) in enumerate(pred_sorted):
        best_iou, best_gj = 0.0, -1
        for gj, gb in enumerate(gt_boxes):
            if gt_matched[gj]:
                continue
            iou = box_iou(pb[:4], gb[:4])
            if iou > best_iou:
                best_iou, best_gj = iou, gj
        if best_iou >= iou_thr and best_gj >= 0:
            gt_matched[best_gj] = True
            tp_idx_pred.append(orig_pi)
            tp_idx_gt.append(best_gj)

    return len(tp_idx_pred), tp_idx_pred, tp_idx_gt


# ── AP aprēķins no confidence scores ─────────────────────────────────────────

def compute_ap(tp_flags: list, conf_scores: list, n_gt: int) -> float:
    """
    Aprēķina Average Precision no bināro TP karodziņu un confidence saraksta.
    11-punktu interpolācija.
    """
    if n_gt == 0 or not tp_flags:
        return 0.0

    # Kārtot pēc confidence dilstošā secībā
    paired = sorted(zip(conf_scores, tp_flags), reverse=True)
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []

    for _, is_tp in paired:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)

    # 11-punktu interpolācija
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = [p for p, r in zip(precisions, recalls) if r >= thr]
        ap += max(prec_at_thr) if prec_at_thr else 0.0
    return ap / 11.0


def pr_curve_points(tp_flags: list, conf_scores: list,
                    n_gt: int) -> tuple:
    """Atgriež (recalls, precisions) vektorus PR līknes zīmēšanai."""
    if n_gt == 0 or not tp_flags:
        return [0, 1], [1, 0]
    paired = sorted(zip(conf_scores, tp_flags), reverse=True)
    tp_cum, fp_cum = 0, 0
    precs, recs = [1.0], [0.0]
    for _, is_tp in paired:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precs.append(tp_cum / (tp_cum + fp_cum))
        recs.append(tp_cum / n_gt)
    return recs, precs


# ── Datu klases ──────────────────────────────────────────────────────────────

class FrameDet:
    """Viena kadra detektēšanas rezultāts (bez GPU tensoriem)."""
    __slots__ = ("boxes", "names", "inf_ms")

    def __init__(self, boxes: list, names: dict, inf_ms: float):
        # boxes: list of (x1,y1,x2,y2, conf, cls_id)
        self.boxes  = boxes
        self.names  = names
        self.inf_ms = inf_ms


class MetricsTracker:
    """
    Uzkrāj per-frame detektēšanas datus un aprēķina kvantitatīvās metrikas
    pret otru modeli (pseudo-GT) vai reālu GT (ja pieejams).
    """

    def __init__(self, name: str):
        self.name = name
        self.inf_times: list[float] = []
        # per-klase: confidence scores, TP karodziņi (pēc match_vs())
        self.cls_conf:   defaultdict = defaultdict(list)   # cls_id -> [conf]
        self.cls_tp:     defaultdict = defaultdict(list)   # cls_id -> [bool]
        self.cls_n_gt:   defaultdict = defaultdict(int)    # cls_id -> int
        self.total_dets: int = 0

    def record_frame(self, det: FrameDet):
        """Reģistrē ātruma un detektēšanas datus."""
        self.inf_times.append(det.inf_ms)
        self.total_dets += len(det.boxes)

    def match_vs(self, pred_det: FrameDet, gt_det: FrameDet,
                 iou_thr: float):
        """
        Salīdzina šī modeļa detektēšanu (pred) pret cita modeļa (gt)
        kā pseudo-GT.  Uzrāda TP/FP per detektēšana per klase.
        """
        # Sagrupēt pēc klases
        pred_by_cls: defaultdict = defaultdict(list)
        for b in pred_det.boxes:
            pred_by_cls[b[5]].append(b)

        gt_by_cls: defaultdict = defaultdict(list)
        for b in gt_det.boxes:
            gt_by_cls[b[5]].append(b)

        all_classes = set(pred_by_cls) | set(gt_by_cls)
        for cls in all_classes:
            preds = pred_by_cls[cls]
            gts   = gt_by_cls[cls]
            self.cls_n_gt[cls] += len(gts)

            if not preds:
                continue

            tp_count, tp_pi, _ = match_boxes(preds, gts, iou_thr)
            tp_set = set(tp_pi)
            for pi, pb in enumerate(preds):
                self.cls_conf[cls].append(pb[4])
                self.cls_tp[cls].append(pi in tp_set)

    def ap_at(self, cls_id: int, iou_thr: float,
              pred_dets: list, gt_dets: list) -> float:
        """AP@iou_thr priekš vienas klases, pārrēķina ar dotajiem IoU."""
        pred_by_frame = [defaultdict(list) for _ in pred_dets]
        gt_by_frame   = [defaultdict(list) for _ in gt_dets]
        for fi, d in enumerate(pred_dets):
            for b in d.boxes:
                pred_by_frame[fi][b[5]].append(b)
        for fi, d in enumerate(gt_dets):
            for b in d.boxes:
                gt_by_frame[fi][b[5]].append(b)

        all_conf, all_tp = [], []
        n_gt = 0
        for fi in range(len(pred_dets)):
            preds = pred_by_frame[fi][cls_id]
            gts   = gt_by_frame[fi][cls_id]
            n_gt += len(gts)
            if not preds:
                continue
            tp_count, tp_pi, _ = match_boxes(preds, gts, iou_thr)
            tp_set = set(tp_pi)
            for pi, pb in enumerate(preds):
                all_conf.append(pb[4])
                all_tp.append(pi in tp_set)

        return compute_ap(all_tp, all_conf, n_gt)

    def compute_all_metrics(self, pred_dets: list, gt_dets: list,
                            class_names: dict) -> dict:
        """
        Aprēķina pilno metrikas kopu:
          - per-klases AP@0.5, AP@0.5:0.95
          - mAP@0.5, mAP@0.5:0.95
          - Precision, Recall, F1 per klase un kopā
          - ātruma statistika
        """
        all_classes = sorted(class_names.keys())
        results = {
            "model":        self.name,
            "per_class":    {},
            "speed": {
                "avg_inf_ms":  round(np.mean(self.inf_times), 2),
                "std_inf_ms":  round(np.std(self.inf_times),  2),
                "min_inf_ms":  round(np.min(self.inf_times),  2),
                "max_inf_ms":  round(np.max(self.inf_times),  2),
                "avg_fps":     round(1000 / np.mean(self.inf_times), 2),
                "total_dets":  self.total_dets,
                "avg_dets_per_frame": round(
                    self.total_dets / max(len(self.inf_times), 1), 2),
            },
        }

        map50_vals, map5095_vals = [], []
        total_tp = total_fp = total_fn = 0

        for cls_id in all_classes:
            cls_name = class_names.get(cls_id, str(cls_id))
            conf_list = self.cls_conf[cls_id]
            tp_list   = self.cls_tp[cls_id]
            n_gt      = self.cls_n_gt[cls_id]

            # AP@0.5 no uzkrātajiem TP/FP
            ap50 = compute_ap(tp_list, conf_list, n_gt)

            # AP@0.5:0.95 — pārrēķina katram IoU slieksnim
            ap_vals = [self.ap_at(cls_id, thr, pred_dets, gt_dets)
                       for thr in MAP_THRESHS]
            ap5095 = float(np.mean(ap_vals))

            # Precision, Recall, F1 pēc optimālā confidence sliekšņa
            tp_count = sum(tp_list)
            fp_count = len(tp_list) - tp_count
            fn_count = max(0, n_gt - tp_count)

            precision = tp_count / max(tp_count + fp_count, 1)
            recall    = tp_count / max(n_gt, 1)
            f1        = (2 * precision * recall / max(precision + recall, 1e-9))

            # Confidence statistika
            confs = [conf_list[i] for i, t in enumerate(tp_list) if t]
            avg_conf_tp = float(np.mean(confs)) if confs else 0.0

            results["per_class"][cls_name] = {
                "ap50":        round(ap50,      4),
                "ap50_95":     round(ap5095,    4),
                "precision":   round(precision, 4),
                "recall":      round(recall,    4),
                "f1":          round(f1,        4),
                "tp":          tp_count,
                "fp":          fp_count,
                "fn":          fn_count,
                "n_gt":        n_gt,
                "avg_conf_tp": round(avg_conf_tp, 4),
            }
            map50_vals.append(ap50)
            map5095_vals.append(ap5095)
            total_tp += tp_count
            total_fp += fp_count
            total_fn += fn_count

        # Kopējās metrikas
        overall_prec = total_tp / max(total_tp + total_fp, 1)
        overall_rec  = total_tp / max(total_tp + total_fn, 1)
        overall_f1   = (2 * overall_prec * overall_rec /
                        max(overall_prec + overall_rec, 1e-9))

        results["overall"] = {
            "mAP50":     round(float(np.mean(map50_vals)),   4),
            "mAP50_95":  round(float(np.mean(map5095_vals)), 4),
            "precision": round(overall_prec, 4),
            "recall":    round(overall_rec,  4),
            "f1":        round(overall_f1,   4),
            "total_tp":  total_tp,
            "total_fp":  total_fp,
            "total_fn":  total_fn,
        }
        return results


# ── Video uzrakstu funkcijas ──────────────────────────────────────────────────

def add_header(frame, title, fps, n_det, inf_ms):
    h, w = frame.shape[:2]
    bar   = np.zeros((36, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    info = f"{title}  |  {n_det} det  |  {inf_ms:.1f} ms  |  {fps:.1f} fps"
    cv2.putText(bar, info, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, frame])


def add_divider(left, right):
    divider = np.full((left.shape[0], 3, 3), 200, dtype=np.uint8)
    return np.hstack([left, divider, right])


def add_global_bar(combined, frame_idx, total):
    h, w = combined.shape[:2]
    bar   = np.zeros((22, w, 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)
    fill = int(w * frame_idx / max(total, 1))
    bar[:, :fill] = (80, 180, 80)
    cv2.putText(bar, f"Frame {frame_idx}/{total}",
                (w // 2 - 55, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([combined, bar])


def draw_stored(frame, det: FrameDet):
    out = frame.copy()
    for x1, y1, x2, y2, conf, cls_id in det.boxes:
        color = color_for(cls_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{det.names.get(cls_id, cls_id)} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return out


# ── Inference gājiens ─────────────────────────────────────────────────────────

def run_model_streaming(video_path, model, conf, start_frame,
                         seg_len, name) -> tuple:
    """Apstrādā video segmentu, saglabā tikai vieglos FrameDet objektus."""
    tracker    = MetricsTracker(name)
    detections = []
    fps_smooth = 0.0

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"\n[INFO] Palaiž {name} uz {seg_len} kadriem...")
    for i in range(seg_len):
        ret, frame = cap.read()
        if not ret:
            break

        t0     = time.perf_counter()
        result = model(frame, verbose=False, conf=conf)[0]
        inf_ms = (time.perf_counter() - t0) * 1000
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1000 / inf_ms)

        boxes_data = []
        for box in result.boxes:
            if float(box.conf[0]) >= conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_data.append((x1, y1, x2, y2,
                                   round(float(box.conf[0]), 3),
                                   int(box.cls[0])))

        fd = FrameDet(boxes_data, result.names, inf_ms)
        detections.append(fd)
        tracker.record_frame(fd)

        if (i + 1) % 100 == 0 or (i + 1) == seg_len:
            print(f"  {i+1}/{seg_len}  {inf_ms:.1f} ms  "
                  f"{fps_smooth:.1f} fps  {len(boxes_data)} det")

    cap.release()
    return detections, tracker


# ── Vizualizācija ─────────────────────────────────────────────────────────────

def _dark_fig(w=14, h=7):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor("#0f1117")
    return fig


def plot_metrics_summary(m1: dict, m2: dict, out_path: str):
    """Joslu diagramma: AP50, AP50-95, Precision, Recall, F1 pa klasi."""
    classes  = list(m1["per_class"].keys())
    metrics  = ["ap50", "ap50_95", "precision", "recall", "f1"]
    m_labels = ["AP@0.5", "AP@0.5:0.95", "Precision", "Recall", "F1"]
    n_cls    = len(classes)
    n_met    = len(metrics)

    fig = _dark_fig(w=5 + n_cls * n_met * 0.6, h=8)
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.45,
                            figure=fig)

    # ── joslu diagramma ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("#181c27")

    x      = np.arange(n_cls)
    n_bars = n_met * 2
    width  = 0.8 / n_bars

    for mi, (metric, label) in enumerate(zip(metrics, m_labels)):
        for ki, key in enumerate(["m1", "m2"]):
            mdata   = m1 if key == "m1" else m2
            vals    = [mdata["per_class"][c][metric] for c in classes]
            color   = PLT_COLORS[key]
            alpha   = 0.9 if key == "m1" else 0.65
            offset  = (mi * 2 + ki - n_bars / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          color=color, alpha=alpha,
                          label=f"{mdata['model']} – {label}" if ki == 0 else None)
            # vērtību uzraksti
            for bar, v in zip(bars, vals):
                if v > 0.02:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom",
                            color="#d1d5db", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, color="#e8e8e8", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Vērtība", color="#9ca3af", fontsize=10)
    ax.set_title("Metrikas pa klasi un modeli", color="#f97316",
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#6b7280")
    ax.spines[:].set_color("#252a38")
    ax.yaxis.grid(True, color="#252a38", linewidth=0.6)
    ax.set_axisbelow(True)

    # Leģenda
    handles = [
        plt.Rectangle((0,0), 1, 1, color=PLT_COLORS["m1"], alpha=0.9,
                       label=m1["model"]),
        plt.Rectangle((0,0), 1, 1, color=PLT_COLORS["m2"], alpha=0.65,
                       label=m2["model"]),
    ]
    ax.legend(handles=handles, loc="upper right",
              facecolor="#181c27", edgecolor="#252a38",
              labelcolor="#e8e8e8", fontsize=9)

    # ── kopsavilkuma tabula ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#181c27")
    ax2.axis("off")

    rows  = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1",
             "Avg inf (ms)", "Avg FPS"]
    v1    = m1["overall"]
    v2    = m2["overall"]
    sp1   = m1["speed"]
    sp2   = m2["speed"]
    col1  = [f"{v1['mAP50']:.3f}", f"{v1['mAP50_95']:.3f}",
             f"{v1['precision']:.3f}", f"{v1['recall']:.3f}",
             f"{v1['f1']:.3f}",
             f"{sp1['avg_inf_ms']}", f"{sp1['avg_fps']}"]
    col2  = [f"{v2['mAP50']:.3f}", f"{v2['mAP50_95']:.3f}",
             f"{v2['precision']:.3f}", f"{v2['recall']:.3f}",
             f"{v2['f1']:.3f}",
             f"{sp2['avg_inf_ms']}", f"{sp2['avg_fps']}"]

    table = ax2.table(
        cellText  = [[r, c1, c2] for r, c1, c2 in zip(rows, col1, col2)],
        colLabels = ["Metrika", m1["model"], m2["model"]],
        loc       = "center",
        cellLoc   = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Tumšs stils tabulai
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("#0f1117" if row == 0 else "#181c27")
        cell.set_edgecolor("#252a38")
        cell.set_text_props(color="#e8e8e8" if row > 0 else "#f97316",
                            fontweight="bold" if row == 0 else "normal")

    fig.suptitle("YOLO modeļu salīdzinājums – kvantitatīvās metrikas",
                 color="#f97316", fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Metrikas diagramma   -> {out_path}")


def plot_pr_curves(m1_dets: list, m2_dets: list,
                   tracker1: MetricsTracker, tracker2: MetricsTracker,
                   class_names: dict, out_path: str):
    """PR līknes katrai klasei abiem modeļiem."""
    classes  = sorted(class_names.keys())
    n_cls    = len(classes)
    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 5))
    fig.patch.set_facecolor("#0f1117")
    if n_cls == 1:
        axes = [axes]

    for ax, cls_id in zip(axes, classes):
        ax.set_facecolor("#181c27")
        cls_name = class_names[cls_id]

        for (tracker, dets, key, name) in [
            (tracker1, m1_dets, "m1", tracker1.name),
            (tracker2, m2_dets, "m2", tracker2.name),
        ]:
            # Savākt TP/conf pa kadriem ar IoU@0.5 (izmantojot otru kā GT)
            gt_dets = m2_dets if key == "m1" else m1_dets
            conf_l, tp_l = [], []
            n_gt = 0
            for fi, (pd, gd) in enumerate(zip(dets, gt_dets)):
                pb_cls = [b for b in pd.boxes if b[5] == cls_id]
                gb_cls = [b for b in gd.boxes if b[5] == cls_id]
                n_gt += len(gb_cls)
                if not pb_cls:
                    continue
                _, tp_pi, _ = match_boxes(pb_cls, gb_cls, IOU_THRESH)
                tp_set = set(tp_pi)
                for pi, pb in enumerate(pb_cls):
                    conf_l.append(pb[4])
                    tp_l.append(pi in tp_set)

            recs, precs = pr_curve_points(tp_l, conf_l, n_gt)
            ap = compute_ap(tp_l, conf_l, n_gt)

            ax.plot(recs, precs, color=PLT_COLORS[key], linewidth=2,
                    label=f"{name}  AP={ap:.3f}")
            ax.fill_between(recs, precs, alpha=0.12,
                            color=PLT_COLORS[key])

        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall",    color="#9ca3af", fontsize=10)
        ax.set_ylabel("Precision", color="#9ca3af", fontsize=10)
        ax.set_title(cls_name, color="#e8e8e8", fontsize=11,
                     fontweight="bold")
        ax.tick_params(colors="#6b7280")
        ax.spines[:].set_color("#252a38")
        ax.grid(True, color="#252a38", linewidth=0.5)
        ax.legend(facecolor="#181c27", edgecolor="#252a38",
                  labelcolor="#e8e8e8", fontsize=8)

    fig.suptitle("Precision-Recall līknes pa klasi",
                 color="#f97316", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  PR līknes            -> {out_path}")


def plot_conf_distribution(m1_dets: list, m2_dets: list,
                           class_names: dict, out_path: str):
    """Confidence sadalījuma histogrammas pa klasi."""
    classes = sorted(class_names.keys())
    n_cls   = len(classes)
    fig, axes = plt.subplots(1, n_cls, figsize=(4 * n_cls, 4))
    fig.patch.set_facecolor("#0f1117")
    if n_cls == 1:
        axes = [axes]

    for ax, cls_id in zip(axes, classes):
        ax.set_facecolor("#181c27")
        cls_name = class_names[cls_id]

        for dets, key, name in [(m1_dets, "m1", ""), (m2_dets, "m2", "")]:
            confs = [b[4] for d in dets for b in d.boxes if b[5] == cls_id]
            if not confs:
                continue
            label = (f"{tracker1.name if key=='m1' else tracker2.name}\n"
                     f"n={len(confs)}  "
                     f"μ={np.mean(confs):.3f}  "
                     f"med={np.median(confs):.3f}")
            ax.hist(confs, bins=20, range=(0, 1),
                    color=PLT_COLORS[key], alpha=0.65,
                    label=label, edgecolor="#252a38")
            ax.axvline(np.mean(confs), color=PLT_COLORS[key],
                       linestyle="--", linewidth=1.5, alpha=0.9)

        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("Confidence",   color="#9ca3af", fontsize=10)
        ax.set_ylabel("Detektēšanu skaits", color="#9ca3af", fontsize=10)
        ax.set_title(cls_name, color="#e8e8e8", fontsize=11,
                     fontweight="bold")
        ax.tick_params(colors="#6b7280")
        ax.spines[:].set_color("#252a38")
        ax.legend(facecolor="#0f1117", edgecolor="#252a38",
                  labelcolor="#d1d5db", fontsize=7)

    fig.suptitle("Confidence sadalījums pa klasi",
                 color="#f97316", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Confidence histogramma -> {out_path}")


def plot_speed_comparison(m1: dict, m2: dict, out_path: str):
    """Ātruma metrikas: latency un fps joslu diagramma."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.patch.set_facecolor("#0f1117")

    names  = [m1["model"], m2["model"]]
    colors = [PLT_COLORS["m1"], PLT_COLORS["m2"]]

    for ax, metric, unit, title in [
        (ax1, "avg_inf_ms", "ms",  "Vidējais secinājuma laiks"),
        (ax2, "avg_fps",    "fps", "Kadri sekundē (FPS)"),
    ]:
        ax.set_facecolor("#181c27")
        vals = [m["speed"][metric] for m in [m1, m2]]
        bars = ax.bar(names, vals, color=colors, width=0.4,
                      edgecolor="#252a38")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{v:.1f} {unit}",
                    ha="center", color="#e8e8e8", fontsize=11,
                    fontweight="bold")
        ax.set_title(title, color="#e8e8e8", fontsize=11,
                     fontweight="bold")
        ax.set_ylabel(unit, color="#9ca3af")
        ax.tick_params(colors="#6b7280")
        ax.spines[:].set_color("#252a38")
        ax.set_facecolor("#181c27")
        ax.yaxis.grid(True, color="#252a38", linewidth=0.5)
        ax.set_axisbelow(True)

        # Kļūdu joslas (std)
        stds = [m["speed"]["std_inf_ms"] for m in [m1, m2]]
        if metric == "avg_inf_ms":
            ax.errorbar(names, vals, yerr=stds, fmt="none",
                        ecolor="#e8e8e8", elinewidth=1.5, capsize=5)

    fig.suptitle("Ātruma salīdzinājums",
                 color="#f97316", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Ātruma diagramma     -> {out_path}")


# ── Galvenā funkcija ──────────────────────────────────────────────────────────

# Globālie tracker reference (vajadzīgs plot_conf_distribution closure)
tracker1 = tracker2 = None


def compare(args):
    global tracker1, tracker2

    # ── Video parametri ──
    cap          = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"[INFO] Video: {src_w}x{src_h}  {src_fps:.1f} fps  "
          f"{total_frames} kadri")

    start_frame = int(total_frames * SEG_START)
    end_frame   = int(total_frames * SEG_END)
    seg_len     = end_frame - start_frame
    print(f"[INFO] Segments: kadri {start_frame}–{end_frame}  "
          f"({seg_len} kadri, {seg_len/src_fps:.1f} s)")

    # ── Ielādē modeļus ──
    print(f"\n[INFO] Ielādē modeli 1: {args.model1}")
    model1 = YOLO(args.model1)
    print(f"[INFO] Ielādē modeli 2: {args.model2}")
    model2 = YOLO(args.model2)

    # ── 1. gājiens: M1 ──
    dets1, tracker1 = run_model_streaming(
        args.video, model1, args.conf,
        start_frame, seg_len, Path(args.model1).stem)
    del model1

    # ── 2. gājiens: M2 ──
    dets2, tracker2 = run_model_streaming(
        args.video, model2, args.conf,
        start_frame, seg_len, Path(args.model2).stem)
    del model2

    # ── Klašu nosaukumi ──
    class_names = dets1[0].names if dets1 else (dets2[0].names if dets2 else {})

    # ── Metrikas: M1 pret M2 kā pseudo-GT un otrādi ──
    print("\n[INFO] Aprēķina metrikas (M1 pret M2 kā pseudo-GT)...")
    for fi, (d1, d2) in enumerate(zip(dets1, dets2)):
        tracker1.match_vs(d1, d2, IOU_THRESH)
        tracker2.match_vs(d2, d1, IOU_THRESH)

    metrics1 = tracker1.compute_all_metrics(dets1, dets2, class_names)
    metrics2 = tracker2.compute_all_metrics(dets2, dets1, class_names)

    # ── Konsoles izvade ──
    print("\n" + "═" * 62)
    print("  KVANTITATĪVĀS METRIKAS")
    print("═" * 62)
    for m in [metrics1, metrics2]:
        ov = m["overall"]
        sp = m["speed"]
        print(f"\n  Modelis      : {m['model']}")
        print(f"  mAP@0.5      : {ov['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95 : {ov['mAP50_95']:.4f}")
        print(f"  Precision    : {ov['precision']:.4f}")
        print(f"  Recall       : {ov['recall']:.4f}")
        print(f"  F1           : {ov['f1']:.4f}")
        print(f"  Avg inf      : {sp['avg_inf_ms']} ms  "
              f"(±{sp['std_inf_ms']} ms)  {sp['avg_fps']} fps")
        print(f"\n  Pa klasēm:")
        print(f"  {'Klase':<12} {'AP@0.5':>8} {'AP@0.5:0.95':>12} "
              f"{'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
        print(f"  {'─'*12} {'─'*8} {'─'*12} {'─'*8} {'─'*8} {'─'*8} "
              f"{'─'*5} {'─'*5} {'─'*5}")
        for cls_name, cv in m["per_class"].items():
            print(f"  {cls_name:<12} {cv['ap50']:>8.4f} {cv['ap50_95']:>12.4f} "
                  f"{cv['precision']:>8.4f} {cv['recall']:>8.4f} "
                  f"{cv['f1']:>8.4f} {cv['tp']:>5} {cv['fp']:>5} {cv['fn']:>5}")
    print("\n" + "═" * 62)

    # ── Saglabā JSON ──
    metrics_dir = Path(METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "metrics_m1.json", "w") as f:
        json.dump(metrics1, f, indent=4)
    with open(metrics_dir / "metrics_m2.json", "w") as f:
        json.dump(metrics2, f, indent=4)

    # ── Zīmē grafikus ──
    print("\n[INFO] Zīmē grafikus...")
    plot_metrics_summary(metrics1, metrics2,
                         str(metrics_dir / "metrics_summary.png"))
    plot_pr_curves(dets1, dets2, tracker1, tracker2, class_names,
                   str(metrics_dir / "pr_curves.png"))
    plot_conf_distribution(dets1, dets2, class_names,
                           str(metrics_dir / "conf_distribution.png"))
    plot_speed_comparison(metrics1, metrics2,
                          str(metrics_dir / "speed_comparison.png"))

    # ── 3. gājiens: video kompozīcija ──
    writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_h  = src_h + 36 + 22
        out_w  = src_w * 2 + 3
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            src_fps, (out_w, out_h))
        print(f"\n[INFO] Raksta video uz {out_path}")

    print("[INFO] Kompozē salīdzinājuma video (3. gājiens)...")
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(seg_len):
        ret, frame = cap.read()
        if not ret or i >= len(dets1) or i >= len(dets2):
            break

        d1, d2   = dets1[i], dets2[i]
        panel1   = draw_stored(frame, d1)
        panel2   = draw_stored(frame, d2)
        panel1   = add_header(panel1, Path(args.model1).stem,
                              1000 / d1.inf_ms, len(d1.boxes), d1.inf_ms)
        panel2   = add_header(panel2, Path(args.model2).stem,
                              1000 / d2.inf_ms, len(d2.boxes), d2.inf_ms)
        combined = add_divider(panel1, panel2)
        combined = add_global_bar(combined, i + 1, seg_len)

        if writer:
            writer.write(combined)
        if args.show:
            cv2.imshow("YOLO salīdzinājums (Q — iziet)", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Lietotājs pārtrauca.")
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n[INFO] Pabeigts.")
    print(f"  Metrikas JSON  -> {metrics_dir}")
    print(f"  Grafiki        -> {metrics_dir}")
    if args.output:
        print(f"  Video          -> {args.output}")


# ── Palaišana ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    class _Args:
        video  = VIDEO
        model1 = MODEL1
        model2 = MODEL2
        output = OUTPUT
        conf   = CONF
        show   = SHOW

    compare(_Args())