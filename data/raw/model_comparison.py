"""
compare_yolo_models.py
----------------------
Compare two YOLO models side-by-side on the same video.

Install dependencies:
    pip install ultralytics opencv-python

Edit the CONFIG block below, then run:
    python compare_yolo_models.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ── colour palette (BGR) ────────────────────────────────────────────────────
PALETTE = [
    (56, 229, 255), (255, 56, 56), (255, 157, 151), (255, 215, 0),
    (0, 255, 127),  (255, 99, 71),  (0, 191, 255),   (255, 20, 147),
    (124, 252, 0),  (255, 140, 0),
]

def color_for(class_id: int) -> tuple:
    return PALETTE[class_id % len(PALETTE)]


# ── drawing helpers ──────────────────────────────────────────────────────────

def add_header(frame: np.ndarray, title: str, fps: float,
               n_det: int, inf_ms: float) -> np.ndarray:
    """Overlay a header bar with model name, FPS and detection count."""
    h, w = frame.shape[:2]
    bar_h = 36
    bar   = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)

    info = f"{title}  |  {n_det} det  |  {inf_ms:.1f} ms  |  {fps:.1f} fps"
    cv2.putText(bar, info, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    return np.vstack([bar, frame])


def add_divider(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack two frames side-by-side with a thin white divider."""
    divider = np.full((left.shape[0], 3, 3), 200, dtype=np.uint8)
    return np.hstack([left, divider, right])


def add_global_bar(combined: np.ndarray, frame_idx: int, total: int) -> np.ndarray:
    """Progress bar + frame counter at the bottom."""
    h, w = combined.shape[:2]
    bar_h = 22
    bar   = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)

    # progress fill
    fill = int(w * frame_idx / max(total, 1))
    bar[:, :fill] = (80, 180, 80)

    text = f"Frame {frame_idx}/{total}"
    cv2.putText(bar, text, (w // 2 - 55, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    return np.vstack([combined, bar])


# ── stats accumulator ────────────────────────────────────────────────────────
class Stats:
    def __init__(self, name: str):
        self.name      = name
        self.inf_times = []
        self.det_counts= []

    def update(self, inf_ms: float, n_det: int):
        self.inf_times.append(inf_ms)
        self.det_counts.append(n_det)

    def summary(self) -> dict:
        if not self.inf_times:
            return {}
        return {
            "model":        self.name,
            "frames":       len(self.inf_times),
            "avg_inf_ms":   round(sum(self.inf_times)  / len(self.inf_times), 2),
            "avg_fps":      round(1000 / (sum(self.inf_times) / len(self.inf_times)), 2),
            "avg_det":      round(sum(self.det_counts)  / len(self.det_counts), 2),
            "total_det":    sum(self.det_counts),
        }


def open_at(video_path: str, start_frame: int) -> cv2.VideoCapture:
    """Open video and seek to start_frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return cap


# Lightweight container for per-frame detection data (no tensors kept alive)
class FrameDet:
    __slots__ = ("boxes", "names", "inf_ms")
    def __init__(self, boxes, names, inf_ms):
        # boxes: list of (x1,y1,x2,y2,conf,cls_id)  — plain Python ints/floats
        self.boxes  = boxes
        self.names  = names
        self.inf_ms = inf_ms


def draw_stored(frame: np.ndarray, det: FrameDet) -> np.ndarray:
    """Draw detections from a FrameDet (no YOLO result object needed)."""
    out = frame.copy()
    for x1, y1, x2, y2, conf, cls_id in det.boxes:
        color = color_for(cls_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{det.names[cls_id]} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def run_model_streaming(video_path: str, model, conf: float,
                         start_frame: int, seg_len: int,
                         name: str) -> tuple[list, Stats]:
    """Stream video, run model, store only lightweight detection data."""
    stats = Stats(name)
    detections = []
    fps_smooth  = 0.0
    cap = open_at(video_path, start_frame)

    print(f"\n[INFO] Running {name} on {seg_len} frames (streaming)...")
    for i in range(seg_len):
        ret, frame = cap.read()
        if not ret:
            break

        t0     = time.perf_counter()
        result = model(frame, verbose=False, conf=conf)
        inf_ms = (time.perf_counter() - t0) * 1000
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1000 / inf_ms)

        # Extract only plain Python data — no GPU tensors kept alive
        boxes_data = []
        for box in result[0].boxes:
            if float(box.conf[0]) >= conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_data.append((x1, y1, x2, y2,
                                   round(float(box.conf[0]), 3),
                                   int(box.cls[0])))

        detections.append(FrameDet(boxes_data, result[0].names, inf_ms))
        stats.update(inf_ms, len(boxes_data))

        if (i + 1) % 100 == 0 or (i + 1) == seg_len:
            print(f"  {i+1}/{seg_len}  {inf_ms:.1f} ms  {fps_smooth:.1f} fps  "
                  f"{len(boxes_data)} det")

    cap.release()
    return detections, stats


# ── main ─────────────────────────────────────────────────────────────────────
def compare(args):
    # ── probe video ──
    cap = open_at(args.video, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"[INFO] Video: {src_w}x{src_h}  {src_fps:.1f} fps  {total_frames} frames")

    # ── 50 %–75 % window ──
    start_frame = int(total_frames * 0.0)
    end_frame   = int(total_frames * 0.10)
    seg_len     = end_frame - start_frame
    print(f"[INFO] Segment: frames {start_frame}–{end_frame}  "
          f"({seg_len} frames, {seg_len/src_fps:.1f} s)")

    # ── load models ──
    print(f"\n[INFO] Loading model 1: {args.model1}")
    model1 = YOLO(args.model1)
    print(f"[INFO] Loading model 2: {args.model2}")
    model2 = YOLO(args.model2)

    # ── pass 1: run model 1, store detections only ──
    dets1, stats1 = run_model_streaming(
        args.video, model1, args.conf, start_frame, seg_len,
        Path(args.model1).stem)
    del model1   # free GPU memory before loading model 2

    # ── pass 2: run model 2, store detections only ──
    dets2, stats2 = run_model_streaming(
        args.video, model2, args.conf, start_frame, seg_len,
        Path(args.model2).stem)
    del model2

    # ── output writer ──
    writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_h  = src_h + 36 + 22   # header + bottom bar
        out_w  = src_w * 2 + 3
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (out_w, out_h))
        print(f"\n[INFO] Writing output to {out_path}")

    # ── pass 3: read video again, compose side-by-side ──
    print("[INFO] Composing comparison video (pass 3)...")
    cap = open_at(args.video, start_frame)

    for i in range(seg_len):
        ret, frame = cap.read()
        if not ret or i >= len(dets1) or i >= len(dets2):
            break

        d1, d2 = dets1[i], dets2[i]

        panel1 = draw_stored(frame, d1)
        panel2 = draw_stored(frame, d2)

        panel1 = add_header(panel1, Path(args.model1).stem,
                            1000 / d1.inf_ms, len(d1.boxes), d1.inf_ms)
        panel2 = add_header(panel2, Path(args.model2).stem,
                            1000 / d2.inf_ms, len(d2.boxes), d2.inf_ms)

        combined = add_divider(panel1, panel2)
        combined = add_global_bar(combined, i + 1, seg_len)

        if writer:
            writer.write(combined)

        if args.show:
            cv2.imshow("YOLO Comparison (press Q to quit)", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit by user.")
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── final summary ──
    print("\n" + "═" * 55)
    print("  SUMMARY")
    print("═" * 55)
    for s in [stats1.summary(), stats2.summary()]:
        print(f"\n  Model      : {s['model']}")
        print(f"  Frames     : {s['frames']}")
        print(f"  Avg inf    : {s['avg_inf_ms']} ms  ({s['avg_fps']} fps)")
        print(f"  Avg dets   : {s['avg_det']}  (total {s['total_det']})")
    print("═" * 55 + "\n")


# ── CONFIGURATION — edit these paths before running ──────────────────────────

VIDEO  = r"C:\Users\fridr\Documents\HooperAI\data\raw\videos\20250711_171647.mp4"
MODEL1 = r"C:\Users\fridr\Documents\HooperAI\archieves\model\best_yolo11.pt"
MODEL2 = r"C:\Users\fridr\Documents\HooperAI\data\model\yolo26m_best.pt"
OUTPUT = r"C:\Users\fridr\Documents\HooperAI\data\processed\evaluation\20250711_171647_comparison2.mp4"
CONF   = 0.75
SHOW   = True

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    class _Args:
        video  = VIDEO
        model1 = MODEL1
        model2 = MODEL2
        output = OUTPUT
        conf   = CONF
        show   = SHOW

    compare(_Args())