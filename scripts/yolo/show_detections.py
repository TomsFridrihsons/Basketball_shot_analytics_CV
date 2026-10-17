import cv2
import torch
from ultralytics import YOLO
import time

# Configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import YOLO_MODEL, RAW_VIDEO_DIR  # noqa: E402

MODEL_PATH = str(YOLO_MODEL)
VIDEO_PATH = str(RAW_VIDEO_DIR / "20240621_223359.mp4")
START_FRAME = 0
TARGET_CLASS = "basket"  # <--- Video will stop when this is detected

# Window sizing
WINDOW_NAME = "Basketball Detection"
MAX_WINDOW_WIDTH = 1280
MAX_WINDOW_HEIGHT = 720

# Pick device automatically
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = YOLO(MODEL_PATH)

# Dictionary to store {track_id: first_seen_frame}
first_appearance_log = {}

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    raise SystemExit(1)

# Set starting frame
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

# Video metadata
video_fps = cap.get(cv2.CAP_PROP_FPS)
if not video_fps or video_fps <= 0:
    video_fps = 30.0

# Create a resizable window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT)

print(f"Using device: {DEVICE}")
print(f"Starting from frame: {START_FRAME}. Target stop class: '{TARGET_CLASS}'")
print("Press 'q' to quit manually")


def resize_to_fit(frame, max_width, max_height):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_width = int(w * scale)
    new_height = int(h * scale)
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


prev_time = time.time()
stop_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    video_time_seconds = current_frame_index / video_fps

    # Run tracking
    results = model.track(frame, device=DEVICE, verbose=False, persist=True)[0]

    # Check for target class and log new detections
    if results.boxes.id is not None:
        track_ids = results.boxes.id.int().cpu().tolist()
        class_indices = results.boxes.cls.int().cpu().tolist()

        for tid, cls_idx in zip(track_ids, class_indices):
            class_name = model.names[cls_idx]

    # Draw and display
    annotated_frame = results.plot()

    now = time.time()
    elapsed = now - prev_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    prev_time = now

    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Frame: {current_frame_index}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                2)

    display_frame = resize_to_fit(annotated_frame, MAX_WINDOW_WIDTH, MAX_WINDOW_HEIGHT)
    cv2.imshow(WINDOW_NAME, display_frame)

    # Break loop if target found or 'q' pressed
    if stop_triggered or (cv2.waitKey(1) & 0xFF == ord("q")):
        # Show the final frame for a moment before closing if it was a target stop
        if stop_triggered:
            cv2.waitKey(2000)  # Wait 2 seconds so you can see the detection
        break

cap.release()
cv2.destroyAllWindows()