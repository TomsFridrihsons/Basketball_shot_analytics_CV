"""
split.py  –  Interactive Basketball Shot Labeler
=================================================
Records per-frame YOLO detections during each shot and saves a CSV
that lstm_train.py can consume directly with zero pre-processing.

Changes vs. original split.py
──────────────────────────────
1.  Two-stage shot detection (possible → confirmed, ≥10 frames elevated)
      Matches lstm_infer.py so the labeler sees the same shots the model will.

2.  Basket position fallback
      Last-known basket coords are reused when the basket is not detected,
      preventing NaN columns that lstm_train.py fills with 0 (silent data loss).

3.  Minimum shot length guard (MIN_SHOT_FRAMES = 8)
      Shots shorter than this are auto-discarded at labeling time with a warning,
      because the LSTM attention mechanism needs enough timesteps to be meaningful.

4.  Duplicate-CSV append mode
      If OUTPUT_CSV already exists, new shots are appended rather than overwriting,
      so you can label across multiple sessions safely.

5.  Column order matches lstm_train.py FEATURE_COLS exactly
      The trainer reads columns by name so order does not matter, but keeping them
      aligned makes visual inspection easier and avoids future drift.

6.  Shot-level validation report on exit
      Prints min/mean/max sequence length so you can spot extremely short shots
      before training.

7.  'r' key to replay the last shot summary
      Helpful when you lose track of what you just watched.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


# ─── configuration ────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\fridr\Documents\HooperAI\data\model\yolo26m_best.pt"
VIDEO_PATH = r"C:\Users\fridr\Documents\HooperAI\data\raw\videos\20250711_171647.mp4"
OUTPUT_CSV = r"C:\Users\fridr\Documents\HooperAI\data\processed\annotations\basketball_shot_data_labeled_3103.csv"

YOLO_CONF       = 0.75   # detection confidence threshold
MIN_SHOT_FRAMES = 8      # discard shots shorter than this (too few timesteps for LSTM)
CONFIRM_FRAMES  = 10     # frames ball must stay elevated to confirm a real shot
DISPLAY_WIDTH   = 1280
# ─────────────────────────────────────────────────────────────────────────────


class InteractiveShotDetector:

    # Column order mirrors lstm_train.py FEATURE_COLS + metadata
    _META_COLS = [
        'frame_id', 'shot_number', 'frame_count',
        'frame_within_shot', 'timestamp',
    ]
    _FEATURE_COLS = [
        'ball_x', 'ball_y', 'ball_w', 'ball_h',
        'ball_velocity_x', 'ball_velocity_y',
        'player_x', 'player_y', 'player_w', 'player_h',
        'basket_x', 'basket_y', 'basket_w', 'basket_h',
        'ball_player_dist', 'ball_basket_dist', 'ball_basket_angle',
        'ball_above_player',
    ]
    _EXTRA_COLS = [
        'player_top', 'player_bottom',
        'num_balls', 'num_players', 'num_baskets',
    ]
    _LABEL_COL = ['shot_result']

    def __init__(self, model_path: str, video_path: str, output_csv: str):
        self.model      = YOLO(model_path)
        self.video_path = video_path
        self.output_csv = output_csv

        # ── shot state machine ────────────────────────────────────────────────
        self.possible_shot       = False   # ball elevated but not yet confirmed
        self.shot_in_progress    = False   # confirmed shot being recorded
        self.poss_frame_count    = 0       # consecutive elevated frames so far
        self.shot_number         = 0
        self.frame_count         = 0
        self.shot_start_frame    = 0

        # ── data buffers ──────────────────────────────────────────────────────
        self.pre_data            = []      # frames collected during 'possible' stage
        self.current_shot_data   = []      # frames collected during confirmed shot
        self.labeled_shot_data   = []      # committed rows (all labeled shots)

        # ── basket fallback ───────────────────────────────────────────────────
        self._last_basket: tuple | None = None   # (x, y, w, h)

        # ── misc ──────────────────────────────────────────────────────────────
        self._last_shot_summary  = ""      # for 'r' replay key

    # ── public entry point ────────────────────────────────────────────────────

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)

        print(f"Video  : {self.video_path}")
        print(f"Frames : {total_frames}  |  FPS : {fps:.1f}")
        self._print_controls()

        paused             = False
        waiting_for_label  = False

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1

                # YOLO detection
                results       = self.model(frame, conf=YOLO_CONF, verbose=False)[0]
                ball_boxes, player_boxes, basket_boxes = self._parse_boxes(results)

                # State machine
                shot_ended = self._update_state(ball_boxes, player_boxes,
                                                basket_boxes, fps)

                # Annotated overlay
                annotated = results.plot()
                self._draw_overlay(annotated, total_frames)

                h, w  = annotated.shape[:2]
                scale = DISPLAY_WIDTH / w
                display = cv2.resize(annotated, (DISPLAY_WIDTH, int(h * scale)))
                cv2.imshow('Basketball Shot Labeling', display)

                if shot_ended:
                    paused            = True
                    waiting_for_label = True
                    n = len(self.current_shot_data)
                    print(f"\n>>> Shot #{self.shot_number}  ({n} frames) – "
                          f"press  1=Made  0=Missed  2=Discard <<<")
                    if n < MIN_SHOT_FRAMES:
                        print(f"    ⚠️  Only {n} frames – below MIN_SHOT_FRAMES={MIN_SHOT_FRAMES}."
                              f"  Auto-discarding.")
                        self._discard_shot(reason="too short")
                        paused            = False
                        waiting_for_label = False

                if self.frame_count % 100 == 0:
                    print(f"  [{self.frame_count}/{total_frames}]")

            # ── keyboard ──────────────────────────────────────────────────────
            key = cv2.waitKey(1 if not paused else 0) & 0xFF

            if key == ord('q'):
                print("\nQuitting …")
                break

            elif key == ord('x') and (self.shot_in_progress or self.possible_shot):
                self._cancel_shot()

            elif key == ord('r'):
                print(self._last_shot_summary or "  (no shot recorded yet)")

            elif waiting_for_label:
                if key == ord('1'):
                    self._save_shot(label=1, label_name="Made")
                    paused = waiting_for_label = False
                elif key == ord('0'):
                    self._save_shot(label=0, label_name="Missed")
                    paused = waiting_for_label = False
                elif key == ord('2'):
                    self._discard_shot(reason="user discarded")
                    paused = waiting_for_label = False

        cap.release()
        cv2.destroyAllWindows()
        self._save_final_data()

    # ── detection parsing ─────────────────────────────────────────────────────

    def _parse_boxes(self, results):
        """Split YOLO results into ball / player / basket lists."""
        ball_boxes = []; player_boxes = []; basket_boxes = []
        for box in results.boxes:
            cls  = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()
            if   cls == 0: ball_boxes.append(xyxy)
            elif cls == 1: basket_boxes.append(xyxy)
            elif cls == 2: player_boxes.append(xyxy)
        return ball_boxes, player_boxes, basket_boxes

    # ── two-stage shot detection state machine ────────────────────────────────

    def _update_state(self, ball_boxes, player_boxes, basket_boxes, fps) -> bool:
        """
        Stage 1 – possible: ball bottom rises above all player tops.
        Stage 2 – confirmed: stays elevated for ≥ CONFIRM_FRAMES frames.

        Returns True on the frame the shot ends (ball descends back down).
        """
        shot_ended = False

        # Always record basket position even outside a shot (builds fallback)
        self._update_basket_fallback(basket_boxes)

        if not ball_boxes or not player_boxes:
            # No detections: keep collecting if confirmed shot is in progress,
            # otherwise age out any possible-shot accumulation.
            if self.shot_in_progress:
                self._record_frame(ball_boxes, player_boxes, basket_boxes, fps)
            elif self.possible_shot:
                self.poss_frame_count += 1
                if self.poss_frame_count > 20:
                    self._cancel_possible()
            return False

        ball   = ball_boxes[0]
        b_bot  = ball[3]
        b_ctr  = (ball[1] + ball[3]) / 2
        p_top  = min(p[1] for p in player_boxes)

        above = b_bot < p_top     # ball fully above all player tops
        below = b_ctr > p_top     # ball centre back below player tops → shot ended

        # ── Stage 1: possible ────────────────────────────────────────────────
        if not self.possible_shot and not self.shot_in_progress and above:
            self.possible_shot    = True
            self.poss_frame_count = 0
            self.pre_data         = []
            self.shot_start_frame = self.frame_count
            print(f"\n  [possible shot – monitoring …]")

        if self.possible_shot and not self.shot_in_progress:
            self.poss_frame_count += 1
            self.pre_data.append(
                self._build_row(ball_boxes, player_boxes, basket_boxes, fps)
            )
            if above:
                if self.poss_frame_count >= CONFIRM_FRAMES:
                    # ── Stage 2: confirmed ───────────────────────────────────
                    self.shot_number      += 1
                    self.shot_in_progress  = True
                    self.possible_shot     = False
                    self.current_shot_data = self.pre_data[:]   # carry over pre-frames
                    self.pre_data          = []
                    print(f"  [SHOT #{self.shot_number} confirmed – "
                          f"{len(self.current_shot_data)} pre-frames carried over]")
            else:
                self._cancel_possible()

        # ── Record confirmed shot frames ──────────────────────────────────────
        elif self.shot_in_progress:
            self._record_frame(ball_boxes, player_boxes, basket_boxes, fps)

            if below:
                self.shot_in_progress = False
                dur = self.frame_count - self.shot_start_frame
                print(f"  [shot #{self.shot_number} ended – "
                      f"{len(self.current_shot_data)} frames, {dur} total]")
                shot_ended = True

        return shot_ended

    def _cancel_possible(self):
        print(f"  [possible shot cancelled – {self.poss_frame_count} frames]")
        self.possible_shot    = False
        self.poss_frame_count = 0
        self.pre_data         = []

    def _cancel_shot(self):
        print(f"  Shot #{self.shot_number} cancelled by user")
        self.shot_in_progress = False
        self.possible_shot    = False
        self.poss_frame_count = 0
        self.pre_data         = []
        self.current_shot_data = []

    # ── basket fallback ───────────────────────────────────────────────────────

    def _update_basket_fallback(self, basket_boxes):
        if basket_boxes:
            b = basket_boxes[0]
            self._last_basket = (
                (b[0]+b[2])/2, (b[1]+b[3])/2,
                b[2]-b[0],     b[3]-b[1],
            )

    # ── feature extraction ────────────────────────────────────────────────────

    def _build_row(self, ball_boxes, player_boxes, basket_boxes, fps) -> dict:
        """Build one data row dict from current detections (no label yet)."""
        frame_within_shot = self.frame_count - self.shot_start_frame
        shot_num          = self.shot_number if self.shot_in_progress else (self.shot_number + 1)

        # Ball
        if ball_boxes:
            ball   = ball_boxes[0]
            ball_x = (ball[0]+ball[2])/2;  ball_y = (ball[1]+ball[3])/2
            ball_w = ball[2]-ball[0];       ball_h = ball[3]-ball[1]
        else:
            ball_x = ball_y = ball_w = ball_h = None

        # Player closest to ball
        player_x = player_y = player_w = player_h = None
        player_top = player_bottom = None
        if player_boxes and ball_x is not None:
            cp = min(player_boxes,
                     key=lambda b: np.hypot((b[0]+b[2])/2 - ball_x,
                                            (b[1]+b[3])/2 - ball_y))
            player_x  = (cp[0]+cp[2])/2;  player_y = (cp[1]+cp[3])/2
            player_w  = cp[2]-cp[0];       player_h = cp[3]-cp[1]
            player_top    = cp[1]
            player_bottom = cp[3]

        # Basket with fallback ← KEY FIX
        if basket_boxes:
            b = basket_boxes[0]
            basket_x = (b[0]+b[2])/2;  basket_y = (b[1]+b[3])/2
            basket_w = b[2]-b[0];       basket_h = b[3]-b[1]
        elif self._last_basket:
            basket_x, basket_y, basket_w, basket_h = self._last_basket
        else:
            basket_x = basket_y = basket_w = basket_h = None

        # Derived features
        ball_player_dist = ball_basket_dist = ball_basket_angle = ball_above_player = None
        if ball_x is not None and player_x is not None:
            ball_player_dist  = float(np.hypot(ball_x - player_x, ball_y - player_y))
            ball_above_player = 1 if ball_y < player_top else 0
        if ball_x is not None and basket_x is not None:
            ball_basket_dist  = float(np.hypot(ball_x - basket_x, ball_y - basket_y))
            ball_basket_angle = float(np.arctan2(basket_y - ball_y, basket_x - ball_x))

        # Velocity from previous frame in whichever buffer is active
        ball_vx = ball_vy = None
        buffer  = self.current_shot_data or self.pre_data
        if buffer and ball_x is not None:
            prev = buffer[-1]
            if prev['ball_x'] is not None:
                dt = 1.0 / fps if fps > 0 else 1/30
                ball_vx = (ball_x - prev['ball_x']) / dt
                ball_vy = (ball_y - prev['ball_y']) / dt

        return {
            # metadata
            'frame_id':          f"{shot_num}.{frame_within_shot}",
            'shot_number':       shot_num,
            'frame_count':       self.frame_count,
            'frame_within_shot': frame_within_shot,
            'timestamp':         self.frame_count / fps,
            # features (lstm_train.py FEATURE_COLS)
            'ball_x':            ball_x,
            'ball_y':            ball_y,
            'ball_w':            ball_w,
            'ball_h':            ball_h,
            'ball_velocity_x':   ball_vx,
            'ball_velocity_y':   ball_vy,
            'player_x':          player_x,
            'player_y':          player_y,
            'player_w':          player_w,
            'player_h':          player_h,
            'basket_x':          basket_x,
            'basket_y':          basket_y,
            'basket_w':          basket_w,
            'basket_h':          basket_h,
            'ball_player_dist':  ball_player_dist,
            'ball_basket_dist':  ball_basket_dist,
            'ball_basket_angle': ball_basket_angle,
            'ball_above_player': ball_above_player,
            # extras (kept for debugging; not used by LSTM)
            'player_top':        player_top,
            'player_bottom':     player_bottom,
            'num_balls':         len(ball_boxes),
            'num_players':       len(player_boxes),
            'num_baskets':       len(basket_boxes),
        }

    def _record_frame(self, ball_boxes, player_boxes, basket_boxes, fps):
        self.current_shot_data.append(
            self._build_row(ball_boxes, player_boxes, basket_boxes, fps)
        )

    # ── labeling ─────────────────────────────────────────────────────────────

    def _save_shot(self, label: int, label_name: str):
        n = len(self.current_shot_data)
        for row in self.current_shot_data:
            row['shot_result'] = label
            self.labeled_shot_data.append(row)
        msg = (f"  ✅ Shot #{self.shot_number} → {label_name}  ({n} frames saved)")
        print(msg)
        self._last_shot_summary = msg
        self.current_shot_data  = []

    def _discard_shot(self, reason: str = ""):
        n   = len(self.current_shot_data)
        msg = f"  ⛔ Shot #{self.shot_number} discarded ({reason}, {n} frames)"
        print(msg)
        self._last_shot_summary = msg
        self.current_shot_data  = []

    # ── overlay ──────────────────────────────────────────────────────────────

    def _draw_overlay(self, frame, total_frames: int):
        status = (f"Frame {self.frame_count}/{total_frames}  "
                  f"Shot #{self.shot_number}  "
                  f"Labeled: {len(self.labeled_shot_data)} rows")

        if self.shot_in_progress:
            n = len(self.current_shot_data)
            cv2.rectangle(frame, (10, 10), (950, 52), (0, 200, 0), -1)
            cv2.putText(frame,
                        f"SHOT IN PROGRESS  ({n} frames) – X to cancel",
                        (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)
        elif self.possible_shot:
            cv2.rectangle(frame, (10, 10), (760, 52), (0, 165, 255), -1)
            cv2.putText(frame,
                        f"Possible shot … ({self.poss_frame_count}/{CONFIRM_FRAMES})",
                        (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)
        else:
            cv2.putText(frame, status, (16, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    # ── CSV save ──────────────────────────────────────────────────────────────

    def _save_final_data(self):
        if not self.labeled_shot_data:
            print("\nNo labeled shots – nothing saved.")
            return

        df_new = pd.DataFrame(self.labeled_shot_data)

        # Column ordering: meta + features + extras + label
        ordered = (self._META_COLS + self._FEATURE_COLS +
                   self._EXTRA_COLS + self._LABEL_COL)
        # Keep any unexpected columns at the end rather than silently dropping
        extra = [c for c in df_new.columns if c not in ordered]
        df_new = df_new[ordered + extra]

        # Append to existing CSV if present (multi-session support)
        out = Path(self.output_csv)
        if out.exists():
            df_existing = pd.read_csv(out)
            # Re-number shots so they don't collide
            offset = int(df_existing['shot_number'].max()) + 1
            df_new['shot_number'] += offset
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(out, index=False)
            print(f"\n  Appended {len(df_new)} rows to existing CSV.")
            df = df_combined
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            df_new.to_csv(out, index=False)
            df = df_new

        # ── shot-level validation report ──────────────────────────────────────
        shot_results = df.groupby('shot_number')['shot_result'].first()
        seq_lengths  = df.groupby('shot_number').size()

        print(f"\n{'=' * 50}")
        print("FINAL SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Total rows saved   : {len(df):,}")
        print(f"  Total shots        : {df['shot_number'].nunique()}")
        print(f"  Made shots         : {(shot_results == 1).sum()}")
        print(f"  Missed shots       : {(shot_results == 0).sum()}")
        print(f"\n  Sequence lengths:")
        print(f"    Min    : {seq_lengths.min()} frames")
        print(f"    Mean   : {seq_lengths.mean():.1f} frames")
        print(f"    Max    : {seq_lengths.max()} frames")
        short = (seq_lengths < MIN_SHOT_FRAMES).sum()
        if short:
            print(f"\n  ⚠️  {short} shots have < {MIN_SHOT_FRAMES} frames – "
                  f"consider re-labeling or raising MIN_SHOT_FRAMES.")

        # NaN coverage check for FEATURE_COLS present in df
        feat_in_df = [c for c in self._FEATURE_COLS if c in df.columns]
        nan_pct = df[feat_in_df].isna().mean() * 100
        bad = nan_pct[nan_pct > 10]
        if not bad.empty:
            print(f"\n  ⚠️  High NaN rate in features (>10%):")
            for col, pct in bad.items():
                print(f"      {col}: {pct:.1f}%")
        else:
            print(f"\n  ✅ All feature columns have ≤10% NaN (lstm_train fillna(0) is safe)")

        print(f"\n  Saved → {out}")

    # ── controls help ─────────────────────────────────────────────────────────

    @staticmethod
    def _print_controls():
        print("\n" + "=" * 40)
        print("CONTROLS")
        print("=" * 40)
        print("  During shot in progress:")
        print("    x  – cancel / discard shot")
        print("  When shot ends:")
        print("    1  – Made")
        print("    0  – Missed")
        print("    2  – Not a real shot (discard)")
        print("  Anytime:")
        print("    r  – replay last shot summary")
        print("    q  – quit and save")
        print("=" * 40 + "\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    detector = InteractiveShotDetector(MODEL_PATH, VIDEO_PATH, OUTPUT_CSV)
    detector.process_video()