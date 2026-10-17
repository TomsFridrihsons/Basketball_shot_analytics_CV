"""
shot_features.py  –  Shared shot-sequence features (training + inference)
=========================================================================
The labeler and lstm_infer.py record 18 raw per-frame columns (RAW_COLS, pixel
units). This module turns one shot's raw frames into the model input, so
training and inference build features with exactly the same code.

Feature version 2 (FEATURE_COLS):
  - positions are relative to the basket and divided by the basket width,
    so the model does not depend on camera position, zoom or resolution
  - short gaps in the ball track (≤ MAX_GAP frames) are interpolated instead
    of being filled with 0, plus a ball_detected flag
  - velocity is a smoothed frame-to-frame difference in basket widths / frame
  - sequences keep the LAST max_len frames (where the ball reaches the rim)
"""

import numpy as np
import pandas as pd

import ball_track

# Raw per-frame columns written by shot_labeler.py and lstm_infer.py
RAW_COLS = [
    'ball_x', 'ball_y', 'ball_w', 'ball_h',
    'ball_velocity_x', 'ball_velocity_y',
    'player_x', 'player_y', 'player_w', 'player_h',
    'basket_x', 'basket_y', 'basket_w', 'basket_h',
    'ball_player_dist', 'ball_basket_dist', 'ball_basket_angle',
    'ball_above_player',
]

# 2: original; 3: ball track cleaned + gaps filled with a parabola (ball_track.py, HPAI-19).
# Saved models record the version they were trained with; inference passes it back.
FEATURE_VERSION = 3
FEATURE_COLS = [
    'ball_dx', 'ball_dy', 'ball_w', 'ball_h',
    'ball_vx', 'ball_vy',
    'player_dx', 'player_dy', 'player_w', 'player_h',
    'basket_h',
    'ball_basket_dist', 'ball_player_dist',
    'ball_above_player', 'ball_detected',
    'rim_zone', 'rim_offset',
]
# Columns whose sign flips when the shot is mirrored left ↔ right
MIRROR_COLS = [FEATURE_COLS.index(c) for c in ('ball_dx', 'ball_vx', 'player_dx')]

MAX_LEN = 64        # frames fed to the LSTM (tail of the shot)
MAX_GAP = 5         # longest ball-track gap that gets interpolated

# Rim-pass geometry, in basket widths relative to the basket centre (y grows down)
RIM_ABOVE = 0.5          # ball counts as "above the rim" when dy < -RIM_ABOVE
RIM_BAND  = (0.2, 2.0)   # "dropping through rim height": -0.2 < dy < 2.0
RIM_FAR   = 3.0          # rim_offset outside the rim zone / clip value

# Shot detection: frames the ball must stay above the players to confirm a shot.
# Shared by shot_labeler.py and lstm_infer.py so both see the same shots.
# Timings are in seconds so 30 and 60 fps videos behave the same (HPAI-19);
# frames(seconds, fps) converts. The *_FRAMES names are the 30 fps equivalents.
CONFIRM_S = 23 / 30          # ball above the players this long → confirmed shot
POSSIBLE_TIMEOUT_S = 25 / 30  # possible shot dropped after this long without detections
CONFIRM_FRAMES = 23
# A confirmed shot ends when the ball has been missing this long (0.5 s at 30 fps):
# after the rim it is often lost in the net or blurred and never seen coming down.
SHOT_LOST_S = 0.5
SHOT_LOST_FRAMES = 15


def frames(seconds: float, fps: float) -> int:
    """Seconds → whole frames at this video's frame rate (at least 1)."""
    return max(1, int(round(seconds * (fps or 30.0))))


def shot_to_features(raw, version: int = FEATURE_VERSION, fps: float = 30.0) -> np.ndarray:
    """(T, len(RAW_COLS)) raw frames → (T, len(FEATURE_COLS)) float32 features."""
    df = pd.DataFrame(np.asarray(raw, dtype=np.float64), columns=RAW_COLS)

    # A ball/player is "missing" when its box is NaN or was zero-filled
    ball_seen = df['ball_x'].notna() & (df['ball_x'] != 0)
    ball = df[['ball_x', 'ball_y', 'ball_w', 'ball_h']].where(ball_seen)
    if version >= 3:
        keep = ball_track.clean_ball(ball)          # drop non-ball boxes and jumps
        ball_seen = keep
        ball = ball_track.fill_gaps(ball, keep, fps)  # parabola through short gaps
    else:
        ball = ball.interpolate(limit=MAX_GAP, limit_area='inside')

    player_seen = df['player_x'].notna() & (df['player_x'] != 0)
    player = df[['player_x', 'player_y', 'player_w', 'player_h']].where(player_seen).ffill().bfill()

    basket_seen = df['basket_w'].notna() & (df['basket_w'] > 0)
    basket = df[['basket_x', 'basket_y', 'basket_w', 'basket_h']].where(basket_seen).ffill().bfill()

    # Scale: basket width (stable per shot); fall back to ~2 ball widths
    if basket_seen.any():
        scale = float(basket['basket_w'].median())
    elif ball_seen.any():
        scale = 2.0 * float(ball['ball_w'].median())
    else:
        scale = 1.0
    scale = max(scale, 1e-6)

    bx0, by0 = basket['basket_x'], basket['basket_y']
    ball_dx = (ball['ball_x'] - bx0) / scale
    ball_dy = (ball['ball_y'] - by0) / scale
    player_dx = (player['player_x'] - bx0) / scale
    player_dy = (player['player_y'] - by0) / scale

    def smooth_diff(s):
        return s.diff().rolling(3, center=True, min_periods=1).mean()

    # Rim pass: after the ball has been above the rim, the frames where it drops
    # back through rim height. A make passes close to the rim centre there
    # (small |ball_dx|); on its own this separates made/missed ~93% across videos.
    seen = ball_dx.notna()
    was_above = (seen & (ball_dy < -RIM_ABOVE)).cummax()
    rim_zone = was_above & seen & (ball_dy > -RIM_BAND[0]) & (ball_dy < RIM_BAND[1])
    rim_offset = ball_dx.abs().clip(upper=RIM_FAR).where(rim_zone, RIM_FAR)

    out = pd.DataFrame({
        'ball_dx':           ball_dx,
        'ball_dy':           ball_dy,
        'ball_w':            ball['ball_w'] / scale,
        'ball_h':            ball['ball_h'] / scale,
        'ball_vx':           smooth_diff(ball_dx),
        'ball_vy':           smooth_diff(ball_dy),
        'player_dx':         player_dx,
        'player_dy':         player_dy,
        'player_w':          player['player_w'] / scale,
        'player_h':          player['player_h'] / scale,
        'basket_h':          basket['basket_h'] / scale,
        'ball_basket_dist':  np.hypot(ball_dx, ball_dy),
        'ball_player_dist':  np.hypot(ball_dx - player_dx, ball_dy - player_dy),
        'ball_above_player': df['ball_above_player'],
        'ball_detected':     ball_seen.astype(float),
        'rim_zone':          rim_zone.astype(float),
        'rim_offset':        rim_offset,
    })
    return out[FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)


def rim_pass_offset(feats: np.ndarray) -> float:
    """Closest the ball came to the rim centre (basket widths) while dropping
    through rim height; RIM_FAR if it never did. Small → made."""
    return float(feats[:, FEATURE_COLS.index('rim_offset')].min()) if len(feats) else RIM_FAR


SUMMARY_COLS = [
    'rim_offset_min', 'rim_offset_log', 'rim_zone_frames', 'rim_zone_any',
    'ball_basket_dist_min', 'ball_detected_tail',
]


def shot_summary(feats: np.ndarray) -> np.ndarray:
    """Shot-level geometry vector (len(SUMMARY_COLS)) from a feature sequence.

    Mirroring or small noise on `feats` changes it consistently, so it can be
    recomputed from augmented sequences.
    """
    col = FEATURE_COLS.index
    zone = feats[:, col('rim_zone')] > 0.5
    detected = feats[:, col('ball_detected')]
    dist = feats[detected > 0.5, col('ball_basket_dist')]
    offset = rim_pass_offset(feats)
    return np.array([
        offset,
        np.log(offset + 0.05),
        min(zone.sum(), 20) / 10.0,
        float(zone.any()),
        float(dist.min()) if len(dist) else RIM_FAR,
        float(detected[-15:].mean()) if len(detected) else 0.0,
    ], dtype=np.float32)


def mirror(seq: np.ndarray) -> np.ndarray:
    """Left ↔ right mirrored copy of a feature sequence (valid augmentation)."""
    out = seq.copy()
    out[:, MIRROR_COLS] *= -1
    return out


def pad_tail(seqs, max_len: int = MAX_LEN) -> np.ndarray:
    """Keep the last max_len frames of each sequence; zero-pad at the end.

    Zero rows are masked by the model, so call this AFTER scaling.
    """
    n_feat = seqs[0].shape[1]
    out = np.zeros((len(seqs), max_len, n_feat), dtype=np.float32)
    for i, seq in enumerate(seqs):
        tail = seq[-max_len:]
        out[i, :len(tail)] = tail
    return out
