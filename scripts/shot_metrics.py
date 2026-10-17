"""
shot_metrics.py  –  Physical measurements of a shot (HPAI-22)
=============================================================
Scale: the basket box is ~0.46 m wide (rim inner diameter 45.7 cm), so one basket
width ≈ 0.46 m at the hoop's distance from the camera. A parabola fitted to the
ball's flight (release → rim height) gives:

  release_speed_ms   ball speed just after release (m/s)
  release_angle_deg  launch angle above horizontal
  arc_height_m       apex height above the rim
  entry_angle_deg    descent angle when the ball reaches rim height (~45° is ideal)
  gravity_ratio      fitted downward acceleration / 9.81 – a quality check:
                     far from 1 means the 2D estimate is unreliable (camera not
                     side-on, ball far from the hoop's depth, bad track)

These are 2D estimates from one camera; they are reliable when the camera looks
at the shot roughly side-on. `reliable` is False when gravity_ratio is off.
"""

import numpy as np

import shot_features as sf
import shot_judge

BASKET_M = shot_judge.BASKET_M
GRAVITY_OK = (0.6, 1.6)          # gravity_ratio range considered reliable


def measure(feats: np.ndarray, fps: float = 30.0) -> dict | None:
    """Metrics for one shot from its features (v3), or None if the flight can't be fitted."""
    C = sf.FEATURE_COLS.index
    dx, dy = feats[:, C('ball_dx')], feats[:, C('ball_dy')]
    det = feats[:, C('ball_detected')] > 0.5
    above_player = feats[:, C('ball_above_player')] > 0.5
    valid = (dx != 0) | (dy != 0)
    idx = np.flatnonzero(valid)
    if len(idx) < 6:
        return None

    # flight: from the first frame above the shooter to the first frame at rim height after the apex
    start_c = idx[above_player[idx]]
    start = start_c[0] if len(start_c) else idx[0]
    apex = idx[idx >= start][np.argmin(dy[idx[idx >= start]])]
    if dy[apex] >= 0:
        return None
    after = idx[(idx > apex) & (dy[idx] >= 0)]
    end = after[0] if len(after) else idx[-1]
    seg = idx[(idx >= start) & (idx <= end) & det[idx]]     # detected points only
    if len(seg) < 5:
        seg = idx[(idx >= start) & (idx <= end)]
    if len(seg) < 5:
        return None

    t = (seg - seg[0]) / fps
    px = np.polyfit(t, dx[seg], 1)            # dx = a t + b
    py = np.polyfit(t, dy[seg], 2)            # dy = c t² + d t + e (image y downwards)
    vx, vy0, c = px[0], py[1], py[0]
    g_fit = 2 * c * BASKET_M                  # m/s² downwards
    gravity_ratio = g_fit / 9.81

    m = BASKET_M
    release_speed = float(np.hypot(vx, vy0) * m)
    release_angle = float(np.degrees(np.arctan2(-vy0, abs(vx))))
    arc_height = float(-dy[apex] * m)

    # entry: velocity where the fitted curve reaches rim height on the way down
    roots = [r.real for r in np.roots([c, py[1], py[2]]) if abs(r.imag) < 1e-9 and r.real > -py[1] / (2 * c + 1e-12)]
    entry_angle = None
    if c > 0 and roots:
        tc = min(roots)
        vy_c = 2 * c * tc + py[1]
        entry_angle = float(np.degrees(np.arctan2(vy_c, abs(vx))))

    return {
        'release_speed_ms': round(release_speed, 1),
        'release_angle_deg': round(release_angle, 0),
        'arc_height_m': round(arc_height, 2),
        'entry_angle_deg': None if entry_angle is None else round(entry_angle, 0),
        'gravity_ratio': round(float(gravity_ratio), 2),
        'reliable': bool(GRAVITY_OK[0] <= gravity_ratio <= GRAVITY_OK[1]),
    }
