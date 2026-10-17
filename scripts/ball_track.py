"""
ball_track.py  –  Clean and complete a shot's ball track (HPAI-19)
==================================================================
YOLO misses the ball often (small, blurred, behind the net) and sometimes
fires on something else (a head, a shoe). Two steps fix most of it:

1. clean_ball()  drops boxes whose size or shape doesn't fit the ball, and
                 isolated jumps that disagree with both neighbours.
2. fill_gaps()   fills short gaps with a parabola fitted to the points on both
                 sides (a ball in flight moves on a parabola: y quadratic in
                 time, x linear), instead of a straight line.

fit_parabola() is shared with the aim check, the made/missed judge and the
measurements.

Units: works on any consistent coordinates (pixels or basket widths); times
are frame indices.
"""

import numpy as np
import pandas as pd

SHAPE_RATIO = 1.8      # ball box w/h (or h/w) above this → not the ball (blur allows ~1.5)
SIZE_RANGE = (0.45, 2.2)   # ball width relative to the shot's median ball width
JUMP_FACTOR = 6.0      # a point more than this many ball widths from both neighbours (per frame gap) is an outlier
MAX_GAP_S = 1.0        # longest gap filled by the parabola (seconds) – LOVO 96.7% vs 92.9% (v2)
FIT_POINTS = 6         # points used on each side of a gap


def clean_ball(ball: pd.DataFrame) -> pd.Series:
    """Boolean mask of ball rows to keep. `ball` has ball_x, ball_y, ball_w, ball_h
    (NaN where not detected)."""
    seen = ball['ball_x'].notna()
    if seen.sum() < 3:
        return seen
    w, h = ball['ball_w'], ball['ball_h']
    keep = seen.copy()

    ratio = np.maximum(w / h, h / w)
    keep &= ~(ratio > SHAPE_RATIO)

    med_w = w[keep].median()
    if med_w and np.isfinite(med_w):
        rel = w / med_w
        keep &= rel.between(*SIZE_RANGE) | ~seen

    # isolated jumps: far from both the previous and the next kept point
    idx = np.flatnonzero(keep.to_numpy())
    x, y = ball['ball_x'].to_numpy(), ball['ball_y'].to_numpy()
    drop = []
    for k in range(1, len(idx) - 1):
        i, p, n = idx[k], idx[k - 1], idx[k + 1]
        lim_p = JUMP_FACTOR * med_w * (i - p)
        lim_n = JUMP_FACTOR * med_w * (n - i)
        far_p = np.hypot(x[i] - x[p], y[i] - y[p]) > lim_p
        far_n = np.hypot(x[i] - x[n], y[i] - y[n]) > lim_n
        near_pn = np.hypot(x[n] - x[p], y[n] - y[p]) <= JUMP_FACTOR * med_w * (n - p)
        if far_p and far_n and near_pn:
            drop.append(i)
    keep.iloc[drop] = False
    return keep


def fit_parabola(t, x, y):
    """Fit x(t) = a t + b and y(t) = c t² + d t + e. Returns (px, py) numpy
    polynomials or None if there are too few points."""
    t, x, y = map(np.asarray, (t, x, y))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    t, x, y = t[ok], x[ok], y[ok]
    deg_y = 2 if len(t) >= 4 else 1
    return np.poly1d(np.polyfit(t, x, 1)), np.poly1d(np.polyfit(t, y, deg_y))


def fill_gaps(ball: pd.DataFrame, keep: pd.Series, fps: float = 30.0) -> pd.DataFrame:
    """Return ball_x/ball_y/ball_w/ball_h with dropped rows removed and gaps of up
    to MAX_GAP_S filled from a parabola through the neighbouring points."""
    out = ball[['ball_x', 'ball_y', 'ball_w', 'ball_h']].where(keep).copy()
    valid = np.flatnonzero(keep.to_numpy())
    if len(valid) < 2:
        return out
    max_gap = max(2, int(round(MAX_GAP_S * fps)))
    for a, b in zip(valid[:-1], valid[1:]):
        gap = b - a - 1
        if gap <= 0 or gap > max_gap:
            continue
        left = valid[max(0, np.searchsorted(valid, a) - FIT_POINTS + 1): np.searchsorted(valid, a) + 1]
        right = valid[np.searchsorted(valid, b): np.searchsorted(valid, b) + FIT_POINTS]
        pts = np.concatenate([left, right])
        fit = fit_parabola(pts, out['ball_x'].to_numpy()[pts], out['ball_y'].to_numpy()[pts])
        fill_t = np.arange(a + 1, b)
        if fit is None:
            continue
        px, py = fit
        out.iloc[fill_t, out.columns.get_loc('ball_x')] = px(fill_t)
        out.iloc[fill_t, out.columns.get_loc('ball_y')] = py(fill_t)
        for col in ('ball_w', 'ball_h'):
            ci = out.columns.get_loc(col)
            out.iloc[fill_t, ci] = np.interp(fill_t, [a, b], [out.iloc[a, ci], out.iloc[b, ci]])
    return out
