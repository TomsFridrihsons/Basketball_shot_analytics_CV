"""
shot_judge.py  –  Made / missed from the ball's trajectory (HPAI-21)
===================================================================
Builds on the rim-pass rule (features v3) with physics:

  rim offset      closest the ball comes to the rim centre while dropping through
                  rim height (basket widths) – as the rim-pass rule.
  crossing offset where a parabola fitted to the descent crosses rim height
                  (uses several points instead of the single closest one).
  fall ratio      ball hidden by the net: a ball that goes through the net is
                  slowed down by it; a ball that drops in front of / behind the
                  rim falls freely. ratio = free-fall time / observed time for the
                  drop from rim height to 1 basket width below (≈1 free fall,
                  ≈0.2–0.7 through the net). Timed on a fitted curve, not by
                  counting frames, so 30 fps works.
  bounce          the ball came down to the rim and rose again (rim bounce);
                  reported, and used by the measurement code.

decide(): made if offset < offset_threshold and the fall is slower than free
fall (ratio < ratio_threshold, when it can be measured). Thresholds are fitted
on training data (lstm_train.py) and stored in the judge config.

Labeled data (184 shots, leave-one-video-out): see progress_context.md.
"""

import numpy as np

import shot_features as sf

BASKET_M = 0.46       # basket box width in metres (rim inner diameter 45.7 cm)
G = 9.81              # m/s²
BELOW_DY = 1.0        # timing window: rim height → this many basket widths below
NEAR_RIM = dict(dx=0.8, dy_min=0.2, dy_max=2.0)   # "under the hoop" region


def _track(feats: np.ndarray):
    C = sf.FEATURE_COLS.index
    dx, dy = feats[:, C('ball_dx')], feats[:, C('ball_dy')]
    valid = (dx != 0) | (dy != 0)
    return dx, dy, valid


def fall_ratio(feats: np.ndarray, fps: float = 30.0):
    """(ratio, observed seconds) for the drop from rim height to BELOW_DY under the
    hoop, or (nan, nan) when the ball isn't seen dropping under the hoop."""
    dx, dy, valid = _track(feats)
    idx = np.flatnonzero(valid)
    under = valid & (np.abs(dx) < NEAR_RIM['dx']) & (dy > NEAR_RIM['dy_min']) & (dy < NEAR_RIM['dy_max'])
    if len(idx) < 4 or not under.any():
        return np.nan, np.nan
    first_under = int(np.argmax(under))
    before = idx[idx < first_under]
    if not len(before):
        return np.nan, np.nan
    apex = before[np.argmin(dy[before])]
    if dy[apex] >= 0:
        return np.nan, np.nan

    # fit dy(t) on the descent from the last apex to the end of the under-hoop window
    seg = idx[(idx >= apex) & (dy[idx] <= BELOW_DY + 1.0)]
    seg = seg[seg <= max(np.flatnonzero(under)) + 2]
    if len(seg) < 4:
        return np.nan, np.nan
    py = np.poly1d(np.polyfit(seg, dy[seg], 2))

    def first_time(level):
        roots = (py - level).roots
        roots = [r.real for r in roots if abs(r.imag) < 1e-9 and seg[0] - 2 <= r.real <= seg[-1] + 2]
        return min(roots) if roots else None

    t0, t1 = first_time(0.0), first_time(BELOW_DY)
    if t0 is None or t1 is None or t1 <= t0:
        return np.nan, np.nan
    t_obs = (t1 - t0) / fps

    g = G / BASKET_M                               # basket widths / s²
    h0 = -dy[apex]                                 # drop from apex to rim height
    t_ff = np.sqrt(2 * (h0 + BELOW_DY) / g) - np.sqrt(2 * h0 / g)
    return float(t_ff / t_obs), float(t_obs)


def crossing_offset(feats: np.ndarray) -> float:
    """|dx| where a parabola through the descent crosses rim height; RIM_FAR if none."""
    dx, dy, valid = _track(feats)
    idx = np.flatnonzero(valid)
    above = idx[dy[idx] < -sf.RIM_ABOVE]
    if not len(above):
        return sf.RIM_FAR
    after = idx[idx > above[0]]
    down = [i for k, i in enumerate(after[1:], 1) if dy[after[k - 1]] < 0 <= dy[i]]
    if not down:
        return sf.RIM_FAR
    c = down[0]
    win = idx[(idx >= c - 6) & (idx <= c + 3)]
    if len(win) < 4:
        return float(min(abs(dx[c]), sf.RIM_FAR))
    px = np.poly1d(np.polyfit(win, dx[win], 1))
    py = np.poly1d(np.polyfit(win, dy[win], 2))
    roots = [r.real for r in py.roots if abs(r.imag) < 1e-9 and win[0] - 2 <= r.real <= win[-1] + 2]
    if not roots:
        return float(min(abs(dx[c]), sf.RIM_FAR))
    t = min(roots, key=lambda r: abs(r - c))
    return float(min(abs(px(t)), sf.RIM_FAR))


def bounced(feats: np.ndarray, rise: float = 0.25) -> bool:
    """Ball came down near the rim and went back up by `rise` basket widths."""
    dx, dy, valid = _track(feats)
    lowest = None
    for i in np.flatnonzero(valid):
        if abs(dx[i]) < 1.2 and -1.4 < dy[i] < 0.2:
            lowest = dy[i] if lowest is None else max(lowest, dy[i])
        if lowest is not None and dy[i] < lowest - rise:
            return True
        if dy[i] > 0.3:
            break
    return False


def signals(feats: np.ndarray, fps: float = 30.0) -> dict:
    ratio, t_obs = fall_ratio(feats, fps)
    return {
        'rim_offset': sf.rim_pass_offset(feats),
        'crossing_offset': crossing_offset(feats),
        'fall_ratio': ratio,
        'fall_time_s': t_obs,
        'bounce': bounced(feats),
    }


def decide(sig: dict, params: dict) -> tuple[int, float]:
    """(label, score) – label 1 made / 0 missed; score in 0..1 (0.5 at the boundary)."""
    off = sig[params.get('offset', 'rim_offset')]
    t_off = params['offset_threshold']
    made = off < t_off
    ratio = sig['fall_ratio']
    t_ratio = params.get('ratio_threshold')
    if made and t_ratio is not None and np.isfinite(ratio) and ratio > t_ratio:
        made = False            # dropped at free-fall speed: in front of / behind the net
    score = 1.0 / (1.0 + np.exp((off - t_off) / 0.1))
    if not made:
        score = min(score, 0.49)
    elif score < 0.5:
        score = 0.51
    return int(made), float(score)
