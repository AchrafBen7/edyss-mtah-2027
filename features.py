"""Per-frame features for the learned possession model, from a loaded match.

Input is the dict that pff_loader.load_match (or the Metrica/SkillCorner loaders) returns:
ball_xy (n,2), home_xy (n,11,2), away_xy (n,11,2), ball_present (n,), period (n,), truth (n,), fps.

The features are deliberately football-sensible and work with a missing ball: ball-based columns
become NaN on the 41% of frames with no ball, and the geometry columns (team centroids, who is
near) stay defined, so a gradient-boosting model (which handles NaN natively) can still decide.
"""

import numpy as np


def _nan_inf(a):
    return np.where(np.isfinite(a), a, np.nan)


def _ball_speed(bx, by, period, fps):
    dbx = np.full(len(bx), np.nan)
    dby = np.full(len(bx), np.nan)
    dbx[1:] = bx[1:] - bx[:-1]
    dby[1:] = by[1:] - by[:-1]
    same = np.ones(len(bx), bool)
    same[1:] = period[1:] == period[:-1]
    sp = np.sqrt(dbx ** 2 + dby ** 2) * fps
    sp[~same] = np.nan
    return sp


def _ball_vel(bx, by, period, fps):
    """Signed ball velocity (m/s), reset at period seams."""
    n = len(bx)
    vx = np.full(n, np.nan); vy = np.full(n, np.nan)
    vx[1:] = (bx[1:] - bx[:-1]) * fps
    vy[1:] = (by[1:] - by[:-1]) * fps
    same = np.ones(n, bool); same[1:] = period[1:] == period[:-1]
    vx[~same] = np.nan; vy[~same] = np.nan
    return vx, vy


def _net_disp(bx, by, w):
    """Magnitude of the net ball displacement over a +/-w window (how far the ball travelled)."""
    n = len(bx)
    dx = np.full(n, np.nan); dy = np.full(n, np.nan)
    t = np.arange(w, n - w)
    dx[t] = bx[t + w] - bx[t - w]
    dy[t] = by[t + w] - by[t - w]
    return np.sqrt(dx ** 2 + dy ** 2)


def _roll(a, w):
    """Centred mean over +/-w (a has no NaN)."""
    cs = np.concatenate([[0.0], np.cumsum(a)])
    idx = np.arange(len(a)); lo = np.maximum(0, idx - w); hi = np.minimum(len(a), idx + w + 1)
    return (cs[hi] - cs[lo]) / (hi - lo)


def _wf(w, fps):
    """Window (frames) expressed in the real time it spans on PFF (29.97 fps), rescaled to this fps.
    So the model sees the SAME seconds at train (PFF) and inference (challenge 25 fps, SkillCorner 10 fps);
    at 29.97 fps it returns the original integer, keeping PFF features/held-out byte-identical."""
    return max(1, int(round(w * fps / 29.97)))


def _roll_nan(a, w):
    """Centred mean over +/-w ignoring NaN; NaN only where the whole window is NaN."""
    m = np.isfinite(a); av = np.where(m, a, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(av)])
    cc = np.concatenate([[0.0], np.cumsum(m.astype(float))])
    idx = np.arange(len(a)); lo = np.maximum(0, idx - w); hi = np.minimum(len(a), idx + w + 1)
    s = cs[hi] - cs[lo]; c = cc[hi] - cc[lo]
    return np.where(c > 0, s / np.maximum(c, 1.0), np.nan)


def _player_vel(xy, period, fps, w=3):
    """Per-player velocity (n,K,2) via a centred difference over +/-w frames; NaN at seams/edges."""
    n = xy.shape[0]
    v = np.full_like(xy, np.nan)
    if n > 2 * w:
        v[w:n - w] = (xy[2 * w:] - xy[:n - 2 * w]) * (fps / (2.0 * w))
    for c in np.where(period[1:] != period[:-1])[0] + 1:
        v[max(0, c - w):min(n, c + w)] = np.nan
    return v


def frame_features(m, extra=False, pitch_control=False, persist=False):
    """(X, y, graded_mask, names). X is (n_frames, n_feats); y is the H/A/D/X truth.

    extra=True appends motion/momentum/local-control features (v2); default keeps the v1 set.
    pitch_control=True additionally appends velocity-aware space-control features (requires extra).
    persist=True appends possession-persistence features (last_poss/streak/obs_roll75; requires extra).
    """
    bx, by = m["ball_xy"][:, 0], m["ball_xy"][:, 1]
    home, away = m["home_xy"], m["away_xy"]

    with np.errstate(invalid="ignore", all="ignore"):
        dh = np.sqrt((home[:, :, 0] - bx[:, None]) ** 2 + (home[:, :, 1] - by[:, None]) ** 2)
        da = np.sqrt((away[:, :, 0] - bx[:, None]) ** 2 + (away[:, :, 1] - by[:, None]) ** 2)
        dh_s = np.sort(np.where(np.isnan(dh), np.inf, dh), axis=1)
        da_s = np.sort(np.where(np.isnan(da), np.inf, da), axis=1)
        dh1, dh2 = _nan_inf(dh_s[:, 0]), _nan_inf(dh_s[:, 1])
        da1, da2 = _nan_inf(da_s[:, 0]), _nan_inf(da_s[:, 1])
        n_home_near = np.sum(np.where(np.isnan(dh), 1e9, dh) < 5.0, axis=1).astype(float)
        n_away_near = np.sum(np.where(np.isnan(da), 1e9, da) < 5.0, axis=1).astype(float)
        home_cx = np.nanmean(home[:, :, 0], axis=1)
        away_cx = np.nanmean(away[:, :, 0], axis=1)
        home_cy = np.nanmean(home[:, :, 1], axis=1)
        away_cy = np.nanmean(away[:, :, 1], axis=1)
        home_spread = np.nanstd(home[:, :, 0], axis=1) + np.nanstd(home[:, :, 1], axis=1)
        away_spread = np.nanstd(away[:, :, 0], axis=1) + np.nanstd(away[:, :, 1], axis=1)

    gap = da1 - dh1                                    # >0: home closer to the ball
    conf = np.abs(dh1 - da1) / np.maximum(dh1 + da1, 1e-9)
    speed = _ball_speed(bx, by, m["period"], m["fps"])

    feats = {
        "bx": bx, "by": by,
        "dh1": dh1, "da1": da1, "dh2": dh2, "da2": da2,
        "gap": gap, "conf": conf,
        "n_home_near": n_home_near, "n_away_near": n_away_near,
        "ball_to_home_cx": bx - home_cx, "ball_to_away_cx": bx - away_cx,
        "ball_to_home_cy": by - home_cy, "ball_to_away_cy": by - away_cy,
        "home_minus_away_cx": home_cx - away_cx,
        "home_spread": home_spread, "away_spread": away_spread,
        "ball_speed": speed,
        "present": m["ball_present"].astype(float),
    }

    if extra:
        n = len(bx)
        present = m["ball_present"]
        fps = m["fps"]
        vx, vy = _ball_vel(bx, by, m["period"], fps)
        speed_sm = _roll_nan(speed, _wf(10, fps))       # denoised ball speed (fps-aware)
        disp30 = _net_disp(bx, by, _wf(15, fps))        # net travel over ~0.5 s (fps-aware)
        with np.errstate(invalid="ignore", all="ignore"):
            min_d1 = np.fmin(dh1, da1)                  # ball looseness (nearest of either team)
            all_d = np.concatenate([np.where(np.isnan(dh), np.inf, dh),
                                    np.where(np.isnan(da), np.inf, da)], axis=1)
            all_s = np.sort(all_d, axis=1)
            ctrl_margin = _nan_inf(all_s[:, 1]) - _nan_inf(all_s[:, 0])   # 2nd - 1st nearest overall
            n2_home = np.sum(np.where(np.isnan(dh), 1e9, dh) < 2.0, axis=1).astype(float)
            n2_away = np.sum(np.where(np.isnan(da), 1e9, da) < 2.0, axis=1).astype(float)
            n10_home = np.sum(np.where(np.isnan(dh), 1e9, dh) < 10.0, axis=1).astype(float)
            n10_away = np.sum(np.where(np.isnan(da), 1e9, da) < 10.0, axis=1).astype(float)
        sign = np.zeros(n)                              # +1 home nearest, -1 away nearest, 0 absent
        valid = np.isfinite(dh1) & np.isfinite(da1) & present
        sign[valid] = np.where(dh1[valid] <= da1[valid], 1.0, -1.0)
        obs_roll25 = _roll(sign, _wf(25, m["fps"]))     # possession momentum ~+/-0.83 s (fps-aware)
        feats.update({
            "vx": vx, "vy": vy, "speed_sm": speed_sm, "disp30": disp30,
            "min_d1": min_d1, "ctrl_margin": ctrl_margin,
            "n2_home": n2_home, "n2_away": n2_away,
            "n10_home": n10_home, "n10_away": n10_away,
            "obs_roll25": obs_roll25,
        })

        if persist:                                     # possession-persistence (encodes the label's inertia)
            cf = feats["conf"]; per = m["period"]
            sign_conf = np.where((sign != 0) & (cf > 0.2), sign, 0.0)
            last_poss = np.empty(n); cur = 0.0
            for i in range(n):
                if i and per[i] != per[i - 1]:
                    cur = 0.0
                if sign_conf[i] != 0.0:
                    cur = sign_conf[i]
                last_poss[i] = cur
            streak = np.zeros(n)
            for i in range(1, n):
                streak[i] = streak[i - 1] + 1.0 if (sign[i] == sign[i - 1] and sign[i] != 0) else 0.0
            feats.update({"last_poss": last_poss,
                          "streak": np.tanh(streak / max(1.0, _wf(25, m["fps"]))),
                          "obs_roll75": _roll(sign, _wf(75, m["fps"]))})

    if pitch_control:
        fps = m["fps"]
        hv = _player_vel(home, m["period"], fps)
        av = _player_vel(away, m["period"], fps)
        TR, VMAX, TAU = 0.7, 5.0, 0.4          # reaction s, max controllable speed m/s, logistic scale
        with np.errstate(invalid="ignore", all="ignore"):
            def _tti(grp, gv):                 # time for each player to reach the ball (n,K)
                pr0 = grp[:, :, 0] + np.nan_to_num(gv[:, :, 0]) * TR
                pr1 = grp[:, :, 1] + np.nan_to_num(gv[:, :, 1]) * TR
                d = np.sqrt((pr0 - bx[:, None]) ** 2 + (pr1 - by[:, None]) ** 2)
                return np.where(np.isnan(grp[:, :, 0]), np.inf, TR + d / VMAX)

            th, ta = _tti(home, hv), _tti(away, av)
            tti_home = _nan_inf(np.min(th, axis=1))       # inf -> nan where the team has no player
            tti_away = _nan_inf(np.min(ta, axis=1))
            tti_gap = tti_away - tti_home                 # >0 : home reaches the ball first
            pc_ball = 1.0 / (1.0 + np.exp(-tti_gap / TAU))   # P(home controls the ball)
            reach15_home = np.sum(th < 1.5, axis=1).astype(float)
            reach15_away = np.sum(ta < 1.5, axis=1).astype(float)

            def _closing(grp, gv):             # best closing speed toward the ball (n,)
                dx = bx[:, None] - grp[:, :, 0]; dy = by[:, None] - grp[:, :, 1]
                nrm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
                cs = (np.nan_to_num(gv[:, :, 0]) * dx + np.nan_to_num(gv[:, :, 1]) * dy) / nrm
                cs = np.where(np.isnan(grp[:, :, 0]), -np.inf, cs)
                return _nan_inf(np.max(cs, axis=1))

            approach_home = _closing(home, hv)
            approach_away = _closing(away, av)
        feats.update({
            "tti_home": tti_home, "tti_away": tti_away, "tti_gap": tti_gap, "pc_ball": pc_ball,
            "reach15_home": reach15_home, "reach15_away": reach15_away,
            "approach_home": approach_home, "approach_away": approach_away,
        })

    names = list(feats)
    X = np.column_stack([feats[k] for k in names])
    y = m["truth"]
    graded = y != 3                                    # X = 3 is ungraded
    return X, y, graded, names


if __name__ == "__main__":
    import sys
    import pff_loader
    m = pff_loader.load_match(sys.argv[1] if len(sys.argv) > 1 else "10506")
    X, y, graded, names = frame_features(m)
    print("features: {}  ({} colonnes)".format(X.shape, len(names)))
    print("colonnes:", names)
    g = graded
    print("images gradees:", int(g.sum()), "sur", len(y))
    print("labels gradees  H/A/D:", [int(np.sum(y[g] == k)) for k in (0, 1, 2)])
    nan_frac = np.isnan(X[g]).mean(axis=0)
    print("colonnes 100% NaN (a corriger):", [names[i] for i in range(len(names)) if nan_frac[i] > 0.999])
    print("fraction NaN par colonne:", {names[i]: round(float(nan_frac[i]), 2) for i in range(len(names))})
