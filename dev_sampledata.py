"""Dev / tuning loop on Metrica's open sample-data (a LEGAL held-out set).

The rules forbid tuning on the public 5 minutes, and 17 possessions cannot rank anything.
So we build a possession truth from the sample games' event logs, run our pipeline on their
tracking, and score with the official metric. Every parameter decision happens HERE.

    python dev_sampledata.py                       # score baseline vs ours on games 1 and 2
    python dev_sampledata.py --sweep               # grid-search the parameters

Needs the sample-data checkout (https://github.com/metrica-sports/sample-data); point --data
at it. The ball is almost always tracked there, unlike the real challenge (41% missing), so we
can simulate that with --drop-ball to make the dev set match the target regime.
"""

import argparse
import os
import sys
import itertools

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "evaluation"))
import possession_metrics as pm            # noqa: E402
import possession as P                      # noqa: E402
import data_paths                           # noqa: E402

PITCH = (105.0, 68.0)
STOP_TYPES = {"BALL OUT", "FAULT RECEIVED", "CARD"}


def load_tracking(home_csv, away_csv):
    """(n_frames, ball_xy, home_xy, away_xy) from the two Metrica tracking files."""
    def read(path):
        df = pd.read_csv(path, skiprows=3, header=None)
        df = df.dropna(axis=1, how="all")           # trailing empty column
        frame = df.iloc[:, 1].astype(int).values
        pos = df.iloc[:, 3:].astype(float).values   # x,y pairs, ball is the last pair
        return frame, pos
    fh, ph = read(home_csv)
    fa, pa = read(away_csv)
    n = int(max(fh.max(), fa.max()))
    ball = np.full((n, 2), np.nan)
    ball[fh - 1] = ph[:, -2:]                        # ball columns (home file)
    nh = (ph.shape[1] - 2) // 2
    na = (pa.shape[1] - 2) // 2
    home = np.full((n, nh, 2), np.nan)
    away = np.full((n, na, 2), np.nan)
    home[fh - 1] = ph[:, :-2].reshape(len(fh), nh, 2)
    away[fa - 1] = pa[:, :-2].reshape(len(fa), na, 2)
    return n, ball, home, away


def tracking_features(ball, home, away, pitch, drop_ball=0.0, seed=0):
    """team_obs (0 home / 1 away / -1), ball_present, and control confidence per frame."""
    length, width = pitch
    bx, by = ball[:, 0], ball[:, 1]
    present = ~np.isnan(bx)
    if drop_ball > 0:
        rng = np.random.RandomState(seed)
        present = present & (rng.rand(len(bx)) > drop_ball)

    def nearest(group):
        dx = (group[:, :, 0] - bx[:, None]) * length
        dy = (group[:, :, 1] - by[:, None]) * width
        d = dx * dx + dy * dy
        with np.errstate(invalid="ignore"):
            return np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    min_h = nearest(home)
    min_a = nearest(away)
    team_obs = np.full(len(bx), -1, dtype=np.int8)
    conf = np.zeros(len(bx))
    valid = present & np.isfinite(min_h) & np.isfinite(min_a)
    team_obs[valid] = np.where(min_h[valid] <= min_a[valid], 0, 1)
    dh, da = np.sqrt(min_h[valid]), np.sqrt(min_a[valid])
    conf[valid] = np.abs(dh - da) / np.maximum(dh + da, 1e-9)
    return team_obs, present & valid, conf


def truth_from_events(events_csv, n_frames):
    """Per-frame H/A/D/X truth and the graded blocks, from the event log (per period)."""
    ev = pd.read_csv(events_csv)
    truth = np.full(n_frames, P.X, dtype=np.int8)
    blocks = []
    for period in sorted(ev["Period"].dropna().unique()):
        evp = ev[ev["Period"] == period].sort_values("Start Frame")
        rows = []
        for _, r in evp.iterrows():
            rows.append((int(r["Start Frame"]),
                         str(r["Type"]).strip().upper(),
                         0 if str(r["Team"]).strip().lower() == "home" else 1))
        if not rows:
            continue
        last_frame = int(evp["End Frame"].max())
        team, alive = None, False
        for i, (sf, typ, tm) in enumerate(rows):
            nf = rows[i + 1][0] if i + 1 < len(rows) else last_frame + 1
            if typ in STOP_TYPES:
                alive = False
            else:
                alive, team = True, tm
            if team is not None:
                truth[sf - 1:nf - 1] = (team if alive else P.D)
        blocks.append((rows[0][0] - 1, last_frame))
    return truth, blocks


def dead_prior_from_events(events_csv, n_frames, lead=50):
    """Dead-ball prior that mimics Smart Tagging: a window just before each set-piece restart."""
    ev = pd.read_csv(events_csv)
    prior = np.zeros(n_frames, dtype=bool)
    for _, r in ev[ev["Type"].astype(str).str.upper() == "SET PIECE"].iterrows():
        sf = int(r["Start Frame"])
        prior[max(0, sf - 1 - lead):max(0, sf - 1)] = True
    return prior


def score_game(game_dir, params, drop_ball=0.41, bootstrap=0):
    name = os.path.basename(game_dir)
    home = os.path.join(game_dir, name + "_RawTrackingData_Home_Team.csv")
    away = os.path.join(game_dir, name + "_RawTrackingData_Away_Team.csv")
    events = os.path.join(game_dir, name + "_RawEventsData.csv")

    n, ball, home_xy, away_xy = load_tracking(home, away)
    truth, blocks = truth_from_events(events, n)
    team_obs, present, conf = tracking_features(ball, home_xy, away_xy, PITCH, drop_ball=drop_ball)
    dead_prior = dead_prior_from_events(events, n)

    base = P.smoothed_baseline(team_obs, present, blocks, n)
    ours = P.build_labels(team_obs, present, dead_prior, blocks, n, params, conf)
    br = pm.evaluate(truth, base, bootstrap_n=bootstrap)
    orr = pm.evaluate(truth, ours, bootstrap_n=bootstrap)
    return br, orr


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=data_paths.METRICA_SAMPLE_DATA)   # see data_paths.py
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--drop-ball", type=float, default=0.41)
    args = ap.parse_args()

    games = [os.path.join(args.data, g) for g in ("Sample_Game_1", "Sample_Game_2")]

    if not args.sweep:
        for g in games:
            br, orr = score_game(g, P.PARAMS, args.drop_ball, bootstrap=500)
            print("\n{}   (truth {} possessions, mean {:.1f} s)".format(
                os.path.basename(g), br["n_gt_spells"], br["gt_mean_spell_duration_s"]))
            row = "  {:<22} spellF1 {:>7.4f}  90% CI [{:.4f}, {:.4f}]   poss {:>4}   frag {:>5.2f}"
            print(row.format("smoothed baseline", br["spell_f1_mean"],
                             br["spell_bootstrap_low"], br["spell_bootstrap_high"],
                             br["n_pred_spells"], br["fragmentation"]))
            print(row.format("ours", orr["spell_f1_mean"],
                             orr["spell_bootstrap_low"], orr["spell_bootstrap_high"],
                             orr["n_pred_spells"], orr["fragmentation"]))
        sys.exit(0)

    # cache tracking + truth once per game, then sweep only the model params
    cache = []
    for g in games:
        name = os.path.basename(g)
        n, ball, hx, ax = load_tracking(os.path.join(g, name + "_RawTrackingData_Home_Team.csv"),
                                        os.path.join(g, name + "_RawTrackingData_Away_Team.csv"))
        truth, blocks = truth_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
        team_obs, present, conf = tracking_features(ball, hx, ax, PITCH, drop_ball=args.drop_ball)
        dead_prior = dead_prior_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
        base = P.smoothed_baseline(team_obs, present, blocks, n)
        cache.append((truth, team_obs, present, dead_prior, blocks, n, conf,
                      pm.evaluate(truth, base)["spell_f1_mean"]))
    base_mean = np.mean([c[7] for c in cache])
    print("baseline spellF1 (mean of 2 games): {:.4f}\n".format(base_mean))

    grid = {"smooth_window": [25, 50, 75, 100], "w_dead": [0.0, 3.0],
            "c_switch_team": [6.0, 8.0, 12.0], "c_switch_dead": [3.5, 5.0]}
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        p = dict(P.PARAMS); p.update(dict(zip(keys, combo)))
        spells = []
        for truth, team_obs, present, dead_prior, blocks, n, conf, _ in cache:
            codes = P.build_labels(team_obs, present, dead_prior, blocks, n, p, conf)
            spells.append(pm.evaluate(truth, codes)["spell_f1_mean"])
        rows.append((np.mean(spells), p["smooth_window"], p["w_dead"],
                     p["c_switch_team"], p["c_switch_dead"]))
    rows.sort(reverse=True)
    print("top configs by mean spellF1 over the 2 games:")
    print("{:>8}  win  wdead cteam cdead".format("spellF1"))
    for s, sw, wd, ct, cd in rows[:12]:
        print("{:>8.4f}  {:>3}  {:>4}  {:>4}  {:>4}  {}".format(
            s, sw, wd, ct, cd, "<- beats baseline" if s > base_mean else ""))
