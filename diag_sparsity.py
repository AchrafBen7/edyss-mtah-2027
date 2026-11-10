"""Does player sparsity hurt our model? Apply challenge-like player dropout to Metrica and re-score.

The challenge tracks ~9.5 players/team vs 11 on Metrica. If masking Metrica players down to the
challenge regime collapses our current model's spellF1, sparsity is a real cause of the transfer wall,
and training with player-dropout augmentation should help. Cheap test: no retraining.

    python diag_sparsity.py
"""
import os
import warnings

import numpy as np
import joblib

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import possession as P
import possession_metrics as pm
import features
from eval_model import smooth_probs, decode
from calibrate_bias import apply_bias
import dev_sampledata as dv

HERE = os.path.dirname(os.path.abspath(__file__))
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py
PITCH = (105.0, 68.0)
SMOOTH, BIAS = 60, 0.25
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
clf = joblib.load(os.path.join(HERE, "possession_model_final.joblib"))["model"]


def load_raw(name):
    g = os.path.join(DATA, name)
    n, ball, home, away = dv.load_tracking(os.path.join(g, name + "_RawTrackingData_Home_Team.csv"),
                                           os.path.join(g, name + "_RawTrackingData_Away_Team.csv"))
    truth, blocks = dv.truth_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
    period = np.ones(n, np.int8)
    if len(blocks) > 1:
        period[blocks[1][0]:] = 2
    return dict(n=n, ball=ball, home=home, away=away, present=~np.isnan(ball[:, 0]),
                period=period, blocks=blocks, truth=truth)


def drop_players(xy, drop, rng):
    out = xy.copy()
    mask = (rng.random_sample(xy.shape[:2]) < drop) & ~np.isnan(xy[:, :, 0])
    out[mask] = np.nan
    return out


def score(raw, drop, seed):
    rng = np.random.RandomState(seed)
    home = drop_players(raw["home"], drop, rng)
    away = drop_players(raw["away"], drop, rng)
    scale = np.array(PITCH)
    m = dict(ball_xy=(raw["ball"] - 0.5) * scale, home_xy=(home - 0.5) * scale,
             away_xy=(away - 0.5) * scale, ball_present=raw["present"],
             period=raw["period"], truth=raw["truth"], fps=25.0, n=raw["n"])
    X, _, _, _ = features.frame_features(m, extra=True)
    proba = apply_bias(clf.predict_proba(X), BIAS)
    codes = decode(smooth_probs(proba, raw["blocks"], SMOOTH), raw["present"], raw["blocks"], raw["n"], PARAMS)
    return pm.evaluate(raw["truth"], codes)["spell_f1_mean"]


games = [load_raw("Sample_Game_1"), load_raw("Sample_Game_2")]
print("drop = fraction de joueurs masques par frame (0.15 -> ~9.5/equipe, comme le defi)\n")
print("{:>6}   {}".format("drop", "spellF1 Metrica (moyenne 2 matchs, 2 seeds)"), flush=True)
for drop in (0.0, 0.15, 0.30):
    vals = [score(r, drop, s) for r in games for s in (0, 1)]
    print("{:>6.2f}   {:.4f}".format(drop, float(np.mean(vals))), flush=True)
