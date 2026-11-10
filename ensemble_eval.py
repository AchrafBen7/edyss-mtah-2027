"""Ensemble: blend the learned model's probabilities with a robust geometric signal (nearest player).

Our one real risk is that the model overfits PFF and does not beat a geometry-only approach on the
Metrica-format challenge. The nearest-player signal is unbiased and transfers across sources; the
model is accurate in-domain. We blend  P = a * P_model + (1-a) * P_geom  and sweep a on BOTH PFF
held-out and Metrica. If an intermediate a beats a=1 (pure model) on Metrica while holding PFF, the
ensemble is the more robust submission. Uses the 64-match model.

    python ensemble_eval.py
"""
import os
import warnings

import numpy as np
import joblib

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import possession as P
import possession_metrics as pm
import pff_loader
import features
from eval_model import smooth_probs, decode
from calibrate_bias import apply_bias
import dev_sampledata as dv

HERE = os.path.dirname(os.path.abspath(__file__))
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py
PITCH = (105.0, 68.0)
SMOOTH, BIAS = 60, 0.25
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
ALPHAS = [0.0, 0.25, 0.5, 0.6, 0.75, 1.0]
TEST = ["10515", "10516", "10517"]
MODEL = "possession_model_v2_64.joblib"


def p_geom(present, team_obs, conf, n):
    """A soft (n,3) possession probability from the nearest-player observation + its confidence."""
    out = np.tile(np.array([0.40, 0.40, 0.20]), (n, 1))
    m = present & (team_obs >= 0)
    pt = 0.5 + 0.45 * np.clip(conf[m], 0.0, 1.0)         # prob the observed team, grows with margin
    obs = team_obs[m]
    g = np.empty((int(m.sum()), 3))
    g[:, 0] = np.where(obs == 0, pt, 1 - pt) * 0.9
    g[:, 1] = np.where(obs == 0, 1 - pt, pt) * 0.9
    g[:, 2] = 0.1
    out[m] = g
    return out / out.sum(1, keepdims=True)


def score_alpha(proba_model, pg, blocks, present, n, truth, a):
    blend = a * apply_bias(proba_model, BIAS) + (1 - a) * pg
    codes = decode(smooth_probs(blend, blocks, SMOOTH), present, blocks, n, PARAMS)
    return pm.evaluate(truth, codes)["spell_f1_mean"]


clf = joblib.load(os.path.join(HERE, MODEL))["model"]

pff = []
for mid in TEST:
    m = pff_loader.load_match(mid)
    X, _, _, _ = features.frame_features(m, extra=True)
    team, present, conf = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
    pff.append((clf.predict_proba(X), p_geom(present, team, conf, m["n"]),
                m["blocks"], present, m["n"], m["truth"]))

met = []
for name in ("Sample_Game_1", "Sample_Game_2"):
    g = os.path.join(DATA, name)
    n, ball, home, away = dv.load_tracking(os.path.join(g, name + "_RawTrackingData_Home_Team.csv"),
                                           os.path.join(g, name + "_RawTrackingData_Away_Team.csv"))
    truth, blocks = dv.truth_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
    team, present, conf = dv.tracking_features(ball, home, away, PITCH, drop_ball=0.0)
    scale = np.array(PITCH); period = np.ones(n, np.int8)
    if len(blocks) > 1:
        period[blocks[1][0]:] = 2
    m = dict(ball_xy=(ball - 0.5) * scale, home_xy=(home - 0.5) * scale, away_xy=(away - 0.5) * scale,
             ball_present=~np.isnan(ball[:, 0]), period=period, truth=truth, fps=25.0, n=n)
    X, _, _, _ = features.frame_features(m, extra=True)
    met.append((clf.predict_proba(X), p_geom(present, team, conf, n), blocks, present, n, truth))

print("alpha = 1.0 -> modele pur ; 0.0 -> geometrie pure ; intermediaire -> ensemble\n")
print("{:>6}{:>12}{:>12}{:>12}".format("alpha", "PFF held-out", "Metrica", "min(2)"), flush=True)
best = (None, -1)
for a in ALPHAS:
    sp = float(np.mean([score_alpha(*s, a) for s in pff]))
    sm = float(np.mean([score_alpha(*s, a) for s in met]))
    mn = min(sp, sm)
    print("{:>6.2f}{:>12.4f}{:>12.4f}{:>12.4f}".format(a, sp, sm, mn), flush=True)
    if mn > best[1]:
        best = (a, mn)
print("\nmeilleur equilibre (max du min des 2 sources): alpha={} (min {:.4f})".format(*best), flush=True)
