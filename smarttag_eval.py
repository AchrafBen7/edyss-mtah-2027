"""Validate the dead-ball prior mechanism on Metrica before using the challenge's Smart Tagging.

The challenge ships smart_tagging.xml (referee dead-ball events). The model path does not use it yet.
Here we test the MECHANISM on Metrica, whose event log gives an analogous dead prior: after smoothing
the model probabilities we boost P(D) on flagged frames, then decode. If this lifts spellF1 on Metrica,
the same injection with dead_prior_from_tagging should help on the real challenge.

    python smarttag_eval.py
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
WEIGHTS = [0.0, 1.0, 2.0, 3.0, 5.0]
clf = joblib.load(os.path.join(HERE, "possession_model_v2_64.joblib"))["model"]


def boost_dead(probs, dead, w):
    if w <= 0:
        return probs
    lp = np.log(np.clip(probs, 1e-9, 1.0))
    lp[dead, 2] += w
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(1, keepdims=True)


samples = []
for name in ("Sample_Game_1", "Sample_Game_2"):
    g = os.path.join(DATA, name)
    n, ball, home, away = dv.load_tracking(os.path.join(g, name + "_RawTrackingData_Home_Team.csv"),
                                           os.path.join(g, name + "_RawTrackingData_Away_Team.csv"))
    truth, blocks = dv.truth_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
    dead = dv.dead_prior_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
    scale = np.array(PITCH); period = np.ones(n, np.int8)
    if len(blocks) > 1:
        period[blocks[1][0]:] = 2
    present = ~np.isnan(ball[:, 0])
    m = dict(ball_xy=(ball - 0.5) * scale, home_xy=(home - 0.5) * scale, away_xy=(away - 0.5) * scale,
             ball_present=present, period=period, truth=truth, fps=25.0, n=n)
    X, _, _, _ = features.frame_features(m, extra=True)
    probs_s = smooth_probs(apply_bias(clf.predict_proba(X), BIAS), blocks, SMOOTH)
    samples.append((probs_s, dead, blocks, present, n, truth))
    print("{}: {} frames flaggees mort ({:.1f}%)".format(name, int(dead.sum()), 100 * dead.mean()), flush=True)

print("\npoids_mort   spellF1 Metrica (moyenne 2 matchs)", flush=True)
best = (0.0, -1)
for w in WEIGHTS:
    sc = []
    for probs_s, dead, blocks, present, n, truth in samples:
        codes = decode(boost_dead(probs_s, dead, w), present, blocks, n, PARAMS)
        sc.append(pm.evaluate(truth, codes)["spell_f1_mean"])
    s = float(np.mean(sc))
    print("  w={:<4} -> {:.4f}".format(w, s), flush=True)
    if s > best[1]:
        best = (w, s)
print("\nmeilleur poids mort = {} ({:.4f})  [w=0 = modele seul]".format(*best), flush=True)
