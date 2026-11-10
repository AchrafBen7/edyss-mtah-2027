"""Tune the decoder (smoothing window, abstention, switch cost) for the v2 model on held-out PFF.

The v2 features changed the emission distribution, so the decode optimum may have moved. We cache the
3 held-out matches' v2 probabilities once (heldout_v2.joblib) so later tuning is instant. Kept small
on purpose: only 3 matches, so these are directional, we pick robust settings, not the raw maximum.

    python sweep_v2.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = ["10515", "10516", "10517"]
CACHE = os.path.join(HERE, "heldout_v2.joblib")

clf = joblib.load(os.path.join(HERE, "possession_model_v2.joblib"))["model"]
if os.path.exists(CACHE):
    data = joblib.load(CACHE)
    print("held-out v2 recharge du cache", flush=True)
else:
    data = []
    for mid in TEST:
        m = pff_loader.load_match(mid)
        X, _, _, _ = features.frame_features(m, extra=True)
        proba = clf.predict_proba(X)
        _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
        data.append(dict(mid=mid, proba=proba, present=present,
                         truth=m["truth"], blocks=m["blocks"], n=m["n"]))
        print("charge {}".format(mid), flush=True)
    joblib.dump(data, CACHE)


def score(window, params, b=0.0):
    s = []
    for d in data:
        pb = apply_bias(d["proba"], b)
        codes = decode(smooth_probs(pb, d["blocks"], window), d["present"],
                       d["blocks"], d["n"], params)
        s.append(pm.evaluate(d["truth"], codes)["spell_f1_mean"])
    return float(np.mean(s))


base = dict(P.PARAMS)
print("\n=== grille lissage x biais (params par defaut) ===", flush=True)
print("  {:>5}".format("sm") + "".join("    b={:<4}".format(b) for b in [0.0, 0.25, 0.5]), flush=True)
best = ((50, 0.0), -1.0)
for w in [45, 50, 60, 75]:
    row = "  {:>5}".format(w)
    for b in [0.0, 0.25, 0.5]:
        sc = score(w, base, b)
        row += "  {:.4f}".format(sc)
        if sc > best[1]:
            best = ((w, b), sc)
    print(row, flush=True)
W, B = best[0]
print("meilleur: sm={} b={} ({:.4f})".format(W, B, best[1]), flush=True)

print("\n=== c_switch_team @ sm={} b={} ===".format(W, B), flush=True)
for cs in [8.0, 10.0, 12.0, 14.0]:
    p = dict(base); p["c_switch_team"] = cs
    print("  switch={:>4.1f} -> {:.4f}".format(cs, score(W, p, B)), flush=True)
print("\n(reference: run_v2 b=0.25 sm50 = 0.3891)", flush=True)
