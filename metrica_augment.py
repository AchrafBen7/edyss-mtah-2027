"""Rigorous leave-one-out test: does adding Metrica (the challenge's own format) to training help?

We train on the 61 PFF train matches PLUS one Metrica game, and score on the OTHER Metrica game,
which was never in training. Then swap. If the held-out Metrica score rises versus the PFF-only
model, in-domain data genuinely helps and should transfer to the Metrica-format challenge. We also
check PFF held-out does not degrade. No tuning on the challenge: decisions are on held-out only.

    python metrica_augment.py
"""
import glob
import os
import warnings

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

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
CACHE = os.path.join(HERE, "features_cache_v2")
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py
PITCH = (105.0, 68.0)
SMOOTH, BIAS = 60, 0.25
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
TEST = ["10515", "10516", "10517"]
GAMES = ["Sample_Game_1", "Sample_Game_2"]


def new_clf():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, random_state=0)


def metrica_full(name):
    g = os.path.join(DATA, name)
    n, ball, home, away = dv.load_tracking(os.path.join(g, name + "_RawTrackingData_Home_Team.csv"),
                                           os.path.join(g, name + "_RawTrackingData_Away_Team.csv"))
    truth, blocks = dv.truth_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
    scale = np.array(PITCH); period = np.ones(n, np.int8)
    if len(blocks) > 1:
        period[blocks[1][0]:] = 2
    present = ~np.isnan(ball[:, 0])
    m = dict(ball_xy=(ball - 0.5) * scale, home_xy=(home - 0.5) * scale, away_xy=(away - 0.5) * scale,
             ball_present=present, period=period, truth=truth, fps=25.0, n=n)
    X, _, _, _ = features.frame_features(m, extra=True)
    return dict(X=X.astype(np.float32), truth=truth, blocks=blocks, present=present, n=n)


def score(clf, s, b=BIAS):
    p = apply_bias(clf.predict_proba(s["X"]), b)
    codes = decode(smooth_probs(p, s["blocks"], SMOOTH), s["present"], s["blocks"], s["n"], PARAMS)
    return pm.evaluate(s["truth"], codes)["spell_f1_mean"]


# PFF training pool (61 matches), cached
ids = sorted((os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(CACHE, "*.npz"))), key=int)
TRAIN = [i for i in ids if i not in TEST]
Xpff = np.concatenate([np.load(os.path.join(CACHE, i + ".npz"))["X"].astype(np.float32) for i in TRAIN])
ypff = np.concatenate([np.load(os.path.join(CACHE, i + ".npz"))["y"] for i in TRAIN]).astype(int)
print("PFF train pool: {} matchs, {} images".format(len(TRAIN), Xpff.shape[0]), flush=True)

met = {g: metrica_full(g) for g in GAMES}
for g in GAMES:
    gr = met[g]["truth"] != P.X
    print("{}: {} frames, {} gradees".format(g, met[g]["n"], int(gr.sum())), flush=True)

pffho = []
for mid in TEST:
    m = pff_loader.load_match(mid)
    X, _, _, _ = features.frame_features(m, extra=True)
    _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
    pffho.append(dict(X=X.astype(np.float32), truth=m["truth"], blocks=m["blocks"], present=present, n=m["n"]))


def graded(s):
    g = s["truth"] != P.X
    return s["X"][g], s["truth"][g].astype(int)


base = joblib.load(os.path.join(HERE, "possession_model_v2_64.joblib"))["model"]
print("\n=== BASELINE (PFF 64 seul) ===", flush=True)
b_g1, b_g2 = score(base, met["Sample_Game_1"]), score(base, met["Sample_Game_2"])
b_pff = float(np.mean([score(base, s) for s in pffho]))
print("  Metrica G1 {:.4f}   G2 {:.4f}   moyenne {:.4f}   | PFF held-out {:.4f}".format(
    b_g1, b_g2, (b_g1 + b_g2) / 2, b_pff), flush=True)

print("\n=== + METRICA en entrainement (leave-one-out) ===", flush=True)
# train on PFF + G2, hold out G1
Xg, yg = graded(met["Sample_Game_2"])
clfA = new_clf().fit(np.concatenate([Xpff, Xg]), np.concatenate([ypff, yg]))
a_g1 = score(clfA, met["Sample_Game_1"]); a_pffA = float(np.mean([score(clfA, s) for s in pffho]))
# train on PFF + G1, hold out G2
Xg, yg = graded(met["Sample_Game_1"])
clfB = new_clf().fit(np.concatenate([Xpff, Xg]), np.concatenate([ypff, yg]))
a_g2 = score(clfB, met["Sample_Game_2"]); a_pffB = float(np.mean([score(clfB, s) for s in pffho]))
print("  Metrica G1 (held-out) {:.4f}   G2 (held-out) {:.4f}   moyenne {:.4f}".format(
    a_g1, a_g2, (a_g1 + a_g2) / 2), flush=True)
print("  PFF held-out: {:.4f} / {:.4f}".format(a_pffA, a_pffB), flush=True)

print("\n=== VERDICT ===", flush=True)
print("  Metrica held-out:  baseline {:.4f}  ->  +Metrica {:.4f}   ({:+.4f})".format(
    (b_g1 + b_g2) / 2, (a_g1 + a_g2) / 2, (a_g1 + a_g2) / 2 - (b_g1 + b_g2) / 2), flush=True)
print("  PFF held-out:      baseline {:.4f}  ->  +Metrica {:.4f}   ({:+.4f})".format(
    b_pff, (a_pffA + a_pffB) / 2, (a_pffA + a_pffB) / 2 - b_pff), flush=True)
