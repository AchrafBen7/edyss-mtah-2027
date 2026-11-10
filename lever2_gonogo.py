"""Lever 2: STRUCTURED masking (keep players near the ball), the fix for the failed random dropout.

The challenge drops off-camera players, i.e. those far from the ball (broadcast framing). Random
dropout failed because it was unstructured. Here we mask PFF players beyond R=40 m of the ball (which
reproduces the challenge's ~9.5 players/team), retrain, and -- crucially -- judge on the BLIND
challenge, not on a masking scheme identical to training. Trains a normal model (A) and a structured-
masked model (B) on 18 matches; the follow-up scores both on the public 5-min.

    python lever2_gonogo.py
"""
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
import predict_challenge as pc
from games import GAME, GAMES, half_video_blocks

HERE = pc.HERE
SMOOTH, BIAS, R = 60, 0.25, 40.0
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
TRAIN = ["3812", "3816", "3820", "3824", "3828", "3832", "3836", "3840", "3844",
         "3848", "3851", "3855", "3859", "10502", "10505", "10508", "10511", "10514"]
TEST = ["10515", "10516", "10517"]


def new_clf():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, random_state=0)


def ffill(a):
    idx = np.where(~np.isnan(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def mask_structured(ball_xy, home, away, R):
    """Drop players farther than R from the ball (ball position forward-filled when missing)."""
    bx, by = ffill(ball_xy[:, 0]), ffill(ball_xy[:, 1])
    out = []
    for grp in (home, away):
        d = np.sqrt((grp[:, :, 0] - bx[:, None]) ** 2 + (grp[:, :, 1] - by[:, None]) ** 2)
        z = grp.copy()
        z[(d > R) & ~np.isnan(grp[:, :, 0])] = np.nan
        out.append(z)
    return out


def graded(m):
    X, y, g, _ = features.frame_features(m, extra=True)
    return X[g].astype(np.float32), y[g].astype(int)


Xa, ya, Xb, yb = [], [], [], []
for mid in TRAIN:
    m = pff_loader.load_match(mid)
    Xc, yc = graded(m)
    h, a = mask_structured(m["ball_xy"], m["home_xy"], m["away_xy"], R)
    Xd, yd = graded({**m, "home_xy": h, "away_xy": a})
    Xa.append(Xc); ya.append(yc); Xb.append(Xd); yb.append(yd)
    print("train {} ok".format(mid), flush=True)
clfA = new_clf().fit(np.concatenate(Xa), np.concatenate(ya))
clfB = new_clf().fit(np.concatenate(Xb), np.concatenate(yb))
joblib.dump({"model": clfA}, os.path.join(HERE, "model_lever2A.joblib"))
joblib.dump({"model": clfB}, os.path.join(HERE, "model_lever2B.joblib"))
print("modeles entraines (A normal | B masquage structure R={})\n".format(R), flush=True)


def pff_score(clf, mid):
    m = pff_loader.load_match(mid)
    _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
    proba = apply_bias(clf.predict_proba(features.frame_features(m, extra=True)[0]), BIAS)
    codes = decode(smooth_probs(proba, m["blocks"], SMOOTH), present, m["blocks"], m["n"], PARAMS)
    return pm.evaluate(m["truth"], codes)["spell_f1_mean"]


def gen_challenge(clf, out_name):
    pitch = GAMES[GAME]["pitch"]; n = GAMES[GAME]["n_video_frames"]
    blocks = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    tac = os.path.join(HERE, "data", "tactical")
    tot = P.team_of_each_track(os.path.join(tac, "{}_tactical_atd.xml".format(GAME)))
    ball, present, home, away = pc.parse_all_positions(os.path.join(tac, "{}_tactical_atd.txt".format(GAME)),
                                                       tot, n, pitch)
    period = np.zeros(n, np.int8)
    for i, (x, y) in enumerate(blocks, 1):
        period[x:y] = i
    m = dict(ball_xy=ball, home_xy=home, away_xy=away, ball_present=present,
             period=period, truth=np.full(n, P.X, np.int8), fps=25.0, n=n)
    proba = apply_bias(clf.predict_proba(features.frame_features(m, extra=True)[0]), BIAS)
    codes = decode(smooth_probs(proba, blocks, SMOOTH), present, blocks, n, PARAMS)
    with open(os.path.join(HERE, out_name), "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, P.LETTER[c]) for i, c in enumerate(codes))
    return out_name


print("=== held-out PFF (propre) ===", flush=True)
pA = float(np.mean([pff_score(clfA, mid) for mid in TEST]))
pB = float(np.mean([pff_score(clfB, mid) for mid in TEST]))
print("  A normal {:.4f}   B structure {:.4f}   delta {:+.4f}".format(pA, pB, pB - pA), flush=True)

gen_challenge(clfA, "submission_lever2A.csv")
gen_challenge(clfB, "submission_lever2B.csv")
print("submissions defi ecrites (A et B) -> scoring dans le follow-up", flush=True)
