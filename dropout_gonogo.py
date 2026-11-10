"""Go/no-go for player-dropout augmentation: does training on sparse frames help the challenge regime?

Diagnostic (diag_sparsity) showed our model loses ~0.025 on Metrica once players are masked to the
challenge's density. Here we train two models on the same 18 PFF matches -- one normal, one with
per-frame player-dropout augmentation -- and score both on (a) clean PFF held-out and (b) Metrica
MASKED to the challenge regime (the relevant test). If the augmented model wins on masked Metrica
without wrecking clean PFF, we rebuild on all 64. Compliant: only external data is masked.

    python dropout_gonogo.py
"""
import os
import warnings

import numpy as np
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
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py
PITCH = (105.0, 68.0)
SMOOTH, BIAS = 60, 0.25
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
TRAIN = ["3812", "3816", "3820", "3824", "3828", "3832", "3836", "3840", "3844",
         "3848", "3851", "3855", "3859", "10502", "10505", "10508", "10511", "10514"]
TEST = ["10515", "10516", "10517"]
GAMES = ["Sample_Game_1", "Sample_Game_2"]
EVAL_DROP = 0.15                     # challenge-like density for the masked-Metrica test


def new_clf():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, random_state=0)


def drop_aug(home, away, rng):
    """Per-frame random player dropout, rate itself varied in [0,0.3] so the model sees all densities."""
    out = []
    for xy in (home, away):
        n, K = xy.shape[0], xy.shape[1]
        rate = rng.uniform(0.0, 0.3, size=(n, 1))
        mask = (rng.random_sample((n, K)) < rate) & ~np.isnan(xy[:, :, 0])
        z = xy.copy(); z[mask] = np.nan
        out.append(z)
    return out


def drop_fixed(xy, drop, rng):
    mask = (rng.random_sample(xy.shape[:2]) < drop) & ~np.isnan(xy[:, :, 0])
    z = xy.copy(); z[mask] = np.nan
    return z


def graded(m):
    X, y, g, _ = features.frame_features(m, extra=True)
    return X[g].astype(np.float32), y[g].astype(int)


Xa, ya, Xb, yb = [], [], [], []
for i, mid in enumerate(TRAIN):
    m = pff_loader.load_match(mid)
    Xc, yc = graded(m)
    rng = np.random.RandomState(100 + i)
    h, a = drop_aug(m["home_xy"], m["away_xy"], rng)
    Xd, yd = graded({**m, "home_xy": h, "away_xy": a})
    Xa.append(Xc); ya.append(yc); Xb.append(Xd); yb.append(yd)
    print("train {} ok".format(mid), flush=True)
clfA = new_clf().fit(np.concatenate(Xa), np.concatenate(ya))
clfB = new_clf().fit(np.concatenate(Xb), np.concatenate(yb))
print("modeles entraines (A normal | B dropout-augmente)\n", flush=True)


def score(clf, m, blocks, present, n, truth):
    proba = apply_bias(clf.predict_proba(features.frame_features(m, extra=True)[0]), BIAS)
    codes = decode(smooth_probs(proba, blocks, SMOOTH), present, blocks, n, PARAMS)
    return pm.evaluate(truth, codes)["spell_f1_mean"]


def pff_eval(mid):
    m = pff_loader.load_match(mid)
    _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
    return m, m["blocks"], present, m["n"], m["truth"]


def metrica_masked(name, drop, seed):
    g = os.path.join(DATA, name)
    n, ball, home, away = dv.load_tracking(os.path.join(g, name + "_RawTrackingData_Home_Team.csv"),
                                           os.path.join(g, name + "_RawTrackingData_Away_Team.csv"))
    truth, blocks = dv.truth_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
    rng = np.random.RandomState(seed)
    period = np.ones(n, np.int8)
    if len(blocks) > 1:
        period[blocks[1][0]:] = 2
    scale = np.array(PITCH)
    m = dict(ball_xy=(ball - 0.5) * scale, home_xy=(drop_fixed(home, drop, rng) - 0.5) * scale,
             away_xy=(drop_fixed(away, drop, rng) - 0.5) * scale, ball_present=~np.isnan(ball[:, 0]),
             period=period, truth=truth, fps=25.0, n=n)
    return m, blocks, ~np.isnan(ball[:, 0]), n, truth


pff = [pff_eval(mid) for mid in TEST]
metm = [metrica_masked(g, EVAL_DROP, s) for g in GAMES for s in (0, 1)]

pA = float(np.mean([score(clfA, *s) for s in pff]))
pB = float(np.mean([score(clfB, *s) for s in pff]))
mA = float(np.mean([score(clfA, *s) for s in metm]))
mB = float(np.mean([score(clfB, *s) for s in metm]))

print("=== GO/NO-GO dropout augmentation ===", flush=True)
print("                         normal      +dropout     delta", flush=True)
print("  PFF held-out (propre)   {:.4f}      {:.4f}      {:+.4f}".format(pA, pB, pB - pA), flush=True)
print("  Metrica MASQUEE (defi)  {:.4f}      {:.4f}      {:+.4f}".format(mA, mB, mB - mA), flush=True)
go = (mB - mA > 0.003)
print("\n  VERDICT: {}".format("GO (aide le regime du defi)" if go else "NO-GO -> rejeter"), flush=True)
