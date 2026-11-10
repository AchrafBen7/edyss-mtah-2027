"""Production model WITH player-dropout augmentation: 61 PFF + 2 Metrica, trained on sparse frames.

The go/no-go showed dropout augmentation lifts BOTH clean PFF (+0.008) and challenge-regime masked
Metrica (+0.016). The challenge tracks only ~9.5 players/team, a regime our model never saw; training
with per-frame player dropout fixes that domain gap. Inference on the (already sparse) challenge is
unchanged. Fixed per-match seeds for reproducibility. Saves possession_model_dropout.joblib.

    python final_dropout.py
"""
import glob
import json
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
TRK = os.path.join(HERE, "pff", "tracking")
CACHE = os.path.join(HERE, "features_cache_dropout")
os.makedirs(CACHE, exist_ok=True)
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py
PITCH = (105.0, 68.0)
SMOOTH, BIAS = 60, 0.25
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
TEST = ["10515", "10516", "10517"]
GAMES = ["Sample_Game_1", "Sample_Game_2"]


def drop_aug(home, away, rng):
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


# 1) build dropout-augmented 30-feature cache for every PFF match (reload; slow)
ids = sorted((f.split(".")[0] for f in os.listdir(TRK) if f.endswith(".jsonl.bz2")), key=int)
names = None
for mid in ids:
    outf = os.path.join(CACHE, "{}.npz".format(mid))
    if os.path.exists(outf):
        continue
    try:
        m = pff_loader.load_match(mid)
        rng = np.random.RandomState(int(mid))
        h, a = drop_aug(m["home_xy"], m["away_xy"], rng)
        X, y, graded, names = features.frame_features({**m, "home_xy": h, "away_xy": a}, extra=True)
        np.savez_compressed(outf, X=X[graded].astype(np.float32), y=y[graded])
        print("build {}  {} img".format(mid, int(graded.sum())), flush=True)
    except Exception as e:
        print("SKIP {}: {}".format(mid, e), flush=True)
if names is None:
    names = json.load(open(os.path.join(CACHE, "names.json")))
json.dump(names, open(os.path.join(CACHE, "names.json"), "w"))


def metrica_aug_graded(name, seed):
    g = os.path.join(DATA, name)
    n, ball, home, away = dv.load_tracking(os.path.join(g, name + "_RawTrackingData_Home_Team.csv"),
                                           os.path.join(g, name + "_RawTrackingData_Away_Team.csv"))
    truth, blocks = dv.truth_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
    rng = np.random.RandomState(seed)
    h, a = drop_aug(home, away, rng)
    scale = np.array(PITCH); period = np.ones(n, np.int8)
    if len(blocks) > 1:
        period[blocks[1][0]:] = 2
    m = dict(ball_xy=(ball - 0.5) * scale, home_xy=(h - 0.5) * scale, away_xy=(a - 0.5) * scale,
             ball_present=~np.isnan(ball[:, 0]), period=period, truth=truth, fps=25.0, n=n)
    X, _, _, _ = features.frame_features(m, extra=True)
    gr = truth != P.X
    return X[gr].astype(np.float32), truth[gr].astype(int)


built = sorted((i for i in ids if os.path.exists(os.path.join(CACHE, "{}.npz".format(i)))), key=int)
TRAIN = [i for i in built if i not in TEST]
parts_X = [np.load(os.path.join(CACHE, i + ".npz"))["X"].astype(np.float32) for i in TRAIN]
parts_y = [np.load(os.path.join(CACHE, i + ".npz"))["y"] for i in TRAIN]
for k, g in enumerate(GAMES):
    Xg, yg = metrica_aug_graded(g, 7 + k)
    parts_X.append(Xg); parts_y.append(yg)
Xtr = np.concatenate(parts_X); ytr = np.concatenate(parts_y).astype(int)
del parts_X, parts_y
print("train dropout: {} images, {} features".format(Xtr.shape[0], Xtr.shape[1]), flush=True)

clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                     l2_regularization=1.0, early_stopping=True,
                                     validation_fraction=0.1, random_state=0).fit(Xtr, ytr)
joblib.dump({"model": clf, "names": names}, os.path.join(HERE, "possession_model_dropout.joblib"))
print("modele dropout sauvegarde -> possession_model_dropout.joblib", flush=True)


def score(clf, m, blocks, present, n, truth, b=BIAS):
    proba = apply_bias(clf.predict_proba(features.frame_features(m, extra=True)[0]), b)
    codes = decode(smooth_probs(proba, blocks, SMOOTH), present, blocks, n, PARAMS)
    return pm.evaluate(truth, codes)["spell_f1_mean"]


print("\n=== PFF held-out propre (modele dropout) ===  (ref sans dropout = 0.401)", flush=True)
for b in (0.0, 0.25, 0.5):
    s = []
    for mid in TEST:
        m = pff_loader.load_match(mid)
        _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
        s.append(score(clf, m, m["blocks"], present, m["n"], m["truth"], b))
    print("  b={:<4} -> {:.4f}".format(b, float(np.mean(s))), flush=True)
