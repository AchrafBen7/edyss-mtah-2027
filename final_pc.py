"""Production model WITH pitch control: 61 PFF + 2 Metrica, 38 features (extra + pitch_control).

The go/no-go (pc_gonogo.py) showed pitch control lifts held-out PFF (+0.019) AND Metrica (+0.010),
so we rebuild the full training set with the 38-feature extraction and train the production model.
Builds features_cache_pc/ (reusable). Held-out 10515-17 kept out so a trustworthy number stays
reportable. Saves possession_model_pc.joblib.

    python final_pc.py
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
CACHE = os.path.join(HERE, "features_cache_pc")
os.makedirs(CACHE, exist_ok=True)
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py
PITCH = (105.0, 68.0)
SMOOTH, BIAS = 60, 0.25
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
TEST = ["10515", "10516", "10517"]
GAMES = ["Sample_Game_1", "Sample_Game_2"]

# 1) build the 38-feature cache for every PFF match (reload; this is the slow part)
ids = sorted((f.split(".")[0] for f in os.listdir(TRK) if f.endswith(".jsonl.bz2")), key=int)
names = None
for mid in ids:
    outf = os.path.join(CACHE, "{}.npz".format(mid))
    if os.path.exists(outf):
        continue
    try:
        m = pff_loader.load_match(mid)
        X, y, graded, names = features.frame_features(m, extra=True, pitch_control=True)
        np.savez_compressed(outf, X=X[graded].astype(np.float32), y=y[graded])
        print("build {}  {} img  {} feat".format(mid, int(graded.sum()), X.shape[1]), flush=True)
    except Exception as e:
        print("SKIP {}: {}".format(mid, e), flush=True)
if names is None:
    names = json.load(open(os.path.join(CACHE, "names.json")))
json.dump(names, open(os.path.join(CACHE, "names.json"), "w"))


def metrica_graded(name):
    g = os.path.join(DATA, name)
    n, ball, home, away = dv.load_tracking(os.path.join(g, name + "_RawTrackingData_Home_Team.csv"),
                                           os.path.join(g, name + "_RawTrackingData_Away_Team.csv"))
    truth, blocks = dv.truth_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
    scale = np.array(PITCH); period = np.ones(n, np.int8)
    if len(blocks) > 1:
        period[blocks[1][0]:] = 2
    m = dict(ball_xy=(ball - 0.5) * scale, home_xy=(home - 0.5) * scale, away_xy=(away - 0.5) * scale,
             ball_present=~np.isnan(ball[:, 0]), period=period, truth=truth, fps=25.0, n=n)
    X, _, _, _ = features.frame_features(m, extra=True, pitch_control=True)
    gr = truth != P.X
    return X[gr].astype(np.float32), truth[gr].astype(int)


built = sorted((i for i in ids if os.path.exists(os.path.join(CACHE, "{}.npz".format(i)))), key=int)
TRAIN = [i for i in built if i not in TEST]
parts_X = [np.load(os.path.join(CACHE, i + ".npz"))["X"].astype(np.float32) for i in TRAIN]
parts_y = [np.load(os.path.join(CACHE, i + ".npz"))["y"] for i in TRAIN]
for g in GAMES:
    Xg, yg = metrica_graded(g)
    parts_X.append(Xg); parts_y.append(yg)
    print("+ {} : {} images".format(g, len(yg)), flush=True)
Xtr = np.concatenate(parts_X); ytr = np.concatenate(parts_y).astype(int)
del parts_X, parts_y
print("train final PC: {} images, {} features ({} PFF + {} Metrica)".format(
    Xtr.shape[0], Xtr.shape[1], len(TRAIN), len(GAMES)), flush=True)

clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                     l2_regularization=1.0, early_stopping=True,
                                     validation_fraction=0.1, random_state=0).fit(Xtr, ytr)
joblib.dump({"model": clf, "names": names}, os.path.join(HERE, "possession_model_pc.joblib"))
print("modele PC sauvegarde -> possession_model_pc.joblib", flush=True)

print("\n=== PFF held-out (modele PC, 38 feat) ===  (ref sans PC = 0.401)", flush=True)
ho = []
for mid in TEST:
    m = pff_loader.load_match(mid)
    X, _, _, _ = features.frame_features(m, extra=True, pitch_control=True)
    _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
    ho.append((clf.predict_proba(X), m["blocks"], present, m["n"], m["truth"]))
for b in (0.0, 0.25, 0.5):
    s = []
    for proba, blocks, present, n, truth in ho:
        codes = decode(smooth_probs(apply_bias(proba, b), blocks, SMOOTH), present, blocks, n, PARAMS)
        s.append(pm.evaluate(truth, codes)["spell_f1_mean"])
    print("  b={:<4} -> {:.4f}".format(b, float(np.mean(s))), flush=True)
