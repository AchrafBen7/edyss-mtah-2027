"""Final submission model: 61 PFF train matches + both Metrica games (source diversity).

The leave-one-out test (metrica_augment.py) proved that adding Metrica-format data lifts the held-out
Metrica score (+0.015) without hurting PFF (+0.001). The challenge is Metrica-format, so we fold both
Metrica games into training for the production model, keeping PFF 10515-17 held out so a trustworthy
number stays reportable. Saves possession_model_final.joblib.

    python train_final.py
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
CACHE = os.path.join(HERE, "features_cache_v2")
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py
PITCH = (105.0, 68.0)
SMOOTH, BIAS = 60, 0.25
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
TEST = ["10515", "10516", "10517"]
GAMES = ["Sample_Game_1", "Sample_Game_2"]


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
    X, _, _, _ = features.frame_features(m, extra=True)
    gr = truth != P.X
    return X[gr].astype(np.float32), truth[gr].astype(int)


names = json.load(open(os.path.join(CACHE, "names.json")))
ids = sorted((os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(CACHE, "*.npz"))), key=int)
TRAIN = [i for i in ids if i not in TEST]

parts_X = [np.load(os.path.join(CACHE, i + ".npz"))["X"].astype(np.float32) for i in TRAIN]
parts_y = [np.load(os.path.join(CACHE, i + ".npz"))["y"] for i in TRAIN]
for g in GAMES:
    Xg, yg = metrica_graded(g)
    parts_X.append(Xg); parts_y.append(yg)
    print("+ {} : {} images".format(g, len(yg)), flush=True)
Xtr = np.concatenate(parts_X); ytr = np.concatenate(parts_y).astype(int)
del parts_X, parts_y
print("train final: {} images ({} PFF matchs + {} Metrica), {} features".format(
    Xtr.shape[0], len(TRAIN), len(GAMES), Xtr.shape[1]), flush=True)

clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                     l2_regularization=1.0, early_stopping=True,
                                     validation_fraction=0.1, random_state=0).fit(Xtr, ytr)
joblib.dump({"model": clf, "names": names}, os.path.join(HERE, "possession_model_final.joblib"))
print("modele final sauvegarde -> possession_model_final.joblib", flush=True)

print("\n=== PFF held-out (modele final) ===", flush=True)
for mid in TEST:
    m = pff_loader.load_match(mid)
    X, _, _, _ = features.frame_features(m, extra=True)
    _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
    globals().setdefault("ho", []).append((clf.predict_proba(X), m["blocks"], present, m["n"], m["truth"]))
for b in (0.0, 0.25, 0.5):
    s = []
    for proba, blocks, present, n, truth in ho:
        codes = decode(smooth_probs(apply_bias(proba, b), blocks, SMOOTH), present, blocks, n, PARAMS)
        s.append(pm.evaluate(truth, codes)["spell_f1_mean"])
    print("  b={:<4} -> {:.4f}".format(b, float(np.mean(s))), flush=True)
print("\n(baseline PFF-seul held-out etait 0.394 ; on verifie qu'on ne perd rien)", flush=True)
