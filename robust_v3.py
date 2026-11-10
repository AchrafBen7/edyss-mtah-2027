"""v3 robustness test: does dropping DIRECTION-dependent features improve cross-source generalisation?

v2 beats the heuristic on PFF held-out (0.39 vs 0.24) but loses on Metrica (0.34 vs 0.40, though the
heuristic's PARAMS were tuned on those Metrica games, so 0.40 is in-sample). Per-frame live H/A falls
0.84 -> 0.65 PFF -> Metrica: real overfitting. Suspect the signed, orientation-dependent features.
We retrain on the invariant subset and score BOTH sources. No cache rebuild: columns are selected on
the existing v2 cache for training; features are recomputed only for the few eval matches.

    python robust_v3.py
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
SMOOTH = 60
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
DROP = {"bx", "by", "ball_to_home_cx", "ball_to_away_cx", "ball_to_home_cy", "ball_to_away_cy",
        "home_minus_away_cx", "vx", "vy"}

names = json.load(open(os.path.join(CACHE, "names.json")))
keep = [i for i, nm in enumerate(names) if nm not in DROP]
print("v3 gardees ({}): {}".format(len(keep), [names[i] for i in keep]))
print("v3 abandonnees ({}): {}\n".format(len(names) - len(keep), sorted(DROP)), flush=True)

ids = sorted((os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(CACHE, "*.npz"))),
             key=lambda s: int(s))
TEST = ["10515", "10516", "10517"]
TRAIN = [i for i in ids if i not in TEST]
Xtr = np.concatenate([np.load(os.path.join(CACHE, i + ".npz"))["X"][:, keep].astype(np.float32) for i in TRAIN])
ytr = np.concatenate([np.load(os.path.join(CACHE, i + ".npz"))["y"] for i in TRAIN]).astype(int)
clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                     l2_regularization=1.0, early_stopping=True,
                                     validation_fraction=0.1, random_state=0).fit(Xtr, ytr)
print("v3 entraine: {} images, {} features\n".format(Xtr.shape[0], Xtr.shape[1]), flush=True)


def score(proba_full, blocks, present, n, truth, b):
    codes = decode(smooth_probs(apply_bias(proba_full, b), blocks, SMOOTH), present, blocks, n, PARAMS)
    return pm.evaluate(truth, codes)["spell_f1_mean"]


print("=== PFF held-out (v3) ===", flush=True)
pff = []
for mid in TEST:
    m = pff_loader.load_match(mid)
    X, _, _, _ = features.frame_features(m, extra=True)
    _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
    pff.append((clf.predict_proba(X[:, keep]), m["blocks"], present, m["n"], m["truth"]))
for b in (0.0, 0.25, 0.5):
    print("  b={:<4} -> {:.4f}".format(b, float(np.mean([score(*p, b) for p in pff]))), flush=True)

print("\n=== Metrica cross-source (v3) ===", flush=True)
met = []
for name in ("Sample_Game_1", "Sample_Game_2"):
    g = os.path.join(DATA, name)
    n, ball, home, away = dv.load_tracking(os.path.join(g, name + "_RawTrackingData_Home_Team.csv"),
                                           os.path.join(g, name + "_RawTrackingData_Away_Team.csv"))
    truth, blocks = dv.truth_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
    scale = np.array(PITCH)
    period = np.ones(n, np.int8)
    if len(blocks) > 1:
        period[blocks[1][0]:] = 2
    present = ~np.isnan(ball[:, 0])
    m = dict(ball_xy=(ball - 0.5) * scale, home_xy=(home - 0.5) * scale, away_xy=(away - 0.5) * scale,
             ball_present=present, period=period, truth=truth, fps=25.0, n=n)
    X, _, _, _ = features.frame_features(m, extra=True)
    met.append((clf.predict_proba(X[:, keep]), blocks, present, n, truth))
for b in (0.0, 0.25, 0.5):
    print("  b={:<4} -> {:.4f}".format(b, float(np.mean([score(*p, b) for p in met]))), flush=True)

print("\n(ref v2 30-feat: PFF 0.391 @b0.25 ; Metrica 0.345 @b0.25 ; heuristique Metrica ~0.40 mais tunee dessus)",
      flush=True)
