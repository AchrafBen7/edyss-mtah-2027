"""Build v2 features (extra=True) for every PFF match on disk, train, and report held-out spellF1.

Fully separate from the v1 artifacts: writes features_cache_v2/ and possession_model_v2.joblib, never
touches the current best. Scans pff/tracking/ so any matches added later are picked up automatically.
Compares against v1's held-out best (b=0.25, sm50 -> 0.3216) so we only promote v2 if it actually wins.

    python run_v2.py
"""
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

HERE = os.path.dirname(os.path.abspath(__file__))
TRK = os.path.join(HERE, "pff", "tracking")
CACHE = os.path.join(HERE, "features_cache_v2")
os.makedirs(CACHE, exist_ok=True)
WINDOWS = [25, 50]
BIASES = [0.0, 0.25]

ids = sorted(f.split(".")[0] for f in os.listdir(TRK) if f.endswith(".jsonl.bz2"))
print("matchs sur disque: {}".format(ids), flush=True)

names = None
for mid in ids:
    outf = os.path.join(CACHE, "{}.npz".format(mid))
    if os.path.exists(outf):
        print("cache v2 deja la: {}".format(mid), flush=True)
        continue
    try:
        m = pff_loader.load_match(mid)
        X, y, graded, names = features.frame_features(m, extra=True)
        np.savez_compressed(outf, X=X[graded], y=y[graded])
        print("v2 build {}  {} images gradees  ({} features)".format(mid, int(graded.sum()), X.shape[1]), flush=True)
    except Exception as e:
        print("SKIP {}: {}".format(mid, e), flush=True)
if names is None:
    _m = pff_loader.load_match(ids[0])
    _, _, _, names = features.frame_features(_m, extra=True)
json.dump(names, open(os.path.join(CACHE, "names.json"), "w"))

# Fixed held-out so numbers stay comparable as matches are added (string sort would otherwise move it)
built = sorted((i for i in ids if os.path.exists(os.path.join(CACHE, "{}.npz".format(i)))), key=int)
TEST = [i for i in ["10515", "10516", "10517"] if i in built]
TRAIN = [i for i in built if i not in TEST]
print("\ntrain {} matchs | held-out {}".format(len(TRAIN), TEST), flush=True)

Xtr = np.concatenate([np.load(os.path.join(CACHE, "{}.npz".format(i)))["X"].astype(np.float32) for i in TRAIN])
ytr = np.concatenate([np.load(os.path.join(CACHE, "{}.npz".format(i)))["y"] for i in TRAIN]).astype(int)
print("train set: {} images, {} features".format(Xtr.shape[0], Xtr.shape[1]), flush=True)

clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                     l2_regularization=1.0, early_stopping=True,
                                     validation_fraction=0.1, random_state=0)
clf.fit(Xtr, ytr)
joblib.dump({"model": clf, "names": names}, os.path.join(HERE, "possession_model_v2.joblib"))
print("modele v2 sauvegarde -> possession_model_v2.joblib", flush=True)

p = dict(P.PARAMS)
res = {}
for mid in TEST:
    m = pff_loader.load_match(mid)
    X, _, _, _ = features.frame_features(m, extra=True)
    proba = clf.predict_proba(X)
    _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
    for b in BIASES:
        pb = apply_bias(proba, b)
        for w in WINDOWS:
            codes = decode(smooth_probs(pb, m["blocks"], w), present, m["blocks"], m["n"], p)
            res.setdefault((b, w), []).append(pm.evaluate(m["truth"], codes)["spell_f1_mean"])
    print("evalue {}".format(mid), flush=True)

print("\n=== V2 held-out spellF1 (reference v1 = 0.3216) ===", flush=True)
best = (None, -1)
for (b, w), v in sorted(res.items()):
    s = float(np.mean(v))
    print("  b={:<4} sm={:<3} -> {:.4f}".format(b, w, s), flush=True)
    if s > best[1]:
        best = ((b, w), s)
delta = best[1] - 0.3216
print("\nV2 MEILLEUR: b={} sm={} -> {:.4f}  ({:+.4f} vs v1)  {}".format(
    best[0][0], best[0][1], best[1], delta, "<<< V2 GAGNE" if delta > 0 else "v1 reste devant"), flush=True)
