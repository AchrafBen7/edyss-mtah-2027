"""Definitive build on all 64 PFF matches: train v2 (30 feat) AND v3 (21 invariant feat), then
compare BOTH on PFF held-out and on Metrica cross-source, and save both models.

We pick the winner on evidence from two sources, not one. v3 drops the signed/orientation-dependent
features that look like a PFF habit. Held-out fixed to 10515-17. Builds features_cache_v2 for any
match still missing (covers the 19 new ones), then trains by column selection on that single cache.

    python final_build_64.py
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
CACHE = os.path.join(HERE, "features_cache_v2")
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py
PITCH = (105.0, 68.0)
SMOOTH = 60
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
TEST = ["10515", "10516", "10517"]
DROP_V3 = {"bx", "by", "ball_to_home_cx", "ball_to_away_cx", "ball_to_home_cy", "ball_to_away_cy",
           "home_minus_away_cx", "vx", "vy"}


def new_clf():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, random_state=0)


# 1) build 30-feature cache for every match on disk (skip the ones already cached)
ids = sorted((f.split(".")[0] for f in os.listdir(TRK) if f.endswith(".jsonl.bz2")), key=int)
names = None
for mid in ids:
    outf = os.path.join(CACHE, "{}.npz".format(mid))
    if os.path.exists(outf):
        continue
    try:
        m = pff_loader.load_match(mid)
        X, y, graded, names = features.frame_features(m, extra=True)
        np.savez_compressed(outf, X=X[graded], y=y[graded])
        print("build {}  {} img".format(mid, int(graded.sum())), flush=True)
    except Exception as e:
        print("SKIP {}: {}".format(mid, e), flush=True)
names = json.load(open(os.path.join(CACHE, "names.json"))) if names is None else names
keep_v3 = [i for i, nm in enumerate(names) if nm not in DROP_V3]

built = sorted((i for i in ids if os.path.exists(os.path.join(CACHE, "{}.npz".format(i)))), key=int)
TRAIN = [i for i in built if i not in TEST]
print("\ntrain {} matchs | held-out {}".format(len(TRAIN), [i for i in TEST if i in built]), flush=True)

Xtr = np.concatenate([np.load(os.path.join(CACHE, i + ".npz"))["X"].astype(np.float32) for i in TRAIN])
ytr = np.concatenate([np.load(os.path.join(CACHE, i + ".npz"))["y"] for i in TRAIN]).astype(int)
print("train set: {} images, {} features".format(Xtr.shape[0], Xtr.shape[1]), flush=True)

clf_v2 = new_clf().fit(Xtr, ytr)
joblib.dump({"model": clf_v2, "names": names}, os.path.join(HERE, "possession_model_v2_64.joblib"))
print("v2 (30 feat) entraine", flush=True)
clf_v3 = new_clf().fit(Xtr[:, keep_v3], ytr)
joblib.dump({"model": clf_v3, "names": [names[i] for i in keep_v3], "keep": keep_v3},
            os.path.join(HERE, "possession_model_v3_64.joblib"))
print("v3 (21 feat) entraine", flush=True)
del Xtr, ytr


def eval_set(samples):
    """samples: list of (X30, blocks, present, n, truth). Returns dict label -> {b: meanspellF1}."""
    out = {"v2": {}, "v3": {}}
    for b in (0.0, 0.25, 0.5):
        sv2, sv3 = [], []
        for X30, blocks, present, n, truth in samples:
            pv2 = apply_bias(clf_v2.predict_proba(X30), b)
            pv3 = apply_bias(clf_v3.predict_proba(X30[:, keep_v3]), b)
            sv2.append(pm.evaluate(truth, decode(smooth_probs(pv2, blocks, SMOOTH), present, blocks, n, PARAMS))["spell_f1_mean"])
            sv3.append(pm.evaluate(truth, decode(smooth_probs(pv3, blocks, SMOOTH), present, blocks, n, PARAMS))["spell_f1_mean"])
        out["v2"][b] = float(np.mean(sv2)); out["v3"][b] = float(np.mean(sv3))
    return out


# 2) PFF held-out samples (30-feat X computed once, shared by both models)
pff = []
for mid in [i for i in TEST if i in built]:
    m = pff_loader.load_match(mid)
    X, _, _, _ = features.frame_features(m, extra=True)
    _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
    pff.append((X, m["blocks"], present, m["n"], m["truth"]))

# 3) Metrica samples
met = []
for name in ("Sample_Game_1", "Sample_Game_2"):
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
    met.append((X, blocks, present, n, truth))

rp = eval_set(pff)
rm = eval_set(met)
print("\n=== COMPARAISON 64 matchs (spellF1, sm60) ===", flush=True)
print("{:<8}{:>10}{:>10}{:>10}   {:>10}{:>10}{:>10}".format(
    "", "PFF b0", "PFF b.25", "PFF b.5", "Met b0", "Met b.25", "Met b.5"), flush=True)
for lab in ("v2", "v3"):
    print("{:<8}{:>10.4f}{:>10.4f}{:>10.4f}   {:>10.4f}{:>10.4f}{:>10.4f}".format(
        lab, rp[lab][0.0], rp[lab][0.25], rp[lab][0.5], rm[lab][0.0], rm[lab][0.25], rm[lab][0.5]), flush=True)
print("\nref 45 matchs: v2 PFF 0.391 / Met 0.345 ; heuristique Met ~0.40 (tunee dessus)", flush=True)
