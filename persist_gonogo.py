"""Go/no-go for possession-persistence features (last_poss/streak/obs_roll75), judged on BOTH honest
sources: PFF held-out and SkillCorner cross-source. Train 18 PFF matches, baseline (30 feat) vs
+persist (33 feat). GO only if +persist >= baseline on BOTH (else reject, keep the 30-feat model).

    python persist_gonogo.py
"""
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "evaluation"))
import possession_metrics as pm
import possession as P
import features
import pff_loader
import skillcorner_loader as skl
from eval_model import smooth_probs, decode
from calibrate_bias import apply_bias
from features import _wf

BIAS, SMOOTH_REF, NBASE = 0.25, 60, 30
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
TRAIN = ["3812", "3816", "3820", "3824", "3828", "3832", "3836", "3840", "3844",
         "3848", "3851", "3855", "3859", "10502", "10505", "10508", "10511", "10514"]
TEST = ["10515", "10516", "10517"]
SK = ["1874553", "1899585", "1886347"]
from data_paths import SKILLCORNER_DATA as SKDIR  # see data_paths.py


def new_clf():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, random_state=0)


Xa, ya, Xb, yb = [], [], [], []
for mid in TRAIN:
    m = pff_loader.load_match(mid)
    X, y, g, _ = features.frame_features(m, extra=True, persist=True)
    Xg = X[g].astype(np.float32); yg = y[g].astype(int)
    Xa.append(Xg[:, :NBASE]); Xb.append(Xg); ya.append(yg); yb.append(yg)
    print("train {} ok".format(mid), flush=True)
clfA = new_clf().fit(np.concatenate(Xa), np.concatenate(ya))
clfB = new_clf().fit(np.concatenate(Xb), np.concatenate(yb))
print("modeles: A=30feat  B=33feat(+persist)\n", flush=True)


def both(m):
    X, _, _, _ = features.frame_features(m, extra=True, persist=True)
    sm = _wf(SMOOTH_REF, m["fps"])
    out = []
    for clf, nc in ((clfA, NBASE), (clfB, X.shape[1])):
        proba = apply_bias(clf.predict_proba(X[:, :nc]), BIAS)
        codes = decode(smooth_probs(proba, m["blocks"], sm), m["ball_present"], m["blocks"], m["n"], PARAMS)
        out.append(pm.evaluate(m["truth"], codes)["spell_f1_mean"])
    return out[0], out[1]


print("{:<12}{:>10}{:>12}".format("match", "base30", "+persist33"), flush=True)
pa, pb = [], []
for mid in TEST:
    a, b = both(pff_loader.load_match(mid)); pa.append(a); pb.append(b)
    print("PFF {:<8}{:>10.4f}{:>12.4f}".format(mid, a, b), flush=True)
print("  PFF moyenne   base {:.4f}   +persist {:.4f}".format(np.mean(pa), np.mean(pb)), flush=True)
sa, sb = [], []
for mid in SK:
    a, b = both(skl.load_match(os.path.join(SKDIR, mid))); sa.append(a); sb.append(b)
    print("SK  {:<8}{:>10.4f}{:>12.4f}".format(mid, a, b), flush=True)
print("  SkillCorner moyenne   base {:.4f}   +persist {:.4f}".format(np.mean(sa), np.mean(sb)), flush=True)
dp, ds = np.mean(pb) - np.mean(pa), np.mean(sb) - np.mean(sa)
print("\ndelta: PFF {:+.4f}  SkillCorner {:+.4f}  -> {}".format(
    dp, ds, "GO (>=0 sur les 2)" if (dp >= -0.001 and ds >= -0.001) else "NO-GO"), flush=True)
