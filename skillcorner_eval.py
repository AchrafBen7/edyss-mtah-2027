"""Honest cross-source baseline: run the deployed model on SkillCorner (real broadcast sparsity).

Also re-scores PFF held-out as a CONTROL: with the new fps-aware feature/smoothing windows, PFF
(29.97 fps) must be unchanged (~0.401) -> proves the seconds refactor introduced no regression.
SkillCorner (10 fps) is the first transfer-honest number: its sparsity is camera-produced, not ours.

    python skillcorner_eval.py                                   # the submitted model
    python skillcorner_eval.py possession_model_dropout.joblib   # any model on the same 30 features
"""
import os
import warnings

import numpy as np
import joblib

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "evaluation"))
import possession_metrics as pm
import possession as P
import features
import pff_loader
import skillcorner_loader as sk
from eval_model import smooth_probs, decode
from calibrate_bias import apply_bias
from features import _wf

BIAS, SMOOTH_REF = 0.25, 60
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
from data_paths import SKILLCORNER_DATA as SKDIR  # see data_paths.py
MODEL = sys.argv[1] if len(sys.argv) > 1 else "possession_model_final.joblib"
clf = joblib.load(os.path.join(HERE, MODEL))["model"]
print("model: {}".format(MODEL), flush=True)


def score(m):
    fps = m["fps"]
    X, _, _, _ = features.frame_features(m, extra=True)
    proba = apply_bias(clf.predict_proba(X), BIAS)
    sm = _wf(SMOOTH_REF, fps)
    codes = decode(smooth_probs(proba, m["blocks"], sm), m["ball_present"], m["blocks"], m["n"], PARAMS)
    r = pm.evaluate(m["truth"], codes)
    return r["spell_f1_mean"], sm, r.get("n_pred_spells"), r.get("n_gt_spells")


print("=== CONTROLE PFF held-out (doit rester ~0.401 ; sinon la refonte a casse qqch) ===", flush=True)
pff = []
for mid in ["10515", "10516", "10517"]:
    m = pff_loader.load_match(mid)
    s, sm, npred, ngt = score(m)
    pff.append(s)
    print("  {}  spellF1 {:.4f}  (smooth {} frames, pred/gt {}/{})".format(mid, s, sm, npred, ngt), flush=True)
print("  moyenne PFF held-out: {:.4f}".format(np.mean(pff)), flush=True)

print("\n=== BASELINE SkillCorner (cross-source honnete, 10 fps) ===", flush=True)
sks = []
for mid in ["1874553", "1899585", "1886347"]:
    m = sk.load_match(os.path.join(SKDIR, mid))
    s, sm, npred, ngt = score(m)
    sks.append(s)
    print("  {}  spellF1 {:.4f}  (smooth {} frames, pred/gt {}/{})".format(mid, s, sm, npred, ngt), flush=True)
print("  moyenne SkillCorner: {:.4f}".format(np.mean(sks)), flush=True)
print("\n(reference: PFF held-out ~0.401 = notre chiffre fiable PFF ; SkillCorner = nouveau juge cross-source)", flush=True)
