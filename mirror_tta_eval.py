"""Orientation fix via test-time augmentation (no retrain): average the model over the scene and its
180 deg pitch rotation, making the output orientation-invariant by construction.

A 180 deg rotation (x,y)->(-x,-y) is possession-preserving: it negates EXACTLY the 9 signed features
and leaves the other 21 (and the label) unchanged. So mirror(X) = X with those 9 columns negated. We
average proba(X) and proba(mirror(X)). Judged on BOTH honest sources: PFF held-out (must stay ~0.401)
and SkillCorner (must stay ~0.585), plus the rotation flip-rate (must collapse to ~0). Go only if it
holds or improves on both.

    python mirror_tta_eval.py
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
import skillcorner_loader as skl
from eval_model import smooth_probs, decode
from calibrate_bias import apply_bias
from features import _wf

BIAS, SMOOTH_REF = 0.25, 60
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
from data_paths import SKILLCORNER_DATA as SKDIR  # see data_paths.py
bundle = joblib.load(os.path.join(HERE, "possession_model_final.joblib"))
clf, names = bundle["model"], bundle["names"]
ORI = ["bx", "by", "ball_to_home_cx", "ball_to_away_cx", "ball_to_home_cy",
       "ball_to_away_cy", "home_minus_away_cx", "vx", "vy"]
idx = [names.index(c) for c in ORI]


def evalm(m):
    fps = m["fps"]
    X, _, _, _ = features.frame_features(m, extra=True)
    Xm = X.copy(); Xm[:, idx] *= -1.0                       # 180 deg rotation
    pb = apply_bias(clf.predict_proba(X), BIAS)
    pm_ = apply_bias(clf.predict_proba(Xm), BIAS)
    ptta = 0.5 * (pb + pm_)
    sm = _wf(SMOOTH_REF, fps)
    base = pm.evaluate(m["truth"], decode(smooth_probs(pb, m["blocks"], sm), m["ball_present"], m["blocks"], m["n"], PARAMS))["spell_f1_mean"]
    tta = pm.evaluate(m["truth"], decode(smooth_probs(ptta, m["blocks"], sm), m["ball_present"], m["blocks"], m["n"], PARAMS))["spell_f1_mean"]
    # rotation flip rate on live (H/A) frames
    a0 = pb.argmax(1); a1 = pm_.argmax(1)
    live = (a0 == 0) | (a0 == 1)
    flip = float(np.mean((a0[live] != a1[live]))) if live.any() else 0.0
    return base, tta, flip


print("{:<12}{:>10}{:>10}{:>12}".format("match", "base", "+mirrorTTA", "flip%(rot)"), flush=True)
pb_, pt_ = [], []
for mid in ["10515", "10516", "10517"]:
    b, t, f = evalm(pff_loader.load_match(mid))
    pb_.append(b); pt_.append(t)
    print("PFF {:<8}{:>10.4f}{:>10.4f}{:>11.1f}%".format(mid, b, t, 100 * f), flush=True)
print("  PFF moyenne   base {:.4f}   +TTA {:.4f}".format(np.mean(pb_), np.mean(pt_)), flush=True)

sb_, st_ = [], []
for mid in ["1874553", "1899585", "1886347"]:
    b, t, f = evalm(skl.load_match(os.path.join(SKDIR, mid)))
    sb_.append(b); st_.append(t)
    print("SK  {:<8}{:>10.4f}{:>10.4f}{:>11.1f}%".format(mid, b, t, 100 * f), flush=True)
print("  SkillCorner moyenne   base {:.4f}   +TTA {:.4f}".format(np.mean(sb_), np.mean(st_)), flush=True)
print("\nGO si: flip% chute vers ~0 ET PFF+TTA >= base ET SkillCorner+TTA >= base", flush=True)
