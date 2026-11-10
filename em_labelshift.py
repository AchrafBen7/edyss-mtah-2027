"""EM label-shift (Saerens 2002) vs the hand-set HOME_BIAS=0.25, judged on PFF held-out + SkillCorner.

The model under-calls home on neutral data; we currently patch it with a fixed +0.25 log-odds on the
home class. The principled replacement estimates each dataset's own class prior q(y) by EM on the
unlabeled outputs and corrects p(y|x) ∝ p_model(y|x)·q(y)/p_src(y). GO only if EM >= fixed-bias on BOTH
judges (then it adapts per-dataset for free and removes a magic constant); else keep the fixed bias.

    python em_labelshift.py
"""
import os
import sys
import warnings

import numpy as np
import joblib

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

BIAS, SMOOTH_REF = 0.25, 60
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
from data_paths import SKILLCORNER_DATA as SKDIR  # see data_paths.py
clf = joblib.load(os.path.join(HERE, "possession_model_final.joblib"))["model"]


def em_prior(probs, p_src, iters=100):
    q = p_src.copy()
    for _ in range(iters):
        post = probs * (q / p_src)[None, :]
        post /= post.sum(1, keepdims=True)
        q = post.mean(0)
    return q


def correct(probs, q, p_src):
    c = probs * (q / p_src)[None, :]
    return c / c.sum(1, keepdims=True)


def probs_of(m):
    X, _, _, _ = features.frame_features(m, extra=True)
    return clf.predict_proba(X)


# source prior = model's mean output on PFF held-out (proxy for the training marginal)
pffm = {mid: pff_loader.load_match(mid) for mid in ["10515", "10516", "10517"]}
pffp = {mid: probs_of(m) for mid, m in pffm.items()}
p_src = np.concatenate(list(pffp.values())).mean(0)
print("p_src (prior source estime) H/A/D = {:.3f}/{:.3f}/{:.3f}\n".format(*p_src), flush=True)


def score(m, probs, kind):
    fps = m["fps"]; sm = _wf(SMOOTH_REF, fps)
    if kind == "fixed":
        ps = apply_bias(probs, BIAS)
    else:
        q = em_prior(probs, p_src)
        ps = correct(probs, q, p_src)
    codes = decode(smooth_probs(ps, m["blocks"], sm), m["ball_present"], m["blocks"], m["n"], PARAMS)
    return pm.evaluate(m["truth"], codes)["spell_f1_mean"], (q if kind == "em" else None)


print("{:<12}{:>12}{:>12}   {}".format("match", "fixed b.25", "EM", "q_EM (H/A/D)"), flush=True)
fb, eb = [], []
for mid in ["10515", "10516", "10517"]:
    m = pffm[mid]; pr = pffp[mid]
    sf, _ = score(m, pr, "fixed"); se, q = score(m, pr, "em")
    fb.append(sf); eb.append(se)
    print("PFF {:<8}{:>12.4f}{:>12.4f}   {:.2f}/{:.2f}/{:.2f}".format(mid, sf, se, *q), flush=True)
print("  PFF moyenne  fixed {:.4f}   EM {:.4f}".format(np.mean(fb), np.mean(eb)), flush=True)

sf_, se_ = [], []
for mid in ["1874553", "1899585", "1886347"]:
    m = skl.load_match(os.path.join(SKDIR, mid)); pr = probs_of(m)
    sf, _ = score(m, pr, "fixed"); se, q = score(m, pr, "em")
    sf_.append(sf); se_.append(se)
    print("SK  {:<8}{:>12.4f}{:>12.4f}   {:.2f}/{:.2f}/{:.2f}".format(mid, sf, se, *q), flush=True)
print("  SkillCorner moyenne  fixed {:.4f}   EM {:.4f}".format(np.mean(sf_), np.mean(se_)), flush=True)
print("\nGO si EM >= fixed sur PFF ET SkillCorner.", flush=True)
