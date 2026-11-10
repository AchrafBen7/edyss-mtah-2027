"""Measure and correct the model's systematic home/away imbalance, tuned on held-out PFF.

Diagnostic finding: on neutral World Cup data the truth is ~50/50 H/A, yet the model under-calls
HOME by ~17 points (predicted H-share 0.32 vs truth 0.49). That is a property of the model, not of
the match, so the same correction carries validly to the challenge. We add a log-odds offset to the
home class, choose it on held-out spellF1 (never on the challenge), and report the gain.

    python calibrate_bias.py
"""
import os
import warnings

import numpy as np
import joblib

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import possession as P
import possession_metrics as pm
import pff_loader
import features
from eval_model import smooth_probs, decode

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = ["10515", "10516", "10517"]
SMOOTHS = [25, 50]
BIASES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def apply_bias(proba, b):
    """Add a log-odds offset b to the HOME class, then renormalise (b>0 -> more home)."""
    lp = np.log(np.clip(proba, 1e-9, 1.0))
    lp[:, 0] += b
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


if __name__ == "__main__":
    bundle = joblib.load(os.path.join(HERE, "possession_model.joblib"))
    clf = bundle["model"]
    p = dict(P.PARAMS)

    cache, tshare = [], []
    for mid in TEST:
        m = pff_loader.load_match(mid)
        X, _, _, _ = features.frame_features(m)
        proba = clf.predict_proba(X)
        _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
        cache.append((mid, m, proba, present))
        t = m["truth"]; liv = (t == 0) | (t == 1)
        tshare.append(float(np.mean(t[liv] == 0)))
        print("charge {}".format(mid), flush=True)

    print("\nverite H-part (live, held-out): {:.3f}".format(np.mean(tshare)))
    print("\n{:<6}{}   predHpart(sm50)".format("b", "".join("   sm{:<3}".format(w) for w in SMOOTHS)))
    best = (None, -1)
    for b in BIASES:
        means = {w: [] for w in SMOOTHS}; hs = []
        for mid, m, proba, present in cache:
            pb = apply_bias(proba, b)
            for w in SMOOTHS:
                codes = decode(smooth_probs(pb, m["blocks"], w), present, m["blocks"], m["n"], p)
                means[w].append(pm.evaluate(m["truth"], codes)["spell_f1_mean"])
                if w == 50:
                    liv = (codes == 0) | (codes == 1)
                    hs.append(float(np.mean(codes[liv] == 0)) if liv.any() else 0.0)
        line = "{:<6.2f}".format(b) + "".join("  {:.4f}".format(np.mean(means[w])) for w in SMOOTHS)
        line += "     {:.3f}".format(np.mean(hs))
        print(line, flush=True)
        for w in SMOOTHS:
            if np.mean(means[w]) > best[1]:
                best = ((b, w), float(np.mean(means[w])))
    print("\nMEILLEUR: b={} smooth={}  ->  held-out spellF1 {:.4f}".format(best[0][0], best[0][1], best[1]))
