"""Feed the learned model's probabilities into the Viterbi and score spellF1 on held-out matches.

The real test: does the model improve the competition metric (spellF1), not just per-frame accuracy?
Scored on PFF matches the model never saw. Both pipelines use the SAME Viterbi + abstention.

KEY LESSON: raw per-frame model probabilities flicker and fragment possessions, so they LOSE on
spellF1 despite better per-frame accuracy. The fix is to smooth the probabilities over a window
(the same denoising that made the heuristic work) before decoding. This sweeps that window.

    python eval_model.py                 # sweep smoothing windows
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

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = ["10515", "10516", "10517"]
WINDOWS = [0, 15, 25, 50]
W_EMIT = 1.0


def smooth_probs(probs, blocks, w):
    if w <= 0:
        return probs
    out = probs.copy()
    for s, e in blocks:
        seg = probs[s:e]
        cs = np.concatenate([np.zeros((1, seg.shape[1])), np.cumsum(seg, axis=0)])
        idx = np.arange(len(seg))
        lo = np.maximum(0, idx - w)
        hi = np.minimum(len(seg), idx + w + 1)
        out[s:e] = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    return out


def decode(probs, ball_present, blocks, n, p):
    emis = W_EMIT * np.log(np.clip(probs, 1e-6, 1.0))
    trans = P.build_transitions(p)
    codes = np.full(n, P.X, np.int8)
    for s, e in blocks:
        path = P.viterbi(emis[s:e], trans)
        path = P.add_abstentions(path, ball_present[s:e], p["abstain_margin"])
        codes[s:e] = path
    return codes


if __name__ == "__main__":
    bundle = joblib.load(os.path.join(HERE, "possession_model.joblib"))
    clf = bundle["model"]
    p = dict(P.PARAMS)

    cache = []
    for mid in TEST:
        m = pff_loader.load_match(mid)
        X, y, graded, nm = features.frame_features(m)
        probs = clf.predict_proba(X)
        team, present, conf = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
        cache.append((mid, m, probs, team, present, conf))
        print("charge {}".format(mid), flush=True)

    heur_scores, learned = [], {w: [] for w in WINDOWS}
    print("\n{:<8} {:>12}  {}".format("match", "heuristique",
                                      "  ".join("sm{}".format(w) for w in WINDOWS)))
    for mid, m, probs, team, present, conf in cache:
        n, blocks, truth = m["n"], m["blocks"], m["truth"]
        heur = pm.evaluate(truth, P.build_labels(team, present, np.zeros(n, bool), blocks, n, p, conf))
        heur_scores.append(heur["spell_f1_mean"])
        line = "{:<8} {:>11.4f} ".format(mid, heur["spell_f1_mean"])
        for w in WINDOWS:
            codes = decode(smooth_probs(probs, blocks, w), present, blocks, n, p)
            s = pm.evaluate(truth, codes)["spell_f1_mean"]
            learned[w].append(s)
            line += "  {:.4f}".format(s)
        print(line, flush=True)

    print("\n=== moyenne held-out (3 matchs) ===")
    print("heuristique:            {:.4f}".format(np.mean(heur_scores)))
    for w in WINDOWS:
        tag = "  <= meilleur" if np.mean(learned[w]) == max(np.mean(learned[x]) for x in WINDOWS) else ""
        print("modele appris sm={:<3}:  {:.4f}{}".format(w, np.mean(learned[w]), tag))
