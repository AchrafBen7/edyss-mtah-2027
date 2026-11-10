"""Extract per-frame features + labels for every PFF training match and cache them.

Run once; it writes features_cache/<match_id>.npz (graded frames only) and names.json, so training
afterwards is instant. Loading a PFF match is the slow part (~1 min each), hence the cache.

    python build_features.py
"""

import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import pff_loader
import features

HERE = os.path.dirname(os.path.abspath(__file__))
IDS = [str(i) for i in range(10506, 10518)]            # the 12 PFF World Cup matches we have
OUT = os.path.join(HERE, "features_cache")
os.makedirs(OUT, exist_ok=True)

names = None
total = 0
for mid in IDS:
    try:
        m = pff_loader.load_match(mid)
    except Exception as e:
        print("SKIP {}: {}".format(mid, e), flush=True)
        continue
    X, y, graded, names = features.frame_features(m)
    Xg, yg = X[graded], y[graded]
    np.savez_compressed(os.path.join(OUT, "{}.npz".format(mid)), X=Xg, y=yg)
    total += len(yg)
    print("{} done  {} graded frames  (H/A/D = {})".format(
        mid, len(yg), [int(np.sum(yg == k)) for k in (0, 1, 2)]), flush=True)

if names is not None:
    json.dump(names, open(os.path.join(OUT, "names.json"), "w"))
print("\nCACHE COMPLETE: {} matches, {} total graded frames in {}".format(
    len(os.listdir(OUT)) - 1, total, OUT), flush=True)
