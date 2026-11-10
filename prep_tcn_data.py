"""Build the TCN input streams (probability-stream decoder idea), saved for a separate torch venv.

Per frame, the TCN sees only transfer-stable signals: [P(H), P(A), P(D) from the GBM (bias-applied),
ball_present, conf, obs_roll25]. It learns temporal shape / boundary placement (source-invariant),
not geometry->team. Saved as npz per match for: 18 PFF train, 3 PFF held-out, 3 SkillCorner. Run in
the MAIN venv (needs sklearn). A clean torch venv then trains/validates on these arrays only.

    python prep_tcn_data.py
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
import possession as P
import features
import pff_loader
import skillcorner_loader as skl
from calibrate_bias import apply_bias

OUT = os.path.join(HERE, "tcn_data")
os.makedirs(OUT, exist_ok=True)
BIAS = 0.25
TRAIN = ["3812", "3816", "3820", "3824", "3828", "3832", "3836", "3840", "3844",
         "3848", "3851", "3855", "3859", "10502", "10505", "10508", "10511", "10514"]
TEST = ["10515", "10516", "10517"]
SK = ["1874553", "1899585", "1886347"]
from data_paths import SKILLCORNER_DATA as SKDIR  # see data_paths.py
bundle = joblib.load(os.path.join(HERE, "possession_model_final.joblib"))
clf, names = bundle["model"], bundle["names"]
CI, OI = names.index("conf"), names.index("obs_roll25")


def build(m, tag):
    X, _, _, _ = features.frame_features(m, extra=True)
    proba = apply_bias(clf.predict_proba(X), BIAS)                 # (n,3) H,A,D
    conf = np.nan_to_num(X[:, CI]); obsr = np.nan_to_num(X[:, OI])
    present = m["ball_present"].astype(np.float32)
    inp = np.column_stack([proba, present, conf, obsr]).astype(np.float32)   # (n,6)
    blocks = np.array(m["blocks"], dtype=np.int64)
    np.savez_compressed(os.path.join(OUT, tag + ".npz"), inp=inp,
                        truth=m["truth"].astype(np.int8), present=present,
                        blocks=blocks, fps=np.float32(m["fps"]))
    return inp.shape


for mid in TRAIN:
    print("train", mid, build(pff_loader.load_match(mid), "train_" + mid), flush=True)
for mid in TEST:
    print("held-out", mid, build(pff_loader.load_match(mid), "test_" + mid), flush=True)
for mid in SK:
    print("skillcorner", mid, build(skl.load_match(os.path.join(SKDIR, mid)), "sk_" + mid), flush=True)
print("FAIT -> tcn_data/", flush=True)
