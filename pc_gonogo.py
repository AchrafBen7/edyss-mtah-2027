"""Go/no-go for pitch-control features: do the 8 extra columns help on held-out PFF AND Metrica?

Efficient test before committing to a full 64-match rebuild: train on ~18 varied PFF matches with the
38-feature set, and compare the model on 30 columns (current) vs 38 columns (with pitch control), on
3 held-out PFF matches and 2 Metrica games. We keep pitch control only if the delta is positive on
BOTH sources (same discipline as everything else; guards against overfitting).

    python pc_gonogo.py
"""
import os
import warnings

import numpy as np
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
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py
PITCH = (105.0, 68.0)
SMOOTH, BIAS = 60, 0.25
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
TRAIN = ["3812", "3816", "3820", "3824", "3828", "3832", "3836", "3840", "3844",
         "3848", "3851", "3855", "3859", "10502", "10505", "10508", "10511", "10514"]
TEST = ["10515", "10516", "10517"]
GAMES = ["Sample_Game_1", "Sample_Game_2"]
NBASE = 30                                   # first 30 columns = the current (extra) feature set


def new_clf():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, random_state=0)


def pff_sample(mid, train=False):
    m = pff_loader.load_match(mid)
    X, y, graded, _ = features.frame_features(m, extra=True, pitch_control=True)
    if train:
        return X[graded].astype(np.float32), y[graded].astype(int)
    _, present, _ = pff_loader.nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
    return dict(X=X.astype(np.float32), blocks=m["blocks"], present=present, n=m["n"], truth=m["truth"])


def metrica_sample(name):
    g = os.path.join(DATA, name)
    n, ball, home, away = dv.load_tracking(os.path.join(g, name + "_RawTrackingData_Home_Team.csv"),
                                           os.path.join(g, name + "_RawTrackingData_Away_Team.csv"))
    truth, blocks = dv.truth_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
    scale = np.array(PITCH); period = np.ones(n, np.int8)
    if len(blocks) > 1:
        period[blocks[1][0]:] = 2
    m = dict(ball_xy=(ball - 0.5) * scale, home_xy=(home - 0.5) * scale, away_xy=(away - 0.5) * scale,
             ball_present=~np.isnan(ball[:, 0]), period=period, truth=truth, fps=25.0, n=n)
    X, _, _, _ = features.frame_features(m, extra=True, pitch_control=True)
    return dict(X=X.astype(np.float32), blocks=blocks, present=~np.isnan(ball[:, 0]), n=n, truth=truth)


def score(clf, s, ncols):
    proba = apply_bias(clf.predict_proba(s["X"][:, :ncols]), BIAS)
    codes = decode(smooth_probs(proba, s["blocks"], SMOOTH), s["present"], s["blocks"], s["n"], PARAMS)
    return pm.evaluate(s["truth"], codes)["spell_f1_mean"]


Xtr, ytr = [], []
for mid in TRAIN:
    X, y = pff_sample(mid, train=True)
    Xtr.append(X); ytr.append(y)
    print("train {} ok".format(mid), flush=True)
Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
print("train: {} images, {} features\n".format(Xtr.shape[0], Xtr.shape[1]), flush=True)

clf30 = new_clf().fit(Xtr[:, :NBASE], ytr)
clf38 = new_clf().fit(Xtr, ytr)
print("modeles entraines (30 vs 38 features)\n", flush=True)

pff = [pff_sample(mid) for mid in TEST]
met = [metrica_sample(g) for g in GAMES]

p30 = float(np.mean([score(clf30, s, NBASE) for s in pff]))
p38 = float(np.mean([score(clf38, s, Xtr.shape[1]) for s in pff]))
m30 = float(np.mean([score(clf30, s, NBASE) for s in met]))
m38 = float(np.mean([score(clf38, s, Xtr.shape[1]) for s in met]))

print("=== GO/NO-GO pitch control (sous-ensemble {} matchs) ===".format(len(TRAIN)), flush=True)
print("                 30 feat    38 feat (+PC)   delta", flush=True)
print("  PFF held-out    {:.4f}     {:.4f}        {:+.4f}".format(p30, p38, p38 - p30), flush=True)
print("  Metrica         {:.4f}     {:.4f}        {:+.4f}".format(m30, m38, m38 - m30), flush=True)
verdict = "GO (positif sur les 2)" if (p38 - p30 > 0 and m38 - m30 > 0) else \
          "NO-GO (pas positif sur les 2) -> rejeter"
print("\n  VERDICT: {}".format(verdict), flush=True)
