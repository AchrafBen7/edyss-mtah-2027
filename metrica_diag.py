"""Diagnose WHY the v2 model underperforms on Metrica: orientation flip, or no signal?

If one half has high normal accuracy and the other high FLIP accuracy, the direction-dependent
features flip at the end-change, i.e. the model keyed on absolute pitch side (a PFF habit), which is
not possession-invariant. That is fixable. If both halves are ~0.5 either way, the model has no grip.
"""
import os
import warnings

import numpy as np
import joblib

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import features
import dev_sampledata as dv

HERE = os.path.dirname(os.path.abspath(__file__))
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py
PITCH = (105.0, 68.0)
H, A = 0, 1
clf = joblib.load(os.path.join(HERE, "possession_model_v2.joblib"))["model"]

for name in ("Sample_Game_1", "Sample_Game_2"):
    g = os.path.join(DATA, name)
    n, ball, home, away = dv.load_tracking(
        os.path.join(g, name + "_RawTrackingData_Home_Team.csv"),
        os.path.join(g, name + "_RawTrackingData_Away_Team.csv"))
    truth, blocks = dv.truth_from_events(os.path.join(g, name + "_RawEventsData.csv"), n)
    scale = np.array(PITCH)
    period = np.ones(n, np.int8)
    if len(blocks) > 1:
        period[blocks[1][0]:] = 2
    m = dict(ball_xy=(ball - 0.5) * scale, home_xy=(home - 0.5) * scale, away_xy=(away - 0.5) * scale,
             ball_present=~np.isnan(ball[:, 0]), period=period, truth=truth, fps=25.0, n=n)
    X, _, _, _ = features.frame_features(m, extra=True)
    pred = clf.predict_proba(X).argmax(1)
    sw = np.where(pred == H, A, np.where(pred == A, H, pred))     # swap H<->A
    live = (truth == H) | (truth == A)
    print("\n{}  ({} images vivantes)".format(name, int(live.sum())))
    print("  global  acc {:.3f}   flip {:.3f}   truth H-part {:.3f}  pred H-part {:.3f}".format(
        np.mean(pred[live] == truth[live]), np.mean(sw[live] == truth[live]),
        np.mean(truth[live] == H), np.mean(pred[live] == H)))
    for pi, (a, b) in enumerate(blocks, 1):
        lh = live & (np.arange(n) >= a) & (np.arange(n) < b)
        if lh.sum():
            print("  mi-temps {}:  acc {:.3f}   flip {:.3f}".format(
                pi, np.mean(pred[lh] == truth[lh]), np.mean(sw[lh] == truth[lh])))
