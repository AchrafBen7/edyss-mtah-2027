"""Cross-source validation: run the PFF-trained v2 model on Metrica sample-data (a DIFFERENT source).

The challenge itself is Metrica-format, so this checks exactly the right thing: does a model trained
only on PFF World Cup generalise to Metrica's convention? Metrica positions are 0-1 pitch fractions;
we centre to metres like the challenge before extracting the identical 30 features. Model and heuristic
share the same ball-present mask so the comparison is head-to-head on identical frames. We report two
regimes: full ball, and ball hidden at 41% to match the challenge's missing-ball rate.

    python metrica_validate.py
"""
import os
import warnings

import numpy as np
import joblib

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import possession as P
import possession_metrics as pm
import features
from eval_model import smooth_probs, decode
from calibrate_bias import apply_bias
import dev_sampledata as dv

HERE = os.path.dirname(os.path.abspath(__file__))
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py
PITCH = (105.0, 68.0)
SMOOTH, BIAS = 60, 0.25
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}
GAMES = [os.path.join(DATA, g) for g in ("Sample_Game_1", "Sample_Game_2")]


def load_game(game_dir, drop_ball, seed=0):
    name = os.path.basename(game_dir)
    n, ball, home, away = dv.load_tracking(
        os.path.join(game_dir, name + "_RawTrackingData_Home_Team.csv"),
        os.path.join(game_dir, name + "_RawTrackingData_Away_Team.csv"))
    truth, blocks = dv.truth_from_events(os.path.join(game_dir, name + "_RawEventsData.csv"), n)
    dead = dv.dead_prior_from_events(os.path.join(game_dir, name + "_RawEventsData.csv"), n)

    present = ~np.isnan(ball[:, 0])
    if drop_ball > 0:
        rng = np.random.RandomState(seed)
        present = present & (rng.rand(n) > drop_ball)

    scale = np.array(PITCH)
    ball_m = (ball - 0.5) * scale
    ball_m[~present] = np.nan                       # hide ball coords where not present
    home_m = (home - 0.5) * scale
    away_m = (away - 0.5) * scale
    period = np.zeros(n, np.int8)
    for i, (a, b) in enumerate(blocks, 1):
        period[a:b] = i
    m = dict(ball_xy=ball_m, home_xy=home_m, away_xy=away_m, ball_present=present,
             period=period, truth=truth, blocks=blocks, fps=25.0, n=n)

    # heuristic inputs on the SAME present mask (nearest player computed with full ball, then masked)
    team_obs, present_full, conf = dv.tracking_features(ball, home, away, PITCH, drop_ball=0.0)
    team_obs = team_obs.copy(); team_obs[~present] = -1
    conf = conf.copy(); conf[~present] = 0.0
    present_h = present & present_full
    return m, team_obs, present_h, conf, dead


clf = joblib.load(os.path.join(HERE, "possession_model_v2.joblib"))["model"]
print("Validation cross-source Metrica (modele v2 entraine SEULEMENT sur PFF)\n")
for drop in (0.0, 0.41):
    rows = {"baseline": [], "heuristique": [], "modele v2": []}
    for g in GAMES:
        m, team_obs, present_h, conf, dead = load_game(g, drop)
        base = P.smoothed_baseline(team_obs, present_h, m["blocks"], m["n"])
        heur = P.build_labels(team_obs, present_h, dead, m["blocks"], m["n"], PARAMS, conf)
        X, _, _, _ = features.frame_features(m, extra=True)
        proba = apply_bias(clf.predict_proba(X), BIAS)
        codes = decode(smooth_probs(proba, m["blocks"], SMOOTH), m["ball_present"], m["blocks"], m["n"], PARAMS)
        rows["baseline"].append(pm.evaluate(m["truth"], base)["spell_f1_mean"])
        rows["heuristique"].append(pm.evaluate(m["truth"], heur)["spell_f1_mean"])
        rows["modele v2"].append(pm.evaluate(m["truth"], codes)["spell_f1_mean"])
    print("--- regime {} ---".format("ballon complet" if drop == 0 else "ballon masque 41% (comme le defi)"))
    for k in ("baseline", "heuristique", "modele v2"):
        print("  {:<12} spellF1  g1 {:.4f}  g2 {:.4f}  moyenne {:.4f}".format(
            k, rows[k][0], rows[k][1], float(np.mean(rows[k]))))
    print()
