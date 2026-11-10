"""Measure the input regime: players tracked per frame and ball presence, challenge vs PFF vs Metrica.

If the challenge tracks far fewer players per frame than PFF (off-camera dropouts), that domain gap is
a prime suspect for the transfer wall, and masking PFF players to match it is a clean, compliant fix.
"""
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import possession as P
import pff_loader
import predict_challenge as pc            # parse_all_positions; its main is guarded
from games import GAME, GAMES
import dev_sampledata as dv

HERE = pc.HERE
from data_paths import METRICA_SAMPLE_DATA as DATA  # see data_paths.py


def summarize(tag, home, away, present):
    nh = np.sum(~np.isnan(home[:, :, 0]), axis=1)
    na = np.sum(~np.isnan(away[:, :, 0]), axis=1)
    print("{:<12} home {:.1f}/frame  away {:.1f}/frame  | ballon present {:.0f}%  "
          "| frames <=10 home {:.0f}%  <=8 away {:.0f}%".format(
              tag, nh.mean(), na.mean(), 100 * present.mean(),
              100 * np.mean(nh <= 10), 100 * np.mean(na <= 8)), flush=True)


# --- Challenge ---
pitch = GAMES[GAME]["pitch"]
n = GAMES[GAME]["n_video_frames"]
tac = os.path.join(HERE, "data", "tactical")
tot = P.team_of_each_track(os.path.join(tac, "{}_tactical_atd.xml".format(GAME)))
ball, present, home, away = pc.parse_all_positions(os.path.join(tac, "{}_tactical_atd.txt".format(GAME)),
                                                   tot, n, pitch)
print("tracks totaux cote defi:", len(tot), flush=True)
summarize("DEFI", home, away, present)

# --- PFF ---
m = pff_loader.load_match("10515")
summarize("PFF 10515", m["home_xy"], m["away_xy"], m["ball_present"])

# --- Metrica ---
g = os.path.join(DATA, "Sample_Game_1")
nn, b, h, a = dv.load_tracking(os.path.join(g, "Sample_Game_1_RawTrackingData_Home_Team.csv"),
                               os.path.join(g, "Sample_Game_1_RawTrackingData_Away_Team.csv"))
summarize("Metrica G1", h, a, ~np.isnan(b[:, 0]))
