"""Apply the learned model to the real challenge match (BAR-ATH) and write a submission.

The model was trained on PFF tracking in metres with the origin at the pitch centre. The challenge
ATD is in 0-1 pitch fractions, so we centre and scale it to the same convention before extracting
the identical features. Then: model -> probabilities -> smoothing -> Viterbi -> edyss.csv.

    python predict_challenge.py                              # -> edyss.csv (the submitted file)
    python predict_challenge.py <model.joblib> <out.csv>     # explicit model and output
"""

import os
import sys
import warnings

import numpy as np
import joblib
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import possession as P
import features
from eval_model import smooth_probs, decode, W_EMIT  # noqa: F401 (reuse the exact decoder)
from calibrate_bias import apply_bias
from games import GAME, GAMES, half_video_blocks

HERE = os.path.dirname(os.path.abspath(__file__))
FIELD_SEP, GROUP_SEP, MISSING = ":", ";", "N"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "possession_model_final.joblib"  # arg1 overrides; 61 PFF + 2 Metrica
EXTRA = True                          # motion/momentum/control features
PITCH_CONTROL = False                 # OFF: pitch control helped PFF/Metrica held-out but was WORSE on the
#                                       blind challenge (0.430 -> 0.415); velocity features don't transfer
#                                       to the challenge's tracking regime. Kept in features.py, off here.
HOME_BIAS = 0.25                     # moderate, held-out-validated; capped below the 0.5 max for transfer safety
SMOOTH = 50                          # 2.0s at the challenge's 25 fps (= PFF held-out optimum 2.0s, fps-consistent)
C_SWITCH_TEAM = 8.0                  # best on the 45-match sweep
W_TEAM = 0.0                         # OFF: phase TEAM prior hurt the 5-min (0.430->0.415) and is not
#                                      validatable on PFF/SkillCorner (no analogous phase tags); the
#                                      ~34% wrong-team rate on contested frames costs more than it adds.
POSS_CODES = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "BALL IN THE BOX",
              "CREATING CHANCES", "LONG BALL", "ATTACKING TRANSITION"}        # tagged team HAS the ball
OPP_CODES = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"}         # tagged team DEFENDS -> other has it
DEAD_W = 0.0                         # OFF: helped on Metrica (narrow 2.7% dead) but the challenge's
#                                      Smart Tagging SET PIECES cover 11.6%, so w=2 over-forces D and
#                                      hurts (0.429 -> 0.371); the weight can't be tuned on the 5 min.


def parse_all_positions(raw_path, team_of_track, n, pitch):
    """Per frame: ball xy and every player's xy, CENTRED to metres like PFF (origin = pitch centre)."""
    length, width = pitch
    k = len(team_of_track)
    ball = np.full((n, 2), np.nan)
    present = np.zeros(n, bool)
    home = np.full((n, k, 2), np.nan)
    away = np.full((n, k, 2), np.nan)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            fr = int(parts[0]) - 1
            b = parts[2]
            if b[0] != MISSING:
                c = b.index(",")
                ball[fr] = ((float(b[:c]) - 0.5) * length,
                            (float(b[c + 1:b.index(",", c + 1)]) - 0.5) * width)
                present[fr] = True
            for slot, g in enumerate(parts[1].split(GROUP_SEP)):
                if g[0] == MISSING:
                    continue
                c = g.index(",")
                x = (float(g[:c]) - 0.5) * length
                y = (float(g[c + 1:g.index(",", c + 1)]) - 0.5) * width
                (home if team_of_track[slot] == 0 else away)[fr, slot] = (x, y)
    return ball, present, home, away


def boost_dead(probs, dead, w):
    """Raise P(D) on Smart-Tagging dead frames, after smoothing and before the Viterbi."""
    if w <= 0:
        return probs
    lp = np.log(np.clip(probs, 1e-9, 1.0))
    lp[dead, 2] += w
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(1, keepdims=True)


def team_prior_from_tagging(tagging_path, n, fps=25.0):
    """Per-frame team prior from Smart-Tagging phases: +1 home (Barcelona), -1 away (Atletic), 0 none.
    DEFENDING/DEFENSIVE-TRANSITION phases tag the team WITHOUT the ball, so the possessor is flipped."""
    votes = np.zeros(n)
    for inst in ET.parse(tagging_path).iter("instance"):
        code = (inst.findtext("code") or "").strip().upper()
        labs = [(l.findtext("text") or "").strip() for l in inst.findall("label")]
        team = labs[0] if labs else ""
        t = 1 if team == "Barcelona" else (-1 if team == "Atletic Club" else 0)
        if t == 0:
            continue
        if code in POSS_CODES:
            poss = t
        elif code in OPP_CODES:
            poss = -t
        else:
            continue
        s = int(round(float(inst.findtext("start")) * fps))
        e = int(round(float(inst.findtext("end")) * fps))
        votes[max(0, s):min(n, e + 1)] += poss
    return np.sign(votes)


def second_half_dead_window(tagging_path, game, fps=25.0):
    """(graded_start, kickoff) video-frames of the 2nd half: graded but BEFORE the kickoff, so the
    ball is dead there by definition and those frames must be D. Derived from the graded block
    boundary + the Smart Tagging 'Kick Off (Start)' 2nd-half event (not hard-coded). None if absent."""
    halves = sorted(half_video_blocks(game))
    if len(halves) < 2:
        return None
    start = halves[1][1]                                # first graded video-frame of the 2nd half
    for inst in ET.parse(tagging_path).iter("instance"):
        labels = [(l.findtext("text") or "").lower() for l in inst.findall("label")]
        if any("kick off (start)" in x for x in labels) and any("2nd" in x for x in labels):
            kickoff = int(round(float(inst.findtext("start")) * fps))
            return (start, kickoff) if kickoff > start else None
    return None


if __name__ == "__main__":
    pitch = GAMES[GAME]["pitch"]
    n = GAMES[GAME]["n_video_frames"]
    blocks = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    tac = os.path.join(HERE, "data", "tactical")
    meta = os.path.join(tac, "{}_tactical_atd.xml".format(GAME))
    raw = os.path.join(tac, "{}_tactical_atd.txt".format(GAME))
    tagging = os.path.join(tac, "{}_tactical_smart_tagging.xml".format(GAME))

    print("lecture + centrage des coordonnees ...", flush=True)
    team_of_track = P.team_of_each_track(meta)
    ball, present, home, away = parse_all_positions(raw, team_of_track, n, pitch)
    dead = P.dead_prior_from_tagging(tagging, n, fps=25)

    period = np.zeros(n, np.int8)
    for i, (a, b) in enumerate(blocks, 1):
        period[a:b] = i
    m = dict(ball_xy=ball, home_xy=home, away_xy=away, ball_present=present,
             period=period, truth=np.full(n, P.X, np.int8), fps=25.0, n=n)

    X, _, _, _ = features.frame_features(m, extra=EXTRA, pitch_control=PITCH_CONTROL)
    bundle = joblib.load(os.path.join(HERE, MODEL))
    probs = apply_bias(bundle["model"].predict_proba(X), HOME_BIAS)   # columns H, A, D
    print("modele {} applique, lissage {} + Viterbi ...".format(MODEL, SMOOTH), flush=True)

    probs_s = boost_dead(smooth_probs(probs, blocks, SMOOTH), dead, DEAD_W)
    if W_TEAM > 0:
        tp = team_prior_from_tagging(tagging, n)
        lp = np.log(np.clip(probs_s, 1e-9, 1.0))
        lp[tp > 0, 0] += W_TEAM                      # boost home where a phase says home possesses
        lp[tp < 0, 1] += W_TEAM                      # boost away where a phase says away possesses
        lp -= lp.max(1, keepdims=True); ex = np.exp(lp); probs_s = ex / ex.sum(1, keepdims=True)
        print("prior equipe Smart Tagging (w={}): H+ {} frames, A+ {} frames".format(
            W_TEAM, int((tp > 0).sum()), int((tp < 0).sum())), flush=True)
    codes = decode(probs_s, present, blocks, n, {**dict(P.PARAMS), "c_switch_team": C_SWITCH_TEAM})

    dw = second_half_dead_window(tagging, GAME)         # graded-but-pre-kickoff 2nd half is dead
    if dw is not None:
        codes[dw[0]:dw[1]] = P.D
        print("fenetre morte 2e mi-temps [{}, {}) forcee D ({} frames)".format(dw[0], dw[1], dw[1] - dw[0]), flush=True)

    out = os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "edyss.csv")
    with open(out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, P.LETTER[c]) for i, c in enumerate(codes))
    counts = {P.LETTER[k]: int(np.sum(codes == k)) for k in range(4)}
    print("ecrit {}  ({})".format(out, counts))
