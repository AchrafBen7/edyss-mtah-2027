"""possession.py -- our More than a Hack 2027 Phase 1 pipeline.

Steps (see WRITEUP.md for the why):
  1. Per-frame signal from tracking: team of the player nearest the ball, whether the
     ball is tracked at all. (Ball-missing frames carry no team evidence.)
  2. Dead-ball prior from Smart Tagging set pieces (+ hook for ball dynamics).
  3. 3-state Viterbi over H/A/D: coherent possessions instead of a flickering vote.
  4. X-abstention: blank ball-missing frames in the interior of a possession. Free on
     spellF1 (the scorer forward-fills them), and it stops noise fragmenting a spell.
  5. Write a frame,label submission.

Runs with numpy only, the same environment as the scorer, by parsing the raw ATD the
way the reference baseline does.

    python possession.py                 # build ./submission.csv from ./data
    python possession.py --selftest      # prove the pipeline on synthetic data, no download

The selftest builds a synthetic match with a known truth, a noisy observation of it
(15% wrong team, 41% of frames with no ball, random team during stoppages, and imperfect
set-piece tags), then scores our pipeline against the plain smoothed baseline with the
official metric. It asserts we beat the baseline on spellF1.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "evaluation"))
sys.path.insert(0, HERE)
import possession_metrics as pm  # noqa: E402

H, A, D, X = 0, 1, 2, 3
LETTER = np.array(["H", "A", "D", "X"])

# -- tunable parameters (tune on sample-data, never on the public 5 minutes) -------------------
PARAMS = {
    "w_obs": 2.0,          # emission weight for the confidence-weighted team vote
    "smooth_window": 25,   # half-width of the vote (smaller: the confidence weight already denoises)
    "w_dead": 3.0,         # emission bonus for D inside a set-piece (dead-ball) prior
    "w_absent_dead": 0.0,  # D bonus for an untracked ball: 0, absence alone is not a stoppage
    "c_switch_team": 8.0,  # transition penalty H<->A (kills flicker, routes changes via D)
    "c_switch_dead": 3.5,  # transition penalty to/from D (so only real stoppages form a D run)
    "abstain_margin": 10,  # keep a committed label within this many frames of a boundary
}


# -- tracking (raw ATD, numpy only) ------------------------------------------------------------

FIELD_SEP, GROUP_SEP, MISSING = ":", ";", "N"


def team_of_each_track(meta_path):
    """Array over track slots: 0 home, 1 away. First Team in the metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([0 if tid == home_id else 1 for tid in slots], dtype=np.int8)


def nearest_team_and_ball(raw_path, team_of_track, n_frames, pitch):
    """Per frame: nearest-player team (-1 if none), ball-present flag, ball xy (0-1), confidence.

    confidence is the normalised gap between the nearest home and nearest away player,
    |d_home - d_away| / (d_home + d_away): ~1 when one team clearly holds the ball, ~0 when two
    opponents are equidistant (a contested or loose ball). The emission weights the vote by it.
    """
    length, width = pitch
    team_obs = np.full(n_frames, -1, dtype=np.int8)
    ball_present = np.zeros(n_frames, dtype=bool)
    ball_xy = np.full((n_frames, 2), np.nan)
    conf = np.zeros(n_frames)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # file counts from 1, submissions from 0
            ball = parts[2]
            if ball[0] == MISSING:
                continue
            comma = ball.index(",")
            fx = float(ball[:comma])
            fy = float(ball[comma + 1:ball.index(",", comma + 1)])
            ball_xy[frame] = (fx, fy)
            ball_x, ball_y = fx * length, fy * width
            ball_present[frame] = True
            best_h, best_a = float("inf"), float("inf")
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                comma = group.index(",")
                dx = float(group[:comma]) * length - ball_x
                dy = float(group[comma + 1:group.index(",", comma + 1)]) * width - ball_y
                distance = dx * dx + dy * dy
                if team_of_track[slot] == 0:
                    if distance < best_h:
                        best_h = distance
                elif distance < best_a:
                    best_a = distance
            if best_h == float("inf") and best_a == float("inf"):
                continue
            if best_h == float("inf"):
                team_obs[frame], conf[frame] = 1, 1.0
            elif best_a == float("inf"):
                team_obs[frame], conf[frame] = 0, 1.0
            else:
                dh, da = best_h ** 0.5, best_a ** 0.5
                team_obs[frame] = 0 if dh <= da else 1
                conf[frame] = abs(dh - da) / (dh + da)
    return team_obs, ball_present, ball_xy, conf


def ball_out_mask(ball_xy, ball_present):
    """Frames where the tracked ball is clearly outside the pitch rectangle: a dead-ball signal.

    A throw-in, corner, goal kick or kick-off is preceded by the ball crossing a line, so this
    marks the START of the stoppage, where the Smart Tagging set piece marks the restart at its
    END. Together they bracket the dead-ball window, which sharpens possession boundaries.
    """
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    return ball_present & ((x < 0.0) | (x > 1.0) | (y < 0.0) | (y > 1.0))


# -- dead-ball prior from Smart Tagging --------------------------------------------------------

def dead_prior_from_tagging(xml_path, n_frames, fps=25):
    """Boolean mask: frames likely dead, from the lead-in of each SET PIECES interval.

    A set piece is a restart at the END of a stoppage, so the dead time is just before it.
    The tag brackets the restart in a ~5 s window, so we mark the first half of each
    set-piece interval as a dead prior. Noisy on purpose -- it is a prior, not a label.
    """
    prior = np.zeros(n_frames, dtype=bool)
    for instance in ET.parse(xml_path).iter("instance"):
        if (instance.findtext("code") or "").strip().lower() != "set pieces":
            continue
        start = int(round(float(instance.findtext("start")) * fps))
        end = int(round(float(instance.findtext("end")) * fps))
        mid = (start + end) // 2
        prior[max(0, start):max(0, mid)] = True
    return prior


# -- model -------------------------------------------------------------------------------------

def build_emissions(team_obs, ball_present, dead_prior, p, conf=None):
    """(n, 3) emission scores for H, A, D.

    The team evidence is a WINDOWED vote of the nearest-player signal, weighted by control
    confidence. On real data the player closest to the ball is often an opponent for a few frames,
    so a raw per-frame emission makes Viterbi chase noise and shatter possessions (our v1 bug: 90
    predicted possessions for 17 real). Smoothing denoises it the way the baseline does, confidence
    down-weights contested frames, and Viterbi adds the dead-ball structure and spell coherence on
    top. conf=None falls back to an unweighted vote (used by the synthetic selftest).
    """
    n = len(team_obs)
    if conf is None:
        conf = np.ones(n)
    w = p["smooth_window"]
    home = np.where(ball_present & (team_obs == 0), conf, 0.0)
    away = np.where(ball_present & (team_obs == 1), conf, 0.0)
    hc = np.concatenate(([0.0], np.cumsum(home)))
    ac = np.concatenate(([0.0], np.cumsum(away)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - w)
    hi = np.minimum(n, idx + w + 1)
    h = hc[hi] - hc[lo]
    a = ac[hi] - ac[lo]
    total = h + a
    home_frac = np.where(total > 0, h / np.maximum(total, 1e-9), 0.5)

    emis = np.zeros((n, 3), dtype=np.float64)
    seen = total > 0
    emis[seen, H] += p["w_obs"] * home_frac[seen]
    emis[seen, A] += p["w_obs"] * (1.0 - home_frac[seen])
    emis[dead_prior, D] += p["w_dead"]
    emis[~ball_present, D] += p["w_absent_dead"]
    return emis


def build_transitions(p):
    """(3, 3) log transition scores, row = from, col = to."""
    ct, cd = p["c_switch_team"], p["c_switch_dead"]
    return np.array([[0.0, -ct, -cd],
                     [-ct, 0.0, -cd],
                     [-cd, -cd, 0.0]], dtype=np.float64)


def viterbi(emis, trans):
    """Most likely H/A/D path for one contiguous block."""
    t_len = emis.shape[0]
    if t_len == 0:
        return np.empty(0, dtype=np.int8)
    dp = emis[0].copy()
    back = np.zeros((t_len, 3), dtype=np.int8)
    for t in range(1, t_len):
        scores = dp[:, None] + trans            # (from, to)
        best_from = np.argmax(scores, axis=0)   # (3,)
        dp = scores[best_from, np.arange(3)] + emis[t]
        back[t] = best_from
    path = np.zeros(t_len, dtype=np.int8)
    path[-1] = int(np.argmax(dp))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def add_abstentions(path, ball_present, margin):
    """Set X on ball-missing frames in the interior of an H or A run (not near a boundary)."""
    out = path.copy()
    n = len(path)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = path[1:] != path[:-1]
    boundary_idx = np.flatnonzero(change)
    # distance to the previous and next boundary
    dist_prev = np.arange(n) - np.maximum.accumulate(np.where(change, np.arange(n), 0))
    nxt = np.where(change, np.arange(n), n)
    dist_next = np.minimum.accumulate(nxt[::-1])[::-1] - np.arange(n)
    interior = (np.minimum(dist_prev, dist_next) >= margin)
    mask = interior & (~ball_present) & ((path == H) | (path == A))
    out[mask] = X
    return out


def build_labels(team_obs, ball_present, dead_prior, blocks, n_frames, p, conf=None):
    """Full-length H/A/D/X code array: Viterbi inside each graded block, X elsewhere."""
    emis = build_emissions(team_obs, ball_present, dead_prior, p, conf)
    trans = build_transitions(p)
    codes = np.full(n_frames, X, dtype=np.int8)
    for start, stop in blocks:
        path = viterbi(emis[start:stop], trans)
        path = add_abstentions(path, ball_present[start:stop], p["abstain_margin"])
        codes[start:stop] = path
    return codes


def smoothed_baseline(team_obs, ball_present, blocks, n_frames, window=50):
    """The reference baseline for comparison: majority vote of the nearest-team signal."""
    codes = np.where(ball_present, team_obs, -1).astype(np.int64)
    out = np.full(n_frames, D, dtype=np.int8)
    home_cum = np.concatenate(([0], np.cumsum(codes == 0)))
    away_cum = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(n_frames)
    low = np.maximum(0, idx - window)
    high = np.minimum(n_frames, idx + window + 1)
    home = home_cum[high] - home_cum[low]
    away = away_cum[high] - away_cum[low]
    seen = (home + away) > 0
    out[seen & (home >= away)] = H
    out[seen & (home < away)] = A
    mask = np.ones(n_frames, dtype=bool)          # outside graded blocks -> X
    for start, stop in blocks:
        mask[start:stop] = False
    out[mask] = X
    return out


# -- real-data entry point ---------------------------------------------------------------------

def build_submission(data_dir, out_path, p=PARAMS):
    from games import GAME, GAMES, half_video_blocks  # local import: needs the repo root
    pitch = GAMES[GAME]["pitch"]
    n_frames = GAMES[GAME]["n_video_frames"]
    blocks = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]  # half-open

    tac = os.path.join(data_dir, "tactical")
    meta = os.path.join(tac, "{}_tactical_atd.xml".format(GAME))
    raw = os.path.join(tac, "{}_tactical_atd.txt".format(GAME))
    tagging = os.path.join(tac, "{}_tactical_smart_tagging.xml".format(GAME))

    print("reading tracking ...")
    team_obs, ball_present, ball_xy, conf = nearest_team_and_ball(
        raw, team_of_each_track(meta), n_frames, pitch)
    print("  ball located on {:.1f}% of frames".format(100.0 * ball_present.mean()))
    dead_prior = dead_prior_from_tagging(tagging, n_frames)
    print("  set-piece dead prior on {} frames".format(int(dead_prior.sum())))
    # ball-out-of-bounds (ball_out_mask) was tested as an extra D signal and was neutral on the
    # sample-data dev set, so it is not used. Kept available for future feature work.

    codes = build_labels(team_obs, ball_present, dead_prior, blocks, n_frames, p, conf)
    with open(out_path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, LETTER[c]) for i, c in enumerate(codes))
    counts = {LETTER[k]: int(np.sum(codes == k)) for k in range(4)}
    print("wrote {}  ({})".format(out_path, counts))
    print("validate it:  python evaluation/validate_submission.py {}".format(out_path))


# -- selftest (no download needed) -------------------------------------------------------------

def _synthetic(seed):
    """A synthetic match: truth, two graded halves, and noisy set-piece tags."""
    rng = np.random.RandomState(seed)
    blocks = [(11185, 90652), (110808, 190198)]       # the real half-open graded ranges
    n = 199319
    gt = np.full(n, X, dtype=np.int8)
    setpieces = []
    for start, stop in blocks:
        t, team = start, rng.randint(0, 2)
        while t < stop:
            length = rng.randint(50, 400)             # 2-16 s possession
            end = min(stop, t + length)
            gt[t:end] = H if team == 0 else A
            t = end
            if t < stop and rng.rand() < 0.5:         # stoppage + set-piece restart
                dead = min(stop, t + rng.randint(25, 150))
                gt[t:dead] = D
                setpieces.append((max(start, dead - 60), min(stop, dead + 60)))
                t, team = dead, rng.randint(0, 2)
            else:
                team ^= 1                             # live turnover
    return n, gt, blocks, setpieces


def _observe(gt, rng, flip=0.15, absent=0.41):
    """A noisy nearest-team / ball-present observation of the truth."""
    n = len(gt)
    graded = gt != X
    present = graded & (rng.rand(n) > absent)
    team_obs = np.full(n, -1, dtype=np.int8)
    alive = present & ((gt == H) | (gt == A))
    true_team = (gt == A).astype(np.int8)                     # 0 for H, 1 for A
    flipped = np.where(rng.rand(n) < flip, 1 - true_team, true_team)
    team_obs[alive] = flipped[alive]
    deadp = present & (gt == D)                               # random team near a dead ball
    team_obs[deadp] = rng.randint(0, 2, size=n)[deadp]
    return team_obs, present


def _tag_prior(n, setpieces, rng, miss=0.15):
    prior = np.zeros(n, dtype=bool)
    for start, end in setpieces:
        if rng.rand() < miss:
            continue
        prior[start:(start + end) // 2] = True
    return prior


def selftest(seed=0):
    n, gt, blocks, setpieces = _synthetic(seed)
    rng = np.random.RandomState(seed + 1)
    team_obs, ball_present = _observe(gt, rng)
    dead_prior = _tag_prior(n, setpieces, rng)

    base = smoothed_baseline(team_obs, ball_present, blocks, n)
    ours = build_labels(team_obs, ball_present, dead_prior, blocks, n, PARAMS)

    base_r = pm.evaluate(gt, base)
    ours_r = pm.evaluate(gt, ours)
    truth_spells = base_r["n_gt_spells"]

    print("synthetic match: {} graded possessions, mean {:.1f} s\n".format(
        truth_spells, base_r["gt_mean_spell_duration_s"]))
    row = "{:<28} spellF1 {:>7.4f}   macroF1 {:>7.4f}   poss {:>4} (truth {})   frag {:>5.2f}"
    print(row.format("smoothed baseline", base_r["spell_f1_mean"], base_r["macro_f1"],
                     base_r["n_pred_spells"], truth_spells, base_r["fragmentation"]))
    print(row.format("ours (D-prior + Viterbi + X)", ours_r["spell_f1_mean"], ours_r["macro_f1"],
                     ours_r["n_pred_spells"], truth_spells, ours_r["fragmentation"]))

    gain = ours_r["spell_f1_mean"] - base_r["spell_f1_mean"]
    print("\nspellF1 gain over the baseline: {:+.4f}".format(gain))
    assert ours_r["spell_f1_mean"] > base_r["spell_f1_mean"], "pipeline did not beat the baseline"
    print("OK -- the pipeline beats the smoothed baseline on the ranking metric.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true", help="run on synthetic data, no download")
    parser.add_argument("--data", default=os.path.join(HERE, "data"), help="data directory")
    parser.add_argument("--out", default=os.path.join(HERE, "submission.csv"), help="output CSV")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        build_submission(args.data, args.out)
