"""Load a PFF FC 2022 World Cup match: tracking (teams pre-split) + per-frame possession truth.

PFF tracking is one bzip2-compressed JSONL file per match, 29.97 fps, positions in metres with the
origin at the pitch centre. Teams are already split into homePlayers / awayPlayers, and each frame
carries a sparse game_event (only ~28% of frames). We reconstruct the per-frame possession the way
the events imply it: carry the team of the last on-ball event (OTB) forward until the next event,
OUT turns the ball dead until the next restart, and END closes the match.

    python pff_loader.py 10506          # load one match and print a sanity report
"""

import bz2
import json
import os
import sys

import numpy as np

H, A, D, X = 0, 1, 2, 3
KICKOFFS = {"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "FIFTHKICKOFF"}
HERE = os.path.dirname(os.path.abspath(__file__))


def team_ids(meta_path):
    m = json.load(open(meta_path))[0]
    return str(m["homeTeam"]["id"]), str(m["awayTeam"]["id"]), float(m.get("fps") or 29.97)


def load_match(match_id, pff_dir=None, smoothed=True):
    """Per-frame arrays for one PFF match, plus the graded blocks and fps."""
    pff_dir = pff_dir or os.path.join(HERE, "pff")
    track = os.path.join(pff_dir, "tracking", "{}.jsonl.bz2".format(match_id))
    home_id, away_id, fps = team_ids(os.path.join(pff_dir, "metadata", "{}.json".format(match_id)))
    pk = "homePlayersSmoothed" if smoothed else "homePlayers"
    ak = "awayPlayersSmoothed" if smoothed else "awayPlayers"

    rows = []
    maxf = 0
    with bz2.open(track, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            maxf = max(maxf, int(d["frameNum"]))
            rows.append(d)
    n = maxf + 1

    ball_xy = np.full((n, 2), np.nan)
    ball_present = np.zeros(n, bool)
    home_xy = np.full((n, 11, 2), np.nan)
    away_xy = np.full((n, 11, 2), np.nan)
    period = np.zeros(n, np.int8)
    truth = np.full(n, X, np.int8)

    cur, alive = None, False
    for d in rows:
        fn = int(d["frameNum"])
        period[fn] = d.get("period") or 0
        raw = d.get("balls")
        b = raw[0] if isinstance(raw, list) and raw else (raw if isinstance(raw, dict) and raw else None)
        if b and b.get("x") is not None and b.get("visibility") == "VISIBLE":
            ball_xy[fn] = (b["x"], b["y"])
            ball_present[fn] = True
        for i, p in enumerate((d.get(pk) or [])[:11]):
            if p.get("x") is not None:
                home_xy[fn, i] = (p["x"], p["y"])
        for i, p in enumerate((d.get(ak) or [])[:11]):
            if p.get("x") is not None:
                away_xy[fn, i] = (p["x"], p["y"])

        ge = d.get("game_event") or {}
        t = ge.get("game_event_type")
        if t in KICKOFFS or t == "OTB":
            tid = ge.get("team_id")
            if tid is not None:
                cur = H if str(tid) == home_id else A
                alive = True
        elif t in ("OUT", "END"):
            alive = False
        if cur is not None:
            truth[fn] = cur if alive else D

    graded = truth != X
    blocks, i = [], 0
    while i < n:
        if graded[i]:
            j = i
            while j < n and graded[j]:
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return dict(ball_xy=ball_xy, ball_present=ball_present, home_xy=home_xy, away_xy=away_xy,
                period=period, truth=truth, blocks=blocks, fps=fps, n=n)


def nearest_team(ball_xy, home_xy, away_xy):
    """team_obs (0/1/-1) and confidence from the nearest player of each team to the ball."""
    bx, by = ball_xy[:, 0], ball_xy[:, 1]
    def mind(group):
        d = (group[:, :, 0] - bx[:, None]) ** 2 + (group[:, :, 1] - by[:, None]) ** 2
        with np.errstate(invalid="ignore"):
            return np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    dh, da = mind(home_xy), mind(away_xy)
    ok = np.isfinite(dh) & np.isfinite(da) & ~np.isnan(bx)
    team = np.full(len(bx), -1, np.int8)
    team[ok] = np.where(dh[ok] <= da[ok], 0, 1)
    conf = np.zeros(len(bx))
    sh, sa = np.sqrt(dh[ok]), np.sqrt(da[ok])
    conf[ok] = np.abs(sh - sa) / np.maximum(sh + sa, 1e-9)
    return team, (np.isnan(bx) == False) & ok, conf


if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else "10506"
    m = load_match(mid)
    n = m["n"]
    pct = lambda k: round(100.0 * np.mean(m["truth"] == k), 1)
    print("match {}: {} frames, {} fps, {} blocs".format(mid, n, m["fps"], len(m["blocks"])))
    print("  verite  H {}%  A {}%  D {}%  X {}%".format(pct(H), pct(A), pct(D), pct(X)))
    print("  ballon visible: {}%".format(round(100.0 * m["ball_present"].mean(), 1)))
    team, present, conf = nearest_team(m["ball_xy"], m["home_xy"], m["away_xy"])
    live = present & ((m["truth"] == H) | (m["truth"] == A))
    agree = np.mean(team[live] == m["truth"][live])
    print("  accord 'joueur le plus proche' vs verite (images vivantes, ballon visible): "
          "{}%".format(round(100.0 * agree, 1)))
    print("  confiance moyenne sur ces images: {}".format(round(float(conf[live].mean()), 3)))
