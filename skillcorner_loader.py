"""Load a SkillCorner open-data broadcast-tracking match -> m dict + per-frame possession truth.

SkillCorner is REAL broadcast tracking with genuine off-camera player dropout (per-object is_detected),
~10 fps, metres pitch-centred (same convention as PFF). Possession truth comes from phases_of_play.csv
(team_in_possession_id over [frame_start, frame_end]); gaps inside a period's play span are D. This is
our transfer-HONEST validation set: its sparsity is produced by a camera, not by our masking code, so a
lever that survives here is real evidence, not the circular masked-Metrica trap.

    python skillcorner_loader.py <match_dir>      # sanity report
"""
import csv
import json
import os
import sys

import numpy as np

H, A, D, X = 0, 1, 2, 3
from data_paths import SKILLCORNER_DATA as DEFAULT_DIR  # see data_paths.py
K = 14                                              # max tracked players per team per frame


def load_match(match_dir, mid=None, fps=10.0, detected_only=False):
    mid = mid or os.path.basename(match_dir.rstrip("/"))
    meta = json.load(open(os.path.join(match_dir, "{}_match.json".format(mid))))
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    pitch = (float(meta.get("pitch_length") or 105), float(meta.get("pitch_width") or 68))

    team_of = {}                                    # tracking player_id -> 0 home / 1 away
    for p in meta.get("players", []):
        t = 0 if p.get("team_id") == home_id else (1 if p.get("team_id") == away_id else -1)
        if t < 0:
            continue
        for key in ("trackable_object", "id", "team_player_id"):
            if p.get(key) is not None:
                team_of[p[key]] = t

    rows = []
    maxf = 0
    with open(os.path.join(match_dir, "{}_tracking_extrapolated.jsonl".format(mid))) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("frame") is None:
                continue
            rows.append(r); maxf = max(maxf, int(r["frame"]))
    n = maxf + 1

    ball_xy = np.full((n, 2), np.nan)
    ball_present = np.zeros(n, bool)
    home_xy = np.full((n, K, 2), np.nan)
    away_xy = np.full((n, K, 2), np.nan)
    period = np.zeros(n, np.int8)
    for r in rows:
        f = int(r["frame"]); period[f] = r.get("period") or 0
        b = r.get("ball_data") or {}
        if b.get("x") is not None and b.get("is_detected"):
            ball_xy[f] = (b["x"], b["y"]); ball_present[f] = True
        hc = ac = 0
        for pl in (r.get("player_data") or []):
            if pl.get("x") is None or (detected_only and not pl.get("is_detected")):
                continue
            t = team_of.get(pl.get("player_id"), -1)
            if t == 0 and hc < K:
                home_xy[f, hc] = (pl["x"], pl["y"]); hc += 1
            elif t == 1 and ac < K:
                away_xy[f, ac] = (pl["x"], pl["y"]); ac += 1

    truth = np.full(n, X, np.int8)
    phases = []
    with open(os.path.join(match_dir, "{}_phases_of_play.csv".format(mid))) as fh:
        for row in csv.DictReader(fh):
            tip = (row.get("team_in_possession_id") or "").strip()
            if not tip or tip.lower() == "none":
                continue
            fs, fe, per = int(row["frame_start"]), int(row["frame_end"]), int(row["period"])
            lab = H if int(tip) == home_id else (A if int(tip) == away_id else D)
            truth[fs:fe + 1] = lab
            phases.append((per, fs, fe))

    # inside each period's play span, frames not covered by a phase are dead (D), not ungraded (X)
    for per in sorted(set(p[0] for p in phases)):
        fr = [ph for ph in phases if ph[0] == per]
        lo = min(f for _, f, _ in fr); hi = max(e for _, _, e in fr)
        span = truth[lo:hi + 1]
        span[span == X] = D
        truth[lo:hi + 1] = span

    graded = truth != X
    blocks, i = [], 0
    while i < n:
        if graded[i]:
            j = i
            while j < n and graded[j]:
                j += 1
            blocks.append((i, j)); i = j
        else:
            i += 1
    return dict(ball_xy=ball_xy, home_xy=home_xy, away_xy=away_xy, ball_present=ball_present,
                period=period, truth=truth, blocks=blocks, fps=fps, n=n, pitch=pitch)


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DEFAULT_DIR, "1874553")
    m = load_match(md)
    n = m["n"]
    pct = lambda k: round(100.0 * np.mean(m["truth"] == k), 1)
    nh = np.sum(~np.isnan(m["home_xy"][:, :, 0]), axis=1)
    na = np.sum(~np.isnan(m["away_xy"][:, :, 0]), axis=1)
    g = m["truth"] != X
    print("match {}: {} frames, {} fps, {} blocs gradés".format(os.path.basename(md), n, m["fps"], len(m["blocks"])))
    print("  verite  H {}%  A {}%  D {}%  X {}%".format(pct(H), pct(A), pct(D), pct(X)))
    print("  joueurs/equipe (gradé): home {:.1f}  away {:.1f}".format(nh[g].mean(), na[g].mean()))
    print("  ballon present (gradé): {:.0f}%".format(100 * m["ball_present"][g].mean()))
