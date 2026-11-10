"""Ingest PFF matches from ~/Downloads into pff/{tracking,metadata}, carefully.

Training needs only tracking (.jsonl.bz2) + metadata (small .json with homeTeam/awayTeam). Events
and rosters are ignored. Duplicates (same id in several folders) are de-duped. Each metadata is
verified by CONTENT, so a big events .json is never mistaken for metadata. Dry-run by default.

    python ingest.py          # report only
    python ingest.py --apply  # actually copy the new, complete matches
"""
import glob
import json
import os
import shutil
import sys

DL = os.path.join(os.path.expanduser("~"), "Downloads")
HERE = os.path.dirname(os.path.abspath(__file__))
TRK = os.path.join(HERE, "pff", "tracking")
MET = os.path.join(HERE, "pff", "metadata")
APPLY = "--apply" in sys.argv


def tid(path):
    return os.path.basename(path).split(".")[0]


trk = {}
for p in glob.glob(os.path.join(DL, "**", "*.jsonl.bz2"), recursive=True):
    i = tid(p)
    if i.isdigit():
        trk.setdefault(i, []).append(p)

met = {}
for p in glob.glob(os.path.join(DL, "**", "*.json"), recursive=True):
    if p.endswith(".jsonl") or os.path.getsize(p) > 1_000_000:   # events are tens of MB
        continue
    i = tid(p)
    if not i.isdigit():
        continue
    try:
        d = json.load(open(p))
        d0 = d[0] if isinstance(d, list) and d else d
        if isinstance(d0, dict) and "homeTeam" in d0 and "awayTeam" in d0:
            met.setdefault(i, []).append(p)
    except Exception:
        pass

have = set(tid(f) for f in glob.glob(os.path.join(TRK, "*.jsonl.bz2")))
complete = sorted(set(trk) & set(met), key=int)
new_complete = [i for i in complete if i not in have]
trk_only = sorted(set(trk) - set(met), key=int)
met_only = sorted(set(met) - set(trk), key=int)
dups = {i: len(ps) for i, ps in trk.items() if len(ps) > 1}

print("tracking ids: {}  | metadata ids: {}".format(len(trk), len(met)))
print("deja dans pff/ ({}): {}".format(len(have), sorted(have, key=int)))
print("\nCOMPLETS NOUVEAUX (tracking+metadata, {}): {}".format(len(new_complete), new_complete))
print("\ntracking SANS metadata (ignores, {}): {}".format(len(trk_only), trk_only))
print("metadata SANS tracking (ignores, {}): {}".format(len(met_only), met_only))
print("tracking en double, on garde 1 ({}): {}".format(len(dups), dups))

if APPLY:
    os.makedirs(TRK, exist_ok=True)
    os.makedirs(MET, exist_ok=True)
    for i in new_complete:
        shutil.copy2(sorted(trk[i])[0], os.path.join(TRK, i + ".jsonl.bz2"))
        shutil.copy2(sorted(met[i])[0], os.path.join(MET, i + ".json"))
    print("\nAJOUTE {} matchs dans pff/.  Total tracking: {}".format(
        len(new_complete), len(glob.glob(os.path.join(TRK, "*.jsonl.bz2")))))
else:
    print("\n(dry-run ; relancer avec --apply pour copier)")
