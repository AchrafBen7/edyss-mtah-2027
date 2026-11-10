"""Lightweight MS-TCN as a LEARNED DECODER on the GBM probability stream (run in the torch venv).

Input per frame: [P(H),P(A),P(D), ball_present, conf, obs_roll25] (transfer-stable). The TCN learns
temporal shape / boundary placement and outputs per-frame H/A/D, REPLACING smooth+Viterbi. We compare,
on the SAME prob streams, TCN-decode vs the current Viterbi-decode on PFF held-out + SkillCorner. GO
only if TCN >= Viterbi on BOTH (baseline refs: PFF 0.401, SkillCorner 0.585). Baseline locked meanwhile.

    torch-venv/bin/python tcn_train.py
"""
import glob
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "evaluation"))
import possession_metrics as pm
import possession as P
from eval_model import smooth_probs, decode
from features import _wf

torch.manual_seed(0); np.random.seed(0)
DATA = os.path.join(HERE, "tcn_data")
WIN, CH, NLAY, NSTAGE, EPOCHS, LAM, BS = 2048, 64, 9, 2, 25, 0.15, 16
PARAMS = {**dict(P.PARAMS), "c_switch_team": 8.0}


def load(prefix):
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, prefix + "*.npz"))):
        z = np.load(f)
        out[os.path.basename(f)[:-4]] = dict(inp=z["inp"], truth=z["truth"].astype(np.int64),
                                             present=z["present"], blocks=z["blocks"], fps=float(z["fps"]))
    return out


train, test, sk = load("train_"), load("test_"), load("sk_")


def windows(d):
    xs, ys = [], []
    for s, e in d["blocks"]:
        s, e = int(s), int(e)
        for a in range(s, e, WIN):
            b = min(a + WIN, e)
            x = d["inp"][a:b]; y = d["truth"][a:b]
            if b - a < WIN:
                x = np.vstack([x, np.zeros((WIN - (b - a), x.shape[1]), np.float32)])
                y = np.concatenate([y, np.full(WIN - (b - a), 3, np.int64)])
            xs.append(x); ys.append(y)
    return xs, ys


TX, TY = [], []
for d in train.values():
    xs, ys = windows(d); TX += xs; TY += ys
TX = torch.tensor(np.stack(TX)).transpose(1, 2).float()   # (N,6,WIN)
TY = torch.tensor(np.stack(TY))                           # (N,WIN)
print("train windows:", tuple(TX.shape), flush=True)


class DR(nn.Module):
    def __init__(self, ch, d):
        super().__init__(); self.c = nn.Conv1d(ch, ch, 3, padding=d, dilation=d); self.o = nn.Conv1d(ch, ch, 1)
    def forward(self, x):
        return x + self.o(F.relu(self.c(x)))


class Stage(nn.Module):
    def __init__(self, inc, ch, ncls, nl):
        super().__init__(); self.i = nn.Conv1d(inc, ch, 1)
        self.l = nn.ModuleList([DR(ch, 2 ** k) for k in range(nl)]); self.o = nn.Conv1d(ch, ncls, 1)
    def forward(self, x):
        x = self.i(x)
        for l in self.l:
            x = l(x)
        return self.o(x)


class MSTCN(nn.Module):
    def __init__(self):
        super().__init__(); self.s1 = Stage(6, CH, 3, NLAY)
        self.ss = nn.ModuleList([Stage(3, CH, 3, NLAY) for _ in range(NSTAGE - 1)])
    def forward(self, x):
        o = [self.s1(x)]
        for st in self.ss:
            o.append(st(F.softmax(o[-1], 1)))
        return o


model = MSTCN()
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
ce = nn.CrossEntropyLoss(ignore_index=3)


def smooth_loss(logp):
    return torch.clamp((logp[:, :, 1:] - logp[:, :, :-1]).pow(2), max=16.0).mean()


N = TX.shape[0]
for ep in range(EPOCHS):
    perm = torch.randperm(N); tot = 0.0; model.train()
    for i in range(0, N, BS):
        idx = perm[i:i + BS]; x = TX[idx]; y = TY[idx]
        outs = model(x); loss = 0.0
        for o in outs:
            loss = loss + ce(o.transpose(1, 2).reshape(-1, 3), y.reshape(-1)) + LAM * smooth_loss(F.log_softmax(o, 1))
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    if ep % 5 == 0 or ep == EPOCHS - 1:
        print("epoch {:>2}  loss {:.3f}".format(ep, tot / (N // BS + 1)), flush=True)


def tcn_codes(d):
    model.eval(); n = len(d["truth"]); codes = np.full(n, 3, np.int8)
    with torch.no_grad():
        for s, e in d["blocks"]:
            s, e = int(s), int(e)
            x = torch.tensor(d["inp"][s:e]).transpose(0, 1).unsqueeze(0).float()
            codes[s:e] = model(x)[-1][0].argmax(0).numpy().astype(np.int8)
    return codes


def base_codes(d):
    blocks = [(int(a), int(b)) for a, b in d["blocks"]]
    ps = smooth_probs(d["inp"][:, :3].astype(np.float64), blocks, _wf(60, d["fps"]))
    return decode(ps, d["present"] > 0.5, blocks, len(d["truth"]), PARAMS)


print("\n{:<14}{:>14}{:>10}".format("source", "Viterbi(base)", "TCN"), flush=True)
for nm, ds in [("PFF held-out", test), ("SkillCorner", sk)]:
    b = [pm.evaluate(d["truth"], base_codes(d))["spell_f1_mean"] for d in ds.values()]
    t = [pm.evaluate(d["truth"], tcn_codes(d))["spell_f1_mean"] for d in ds.values()]
    print("{:<14}{:>14.4f}{:>10.4f}".format(nm, np.mean(b), np.mean(t)), flush=True)
print("\nGO si TCN >= Viterbi sur PFF held-out ET SkillCorner.", flush=True)
