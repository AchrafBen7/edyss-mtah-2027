# Possession classification: method, validation, reproducibility

**Team edyss** · Phase 1 · match BAR-ATH (199,319 frames, labels H / A / D / X).

Our submission uses a gradient-boosting classifier trained only on external tracking data: PFF FC
World Cup 2022 and two Metrica Sports sample games. It outputs per-frame probabilities, which are
smoothed and decoded into possessions by a 3-state Viterbi.

- **Our reliable figure is 0.401 spellF1 on three held-out PFF matches.**
- **It transfers to another provider and league:** 0.585 on SkillCorner, scored against that
  dataset's own possession definition.
- **On the public 5-minute window it scores 0.430**, against 0.326 for the official baseline on the
  same window. That window holds 17 possessions and cannot rank anything: `baseline_smooth125` scores
  0.424 there, yet loses to the 0.2433 floor of `baseline_smooth50` on the full match.

This note covers the method, the data, the possession rules and the validation. Section 6 includes
how we used the public window. We also list the levers we rejected and the limitations.

## 1. The metric drives every choice

The ranking metric is **spellF1**, not per-frame accuracy.
- A *possession* is a maximal run of H or of A. D only separates possessions and is never one.
- Predicted possessions are matched one-to-one to true possessions **of the same team** by temporal
  IoU. The score is the mean F1 over the IoU thresholds {0.3, 0.4, 0.5, 0.6, 0.7}.
- X is forward-filled before possessions are built. Abstaining *inside* a possession is therefore
  free, while abstaining *at a hand-over* is costly.

Two consequences shape the whole pipeline:
- **Flicker is lethal.** One noisy frame can split a true possession into two, which costs an F1 match.
- **A team error is penalised twice:** a miss on the true team, plus a false positive on the wrong one.

So we optimise for coherent, correctly segmented, correctly teamed possessions, not for raw per-frame
accuracy. The TCN experiment (section 8) shows that per-frame accuracy is actively misleading here.

## 2. Approach

```
tactical ATD (x, y per track)
  --> 30 football features per frame
  --> HistGradientBoosting: P(H), P(A), P(D)
  --> home calibration (+0.25 log-odds) --> moving average over +/-2.0 s
  --> 3-state Viterbi over H / A / D --> X on ball-less frames inside a possession
```

- **Learned per-frame model.** We use `HistGradientBoostingClassifier`. Gradient boosting is strong on
  tabular football features and handles missing values natively. This matters because the ball is
  untracked on ~41% of graded frames. Our geometry features stay defined there, so the model can
  still decide.
- **Smooth the probabilities before decoding.** Raw per-frame probabilities flicker and fragment
  possessions. They score *worse* on spellF1 despite better per-frame accuracy. Averaging them over
  ±2.0 s before the Viterbi is what makes the learned model win.
- **3-state Viterbi over H/A/D.** Switching team directly costs 8 (log-score), and entering or leaving
  D costs 3.5. A change of possession therefore needs sustained evidence, not a few noisy frames.
- **X abstention.** We abstain on frames where the ball is not tracked, inside an H or A run, and at
  least 10 frames (0.4 s) from the run's boundaries. The scorer bridges such X, so this is free. We
  never abstain near a boundary or on D.

## 3. Features (30)

- **Distances and control:** distances from the ball to the two nearest players of each team; a
  signed control gap and a normalised control confidence; a ball-looseness term and an overall
  control margin.
- **Local and team geometry:** player counts within 2 m, 5 m and 10 m of the ball; team centroids
  and spreads.
- **Ball state:** position, velocity, smoothed speed and net travel.
- **Possession momentum:** a rolling mean of the nearest-team sign. It gives the otherwise memoryless
  model short-horizon context.

Ball-based columns go NaN when the ball is untracked, while the geometry columns stay defined.
**All temporal windows are expressed in seconds and converted to each dataset's frame rate.** The
model therefore sees the same real-time context on 29.97 fps PFF training data and on 25 fps
challenge data (section 7).

## 4. Possession rules we implemented

The labels are shaped in two places. The first is the per-frame truth we reconstructed for training
from each provider's events. The second is the decoder that turns the model's probabilities into the
submitted labels.

**Training truth, PFF FC.** The team of the last on-ball event (PFF `OTB`, or a kick-off) is carried
forward frame by frame until the next on-ball event. `OUT` (ball out of play) makes the ball dead (D)
until the next on-ball event, which is the restart. `END` closes the period. Frames before the first
event are not graded.
- *Loose balls* belong to the last team that played the ball, until the other team's next on-ball event.
- *Challenges* change possession only when the challenging team records an on-ball event. A duel
  without a clean touch leaves possession where it was.
- *Restarts:* the dead window runs from `OUT` to the restart's on-ball event, and the restarting
  team's possession starts there. Stoppages where the ball stays on the pitch (fouls, offsides) are
  not marked D in this truth (section 10).

**Training truth, Metrica sample games.** The team of each event is carried forward in the same way
until the next event. `BALL OUT`, `FAULT RECEIVED` and `CARD` open a dead window that ends at the
next event, normally the `SET PIECE` restart. Here fouls are dead.

**Validation truth, SkillCorner.** We use SkillCorner's own phases of play, i.e. the team in
possession over each phase's frames. Gaps between phases inside a period count as D.

**Decoder rules on the provided match.**
- A possession changes team only through the Viterbi, after sustained evidence: a switch cost of 8
  against ±2 s of smoothed evidence. A loose ball or a lost duel does not split a possession unless
  the other team keeps the ball.
- The model itself predicts D from the geometry of stoppages. The Viterbi keeps D only where it is
  sustained (entry and exit cost 3.5).
- In the second half, grading starts before the kick-off. The frames between the start of the
  graded block (110,808) and the Smart-Tagging `Kick Off (Start)` event (113,660) are forced to D.
- X goes on ball-less frames inside a possession (section 2) and on every ungraded frame: before the
  kick-off, at half-time and after the final whistle.

## 5. Data: what we used and what we dropped

**Provided assets.**

| asset | use |
|---|---|
| Tactical tracking (`_atd.txt`) | the only per-frame input of the model |
| Tracking metadata (`_atd.xml`) | the team of each track slot (the first team listed is home) |
| Smart Tagging | only the second-half `Kick Off (Start)` event (section 4). Its phases (team tags) and set pieces were tested as priors and dropped (section 6) |
| Graded-block boundaries, as published in `games.py` | where to decode; X elsewhere |
| Homography; tactical and broadcast videos; broadcast tracking | not used. The tactical tracking already holds pitch coordinates, and the method needs no vision |

**Labelled data, all from other games.** We labelled nothing ourselves. We never hand-label BAR-ATH
and we use no human-produced data about it.

| dataset | source and labeller | amount | used for |
|---|---|---|---|
| PFF FC FIFA World Cup 2022 | tracking and event data by PFF FC, free on request from PFF FC | 64 matches, ~11.2M graded frames: 61 for training, 3 held out (10515, 10516, 10517) | training; held-out validation |
| Metrica Sports sample data | open tracking and event data by Metrica Sports, github.com/metrica-sports/sample-data | Sample_Game_1 and Sample_Game_2 | training |
| SkillCorner open data | open broadcast tracking and phases of play by SkillCorner, github.com/SkillCorner/opendata | 3 A-League matches (1874553, 1899585, 1886347) | cross-source validation only |

We credit **PFF FC**, **Metrica Sports** and **SkillCorner**.

## 6. Validation, and how we used the public 5 minutes

We validate on two external sources:
- **PFF held-out:** 3 full World Cup matches never used in training. This is our trustworthy
  in-distribution number, with hundreds of possessions rather than 17.
- **SkillCorner cross-source:** 3 open-data A-League matches. The tracking comes from broadcast video
  in a different league, with real off-camera player dropout. The camera produced this sparsity, not
  our code, so a change that survives here is real evidence.

Every numeric parameter was chosen on external data:
- on PFF held-out matches: the smoothing window (±2.0 s), the team-switch cost (8) and the home
  calibration (+0.25 log-odds, kept deliberately moderate as a hedge against domain shift);
- in our earlier tuning loop on the Metrica sample games: the dead-state cost (3.5) and the
  abstention margin (10 frames).

**Disclosure: the public 5 minutes.** RULES.md says the published 5 minutes are for checking the
pipeline, not for tuning. We used them for that check. During development, however, we also scored
candidate variants on them, and those scores weighed in four on/off decisions. All four ended with
the add-on left out, so the submission is the base pipeline without them. Differences of this size
on 17 possessions are within noise. We therefore re-took, on external data alone, the one decision
that could still be re-taken (dropout, below).

| add-on | external evidence | public 5 min | in the submission |
|---|---|---|---|
| Pitch-control features | helped PFF and Metrica held-out | 0.430 -> 0.415 | off; not re-tested on SkillCorner |
| Player-dropout augmentation | re-tested: PFF held-out 0.401 -> 0.404, SkillCorner 0.585 -> 0.600 | 0.430 -> 0.386 | off: within noise under the rule below |
| Smart-Tagging dead-ball prior | helped on Metrica, with set-piece windows built from events covering 2.7% of frames | 0.429 -> 0.371 | off: the challenge's set-piece tags cover 11.6% of frames, so a weight fitted on Metrica does not carry over, and it cannot be fitted without labels on the target |
| Smart-Tagging team prior | not testable: no other game carries these tags | 0.430 -> 0.415 | off: it cannot be validated outside the provided game |

**Dropout re-test.** We fixed the rule before running it. We would switch to the dropout model only
if it beat the submitted model on SkillCorner by more than 0.04 spellF1 (the official noise band),
while losing no more than 0.04 on PFF held-out. It gained +0.015 on SkillCorner and +0.004 on PFF
held-out. That is the same direction on both sources, but within noise, so the submitted model
stands. The public window played no part in this re-test.

## 7. Two structural fixes that survived validation

- **Second-half dead-ball window.** The second half is graded from frame 110,808, but it kicks off
  only at frame 113,660. We take that frame from the Smart-Tagging kick-off event; nothing is
  hard-coded. The 2,852 frames in between are dead by definition, so we force D there. This removes
  phantom possessions at the start of the second half.
- **Frame-rate-consistent windows.** Our smoothing and feature windows used to be counted in frames.
  A 60-frame window is 2.0 s at 29.97 fps (PFF) but 2.4 s at 25 fps (the challenge), which merges
  more possessions than intended on a metric that punishes merging. We now express every window in
  seconds and convert it per frame rate. PFF results are unchanged (the conversion is the identity at
  29.97 fps), and the challenge becomes consistent with training.

## 8. What we tested and rejected

The challenge (Metrica tracking of a LaLiga match) comes from a different distribution than our
training data. Several techniques that helped our training proxies did not hold up beyond them. We
report the twelve levers we rejected, because a rejection backed by measurement is a result too.
"Public 5 min" marks numbers measured on the public window (section 6).

| Lever | Verdict (measured) |
| --- | --- |
| Dropping direction-dependent features (v3) | worse on PFF held-out and on Metrica cross-source |
| Model + geometry ensemble | no reliable gain |
| Smart-Tagging dead-ball prior | over-forces D on the challenge, where tags cover 11.6% of frames (section 6) |
| Pitch-control / velocity features | helped held-out, lower on the public 5 min (section 6) |
| Random player-dropout augmentation | within noise on PFF held-out and SkillCorner (section 6) |
| Structured distance masking | destroys the team-geometry features: PFF held-out -0.08 (public 5 min -0.17) |
| Minimum-possession-duration decode | targets flicker that the X-bridging already removes; we under-segment rather than over-segment |
| Orientation invariance (mirror test-time augmentation) | neutral: the orientation bias exists but costs no points |
| Smart-Tagging team prior (with DEFENDING flip) | ~34% wrong team on contested frames (section 6) |
| EM label-shift (replacing the fixed home bias) | priors diverge under covariate shift: SkillCorner -0.11 |
| Possession-persistence features | within noise: +0.004 PFF held-out, 0.000 SkillCorner |
| Lightweight MS-TCN sequence decoder | per-frame loss fell, but spellF1 was far worse: PFF held-out -0.11, SkillCorner -0.22 |

On this evidence, structural corrections transferred, while learned additions, post-hoc injections
and augmentations did not clearly pay off. That is why we ship the simpler pipeline.

## 9. Results

| Measure | spellF1 |
| --- | --- |
| **PFF held-out** (3 full matches): our reliable figure | **0.401** |
| **SkillCorner cross-source** (different league, broadcast tracking, its own truth definition) | **0.585** |
| Public 5 minutes (17 possessions: a pipeline check, too small to rank) | 0.430 |
| Official `baseline_smooth50` on the same public 5 minutes | 0.326 |
| Official `baseline_smooth50` on the full match (the published floor) | 0.2433 |

SkillCorner defines its truth differently (its own phases of play), so its absolute value is not an
estimate of our challenge score. We report it as a check of cross-source generalisation. The scores
of the provided match that decide the ranking are unknown to us: no figure above is an estimate of
them.

## 10. Limitations, and what we would fix with more time

- **Domain shift is the main risk.** The PFF training tracking runs at 29.97 fps, with stable
  identities and a mostly visible ball. The challenge tracking runs at 25 fps, with the ball missing
  on 41% of graded frames, unstable identities and about 9.5 players tracked per team. Our only check
  on the target itself is 17 possessions.
- **Truth definitions differ between sources.** PFF marks D only when the ball leaves the pitch.
  Metrica also marks it after fouls and cards. The challenge truth most likely covers every stoppage.
  Stoppages for fouls and offsides are therefore under-represented in training, and the model
  probably under-predicts D after a foul.
- **Short possessions and quick turnovers.** The ±2 s smoothing and the team-switch cost suppress
  flicker. They also merge or drop possessions shorter than about 2 s, which is the hardest part of
  the metric.
- **Boundary precision.** Smoothing can shift a boundary by up to about a second, and we do not snap
  boundaries to ball contacts.
- **No explicit dead-ball detector.** D comes only from the model's D class. We do not use
  ball-out-of-bounds, a stationary ball or player speeds directly. The Smart-Tagging set pieces stay
  unused because their weight cannot be calibrated without labels on the target.
- **One global home calibration** (+0.25 log-odds), estimated on World Cup matches.

With more time:
1. Build an explicit dead-ball detector (ball out of bounds, ball still, all players slow) and
   calibrate it on Metrica-format games with event truth.
2. Harmonise the training truth across sources, e.g. turn fouls and offsides into D in the PFF truth.
3. Snap boundaries to ball-contact moments (ball acceleration peaks near a player).
4. Train a sequence model with a segment-level loss; the per-frame-loss TCN failed on spellF1
   (section 8).
5. Add more Metrica-format training games in the challenge's missing-ball regime.

## 11. Reproduce

The pipeline is deterministic: regenerating the file gives a byte-identical copy (tested with
Python 3.9.6 and the versions pinned in `requirements.txt`).

```
pip install -r requirements.txt                # Python 3.9, pinned versions
# copy BAR-ATH_tactical_atd.txt, _atd.xml and _smart_tagging.xml into data/tactical/
python predict_challenge.py                    # -> edyss.csv (199,319 rows), about 10 s
python evaluation/validate_submission.py edyss.csv
```

SHA-256 of the submitted file: `aa891d65db678993a5a70d572d9e134fc123b067b0aea3d342b360d48c7013d5`.

Retraining and validation need the external datasets of section 5; README.md gives the downloads.

```
python final_build_64.py      # PFF feature cache (features_cache_v2/), v2/v3 feature comparison
python train_final.py         # possession_model_final.joblib from 61 PFF + 2 Metrica games
python skillcorner_eval.py    # PFF held-out and SkillCorner spellF1 of the shipped model
```

Fixed configuration:
- 30 features.
- `HistGradientBoostingClassifier`: 400 iterations, learning rate 0.08, 63 leaves, L2 1.0, early
  stopping, `random_state=0`.
- Home calibration +0.25 log-odds; smoothing ±2.0 s; switch costs 8 (team) and 3.5 (dead);
  abstention margin 10 frames.
- Second-half pre-kick-off window forced to D.

## 12. Compliance statement

- **Algorithmic throughout.** No human labels on BAR-ATH, and no human-produced data about it.
- **Trained on other games only.** Every numeric parameter was chosen on external data.
- **Public window.** The public 5-minute window was also scored during development and weighed in
  four on/off decisions, disclosed in section 6.
- **Data and reproducibility.** The challenge match data stays out of the repository, which
  regenerates the exact submitted file. Credits: PFF FC, Metrica Sports, SkillCorner.
