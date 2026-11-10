# edyss: More than a Hack 2027, Phase 1 (ball possession)

This is the code behind team **edyss**'s possession file for the More than a Hack 2027 initial challenge:
FC Barcelona vs Athletic Club, 199,319 frames, with one label per frame (`H` / `A` / `D` / `X`).

A gradient-boosting classifier turns the provided tactical tracking into per-frame P(H), P(A) and P(D).
It is trained only on other games: PFF FC World Cup 2022 and the Metrica Sports sample games. The
probabilities are smoothed over ±2 s and decoded into possessions by a 3-state Viterbi. The
technical document [WRITEUP.md](WRITEUP.md) covers the method, the data, the possession rules,
the validation, what we rejected and the limitations.

## Regenerate the submission

Tested on Python 3.9.6, macOS arm64, in about 10 seconds.

```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The challenge data is competition-only, so it is not in this repository.
# Copy these three files from the tactical bundle into data/tactical/:
#   BAR-ATH_tactical_atd.txt  BAR-ATH_tactical_atd.xml  BAR-ATH_tactical_smart_tagging.xml
mkdir -p data/tactical

python predict_challenge.py                          # writes edyss.csv
python evaluation/validate_submission.py edyss.csv   # the official format check
shasum -a 256 edyss.csv                              # sha256sum on Linux
```

The output is byte-identical to the file we submitted:

```
aa891d65db678993a5a70d572d9e134fc123b067b0aea3d342b360d48c7013d5  edyss.csv
```

`predict_challenge.py` uses the trained model shipped here, `possession_model_final.joblib`. That file
is a scikit-learn pickle, so keep the pinned versions in `requirements.txt`.

From the challenge data, the pipeline reads:
- the tactical tracking (`_atd.txt`);
- its metadata (`_atd.xml`), which gives the team of each track;
- from the Smart Tagging, only the second-half `Kick Off (Start)` event.

It does not use the homography, either video, or the broadcast tracking.

## Retrain and validate (optional)

You do not need this step to regenerate the submission. The external datasets are public, but they
are not redistributed here. The table below gives the default locations. You can override them with
environment variables (see `data_paths.py`).

| dataset | used for | expected location |
|---|---|---|
| PFF FC FIFA World Cup 2022, tracking + metadata (free on request from PFF FC) | training (61 matches), held-out validation (3) | `pff/tracking/<id>.jsonl.bz2`, `pff/metadata/<id>.json` |
| Metrica Sports sample data, <https://github.com/metrica-sports/sample-data> | training (Sample_Game_1 and 2) | `external/sample-data/data/` or `$METRICA_SAMPLE_DATA` |
| SkillCorner open data, <https://github.com/SkillCorner/opendata> | cross-source validation only (3 matches) | `external/skillcorner/data/matches/` or `$SKILLCORNER_DATA` |

```bash
git clone https://github.com/metrica-sports/sample-data external/sample-data

# SkillCorner tracking files are stored with Git LFS, so fetch the three matches directly:
for id in 1874553 1899585 1886347; do
  d=external/skillcorner/data/matches/$id; mkdir -p $d
  for f in match.json phases_of_play.csv; do
    curl -fL -o $d/${id}_$f https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/$id/${id}_$f
  done
  curl -fL -o $d/${id}_tracking_extrapolated.jsonl \
    https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/$id/${id}_tracking_extrapolated.jsonl
done

python final_build_64.py     # builds the PFF feature cache (features_cache_v2/) and compares feature sets
python train_final.py        # trains possession_model_final.joblib (61 PFF + 2 Metrica), scores PFF held-out
python skillcorner_eval.py   # spellF1 of the shipped model on PFF held-out and on SkillCorner
```

## What is where

| | files |
|---|---|
| submission pipeline | `predict_challenge.py`, `features.py`, `possession.py` (Viterbi, abstention), `eval_model.py` (smoothing, decoding), `calibrate_bias.py`, `games.py`, `possession_model_final.joblib` |
| training | `pff_loader.py`, `dev_sampledata.py` (Metrica loader and event-based truth), `final_build_64.py`, `train_final.py`, `data_paths.py` |
| validation | `skillcorner_loader.py`, `skillcorner_eval.py`, `metrica_validate.py` |
| development scripts | tuning, diagnostics and the rejected experiments of WRITEUP section 8: `calibrate_bias.py`, `sweep_v2.py`, `metrica_augment.py`, `robust_v3.py`, `ensemble_eval.py`, `smarttag_eval.py`, `final_pc.py`, `pc_gonogo.py`, `final_dropout.py`, `dropout_gonogo.py`, `lever2_gonogo.py`, `mirror_tta_eval.py`, `em_labelshift.py`, `persist_gonogo.py`, `prep_tcn_data.py`, `tcn_train.py`, `diag_sparsity.py`, `measure_presence.py`, `metrica_diag.py`, `run_v2.py`, `train_model.py`, `build_features.py`, `ingest.py` |
| official scorer, from the challenge repository | `evaluation/` |
| technical document | `WRITEUP.md`; the PDF is built with `python md_to_pdf.py WRITEUP.md edyss.pdf` (`requirements-extra.txt`) |

## Rules compliance

- Code produces every label. We used no hand labels on the provided game and no human-produced data
  about it.
- The model is trained on other games only. Numeric parameters were chosen on external data (PFF
  held-out matches and the Metrica sample games).
- During development we also scored variants on the public 5-minute window. WRITEUP section 6 lists
  the decisions this influenced and how each one stands in the submission.
- This repository contains neither the match data nor any frame-level export of it.

## Credits

- **Metrica Sports**: the challenge data, the open sample data and the scorer.
- **FC Barcelona Innovation Hub** and **Mobile World Capital Barcelona**: co-organisers.
- **PFF FC**: the FIFA World Cup 2022 tracking and event data.
- **SkillCorner**: the open broadcast-tracking data.

MIT licence, see [LICENSE](LICENSE). `games.py` and `evaluation/` remain under the challenge
repository's licence ([LICENSE-challenge.txt](LICENSE-challenge.txt)).
