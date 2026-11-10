"""Where the external (non-challenge) datasets live. Override any of them with an environment variable.

The challenge files go in data/tactical/ (see data/README.md). The external datasets below are public,
are used only for training and validation, and are NOT part of this repository: download them as
described in README.md. By default they are looked up under external/ next to this file.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Metrica Sports open sample data -- the data/ folder of https://github.com/metrica-sports/sample-data
METRICA_SAMPLE_DATA = os.environ.get(
    "METRICA_SAMPLE_DATA", os.path.join(HERE, "external", "sample-data", "data"))

# SkillCorner open data -- the data/matches/ folder of https://github.com/SkillCorner/opendata
SKILLCORNER_DATA = os.environ.get(
    "SKILLCORNER_DATA", os.path.join(HERE, "external", "skillcorner", "data", "matches"))
