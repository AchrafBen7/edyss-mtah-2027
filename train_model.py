"""Train the learned possession model on the cached PFF features, held out by match.

Splits by MATCH (not by frame) so there is no leakage, trains a gradient-boosting classifier to
predict H/A/D per frame, and reports held-out accuracy against the nearest-player heuristic. The
real metric is spellF1 (scored after feeding these probabilities into the Viterbi); per-frame
accuracy here is the fast first signal that the model learns something the heuristic does not.

    python train_model.py
"""

import glob
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "features_cache")
H, A, D = 0, 1, 2


def load_cache():
    names = json.load(open(os.path.join(CACHE, "names.json")))
    data = {}
    for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        mid = os.path.splitext(os.path.basename(f))[0]
        z = np.load(f)
        data[mid] = (z["X"], z["y"])
    return names, data


def nearest_player_pred(X, names):
    """Heuristic baseline: home if the home player is closer (gap>0), else away. Ball present only."""
    gap = X[:, names.index("gap")]        # da1 - dh1, >0 means home closer
    present = X[:, names.index("present")] > 0.5
    pred = np.full(len(X), -1, np.int8)
    pred[present] = np.where(gap[present] > 0, H, A)
    return pred, present


if __name__ == "__main__":
    names, data = load_cache()
    ids = sorted(data)
    test_ids = ids[-3:]
    train_ids = ids[:-3]
    print("train: {} matchs | held-out: {}".format(len(train_ids), test_ids))

    Xtr = np.concatenate([data[i][0] for i in train_ids])
    ytr = np.concatenate([data[i][1] for i in train_ids])
    print("train set: {} images, {} features".format(Xtr.shape[0], Xtr.shape[1]))

    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=0)
    clf.fit(Xtr, ytr.astype(int))

    print("\n=== held-out, par match ===")
    print("{:<8} {:>10} {:>12} {:>14}".format("match", "acc modele", "acc heur.", "acc H/A (vivant)"))
    for mid in test_ids:
        X, y = data[mid]
        y = y.astype(int)
        pred = clf.predict(X)
        acc3 = np.mean(pred == y)
        heur, present = nearest_player_pred(X, names)
        live = present & ((y == H) | (y == A))
        acc_model_live = np.mean(pred[live] == y[live])
        acc_heur_live = np.mean(heur[live] == y[live])
        print("{:<8} {:>9.1f}% {:>11.1f}% {:>8.1f}% vs {:.1f}%".format(
            mid, 100 * acc3, 100 * np.mean(heur[present] == y[present]),
            100 * acc_model_live, 100 * acc_heur_live))

    # overall held-out live H/A: model vs heuristic
    Xte = np.concatenate([data[i][0] for i in test_ids])
    yte = np.concatenate([data[i][1] for i in test_ids]).astype(int)
    pred = clf.predict(Xte)
    heur, present = nearest_player_pred(Xte, names)
    live = present & ((yte == H) | (yte == A))
    print("\nGLOBAL held-out  H/A sur images vivantes:  modele {:.1f}%   heuristique {:.1f}%".format(
        100 * np.mean(pred[live] == yte[live]), 100 * np.mean(heur[live] == yte[live])))
    print("GLOBAL held-out  3-classes (H/A/D):  modele {:.1f}%".format(100 * np.mean(pred == yte)))

    joblib.dump({"model": clf, "names": names}, os.path.join(HERE, "possession_model.joblib"))
    print("\nmodele sauvegarde -> possession_model.joblib")
