"""Correlation + redundancy analysis (step 3a of the assignment), fitted on TRAIN data only.

1. drop columns that are constant in train (VIF is undefined for them)
2. correlation analysis: hierarchical clustering on Spearman |rho| (distance 1 - |rho|, cut at |rho| = 0.7);
   in each cluster keep the feature with the lowest mean |rho| to all other features
   (same rule as AutoSpearman, Jiarpakdee et al. ICSME'18)
3. redundancy analysis: iteratively drop the feature with the highest VIF until all VIF <= 5
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

CORR_THRESHOLD = 0.7
VIF_THRESHOLD = 5.0
# NB: use ALL train rows. A subsample makes rare templates (often the anomaly signal) look constant
# and drops them (HDFS: 6 "constant" columns on a 100k subsample vs 2 on the full train set).


def vif(X):
    """VIF_j = 1 / (1 - R^2_j) = diag of the inverse correlation matrix."""
    corr = np.corrcoef(X, rowvar=False)
    return np.diag(np.linalg.pinv(corr))


def select_features(X_train, names):
    """Returns (kept column indices, report dict)."""
    X = np.asarray(X_train, dtype=np.float64)
    cols = np.arange(X.shape[1])

    # 1. constant columns
    nonconst = X.std(axis=0) > 0
    dropped_const = names[cols[~nonconst]].tolist()
    cols = cols[nonconst]

    # 2. Spearman correlation clustering
    dropped_corr = []
    if len(cols) > 1:
        ranks = np.apply_along_axis(rankdata, 0, X[:, cols])
        rho = np.abs(np.nan_to_num(np.corrcoef(ranks, rowvar=False)))
        np.fill_diagonal(rho, 1.0)
        dist = squareform(1 - rho, checks=False).clip(min=0)
        clusters = fcluster(linkage(dist, method="average"), t=1 - CORR_THRESHOLD, criterion="distance")
        mean_rho = (rho.sum(axis=1) - 1) / max(len(cols) - 1, 1)
        keep = []
        for c in np.unique(clusters):
            members = np.flatnonzero(clusters == c)
            best = members[np.argmin(mean_rho[members])]
            keep.append(best)
            dropped_corr += names[cols[np.setdiff1d(members, best)]].tolist()
        cols = cols[np.sort(keep)]

    # 3. VIF
    dropped_vif = []
    while len(cols) > 1:
        v = vif(X[:, cols])
        worst = int(np.argmax(v))
        if v[worst] <= VIF_THRESHOLD:
            break
        dropped_vif.append((names[cols[worst]], round(float(v[worst]), 1)))
        cols = np.delete(cols, worst)

    report = {"n_in": len(names), "n_out": len(cols),
              "dropped_constant": dropped_const, "dropped_correlation": dropped_corr,
              "dropped_vif": dropped_vif, "kept": names[cols].tolist()}
    return cols, report
