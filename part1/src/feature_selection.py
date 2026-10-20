"""Step 3a: correlation analysis + redundancy analysis (VIF) on the MCV features.

Everything is fitted on the training matrix only; the caller applies the kept columns to test.
  1. drop constant (zero-variance) features
  2. Spearman rho -> hierarchical clustering (average linkage, distance 1-|rho|),
     cut at distance `corr_cut` (0.3 <=> |rho| >= 0.7); keep one feature per cluster:
     the template with most training occurrences, ties -> smaller template id
  3. iteratively drop the feature with the highest VIF (statsmodels, with constant)
     while max VIF > `vif_threshold`; ties (e.g. several inf) -> fewer training
     occurrences, then larger template id
"""
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def _spearman(X):
    if X.shape[1] == 1:
        return np.ones((1, 1))
    rho = spearmanr(X).statistic
    if np.ndim(rho) == 0:
        rho = np.array([[1.0, rho], [rho, 1.0]])
    return np.nan_to_num(rho, nan=0.0)


def _vif_all(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    exog = add_constant(X.astype(np.float64), has_constant="add")
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        return np.array([variance_inflation_factor(exog, i) for i in range(1, exog.shape[1])])


def select_features(X, ids, names, occurrences, corr_cut=0.3, vif_threshold=5.0, fig_path=None, log=None):
    """X: [n_train, k] MCV for columns `ids`. Returns (kept_ids, report DataFrame, info dict)."""
    ids = np.asarray(ids)
    occ = dict(zip(ids.tolist(), np.asarray(occurrences).tolist()))
    rows = []

    def rec(tid, step, reason, value):
        rows.append(dict(template_id=int(tid), template=names[int(tid)], step=step, reason=reason, value=value))

    # 1. constant features
    var = X.var(axis=0)
    const = var == 0
    for j in np.flatnonzero(const):
        rec(ids[j], "constant", f"zero variance in train (always {X[0, j]:g})", 0.0)
    cols = np.flatnonzero(~const)
    if log:
        log.info("Step 3a: %d features, %d constant", len(ids), int(const.sum()))

    # 2. correlation analysis
    rho = _spearman(X[:, cols])
    n_clusters = len(cols)
    if len(cols) > 1:
        dist = 1.0 - np.abs(rho)
        np.fill_diagonal(dist, 0.0)
        dist = np.clip((dist + dist.T) / 2, 0, None)
        Z = linkage(squareform(dist, checks=False), method="average")
        clusters = fcluster(Z, t=corr_cut, criterion="distance")
        n_clusters = len(np.unique(clusters))
        keep_mask = np.ones(len(cols), dtype=bool)
        for c in np.unique(clusters):
            members = np.flatnonzero(clusters == c)
            if len(members) < 2:
                continue
            rep = min(members, key=lambda m: (-occ[int(ids[cols[m]])], int(ids[cols[m]])))
            for m in members:
                if m == rep:
                    continue
                keep_mask[m] = False
                r = rho[m, rep]
                rec(ids[cols[m]], "correlation",
                    f"correlated with template {int(ids[cols[rep]])} (rho = {r:.3f}; cluster of {len(members)})", float(r))
        if fig_path is not None:
            _plot_dendrogram(Z, [str(int(i)) for i in ids[cols]], corr_cut, fig_path)
        cols = cols[keep_mask]
    if log:
        log.info("Step 3a: %d correlation clusters -> %d features after correlation analysis", n_clusters, len(cols))

    # 3. redundancy analysis (VIF)
    vif_iters = []
    while len(cols) > 1:
        v = _vif_all(X[:, cols])
        v = np.where(np.isnan(v), np.inf, v)
        vmax = v.max()
        vif_iters.append(dict(n_features=len(cols), max_vif=float(vmax)))
        if vmax <= vif_threshold:
            break
        cand = np.flatnonzero(v == vmax)
        j = min(cand, key=lambda m: (occ[int(ids[cols[m]])], -int(ids[cols[m]])))
        rec(ids[cols[j]], "VIF", f"VIF = {vmax:.2f}" if np.isfinite(vmax) else "VIF = inf (perfect multicollinearity)",
            float(vmax))
        cols = np.delete(cols, j)
    final_vif = _vif_all(X[:, cols]) if len(cols) > 1 else np.array([1.0])
    if log:
        log.info("Step 3a: %d VIF removals -> %d features kept (max VIF %.2f)",
                 sum(r["step"] == "VIF" for r in rows), len(cols), float(np.max(final_vif)))
    report = pd.DataFrame(rows, columns=["template_id", "template", "step", "reason", "value"])
    info = dict(n_in=len(ids), n_constant=int(const.sum()),
                n_correlation=int((report.step == "correlation").sum()),
                n_vif=int((report.step == "VIF").sum()), n_kept=len(cols), n_corr_clusters=int(n_clusters),
                kept_ids=[int(i) for i in ids[cols]],
                final_vif={int(i): float(x) for i, x in zip(ids[cols], final_vif)}, vif_iterations=vif_iters)
    return ids[cols], report, info


def _plot_dendrogram(Z, labels, cut, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = max(8, min(40, 0.18 * len(labels)))
    fig, ax = plt.subplots(figsize=(w, 5))
    dendrogram(Z, labels=labels, ax=ax, color_threshold=cut, leaf_font_size=7 if len(labels) < 120 else 4)
    ax.axhline(cut, color="red", ls="--", lw=1, label=f"cut at 1-|rho| = {cut} (|rho| >= {1 - cut:.1f})")
    ax.set_ylabel("1 - |Spearman rho|  (average linkage)")
    ax.set_xlabel("template id")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
