"""E3: explain RF + MCV (without step 3a) with SHAP (brief Section 9).

  python -m src.explain [--seed 0] [--stage all|global|grouped|crosscheck|local|stability]

Inputs : data/models/rf_seed{s}.joblib and results/scores/rf_seed{s}.parquet (from run_resamples.py),
         Part 1's sessions, split and templates.
Outputs: results/explain/ -- sample_seed0.csv, shap_global.csv, shap_grouped.csv, clusters_seed0.json,
         shap_crosscheck.csv (+ _detail.csv), permutation_importance.csv, local_cases.md, stability.csv
         (+ stability_per_seed.csv, sk_feature_importance.csv), figures (*.png).
Caches : data/explain/ (SHAP values per seed, RF + step 3a model).

All SHAP values are for the anomaly class (predict_proba[:, 1]); TreeExplainer checks additivity.
"""
import argparse
import json
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau
from sklearn.metrics import adjusted_rand_score

from . import p1  # noqa: F401
from .p1 import PART2_ROOT
from .run_resamples import load_config, p2path
from p1.common import data_dir as p1_data_dir
from p1.common import get_logger, write_json
from p1.common import load_config as load_p1_config
from p1.feature_selection import _spearman
from p1.partition import load_sessions
from p1.represent_mcv import MCVVectorizer, count_matrix
from p1.split import get_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class E:
    """Shared state: config, templates, sessions."""

    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log
        self.cfg1 = load_p1_config(cfg["part1_config"])
        t = pd.read_csv(p1_data_dir(self.cfg1, "parsed") / "templates.csv", keep_default_na=False)
        self.text = dict(zip(t.TemplateId.astype(int), t.EventTemplate))
        self.n_templates = len(t)
        self.s = load_sessions(self.cfg1)
        self.out = p2path(cfg, "results_dir", "explain")
        self.cache = p2path(cfg, "data_root", "explain")
        self.models = p2path(cfg, "data_root", "models")
        self.scores = p2path(cfg, "results_dir", "scores")

    def label(self, tid, n=45):
        t = self.text[int(tid)]
        return f"T{int(tid)}: {t[:n] + ('…' if len(t) > n else '')}"

    def X(self, sessions, columns):
        return count_matrix(sessions, self.n_templates + 1)[:, columns].toarray().astype(np.float32)


# ---------------------------------------------------------------------------- correlation clusters
def correlation_clusters(X, ids, occ, corr_cut):
    """Steps 1-2 of Part 1's step 3a (p1.feature_selection.select_features), returning the cluster
    membership that select_features uses internally: constant features removed, Spearman rho,
    average linkage on 1 - |rho|, cut at `corr_cut`; representative = most training occurrences,
    ties -> smaller id. Constant features become singleton groups (their SHAP value is 0)."""
    ids = np.asarray(ids)
    occ = dict(zip(ids.tolist(), np.asarray(occ).tolist()))
    var = X.var(axis=0)
    cols = np.flatnonzero(var != 0)
    rho = _spearman(X[:, cols])
    dist = 1.0 - np.abs(rho)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    Z = linkage(squareform(dist, checks=False), method="average")
    lab = fcluster(Z, t=corr_cut, criterion="distance")
    clusters = []
    for c in np.unique(lab):
        members = [int(ids[cols[m]]) for m in np.flatnonzero(lab == c)]
        rep = min(members, key=lambda t: (-occ[t], t))
        clusters.append(dict(rep=rep, members=sorted(members)))
    for j in np.flatnonzero(var == 0):
        clusters.append(dict(rep=int(ids[j]), members=[int(ids[j])], constant=True))
    clusters.sort(key=lambda c: c["rep"])
    for c in clusters:
        c["name"] = f"G{c['rep']}"
    return clusters


def get_clusters(e, seed):
    path = e.cache / f"clusters_seed{seed}.json"
    if path.exists():
        return json.load(open(path))
    tr, _ = get_split(e.cfg1, seed)
    train = e.s.subset(tr)
    vec = MCVVectorizer(e.n_templates).fit(train)
    X = vec.transform(train)
    cl = correlation_clusters(X, vec.vocab, vec.train_occurrences, e.cfg1["feature_selection"]["corr_cut"])
    write_json(path, cl)
    return cl


def group_matrix(clusters, columns):
    """[n_features, n_groups] 0/1 matrix: feature (template id in `columns`) -> its cluster."""
    pos = {int(t): i for i, t in enumerate(columns)}
    G = np.zeros((len(columns), len(clusters)))
    for k, c in enumerate(clusters):
        for t in c["members"]:
            if t in pos:
                G[pos[t], k] = 1
    assert (G.sum(1) == 1).all(), "every feature must belong to exactly one cluster"
    return G


# ---------------------------------------------------------------------------- SHAP helpers
def load_rf(e, name, seed):
    d = joblib.load(e.models / f"{name}_seed{seed}.joblib")
    return d["clf"], np.asarray(d["columns"])


def shap_values(e, clf, X, cache_name):
    import shap
    path = e.cache / f"{cache_name}.npz"
    if path.exists():
        z = np.load(path)
        if z["values"].shape[0] == X.shape[0]:
            return z["values"], float(z["base"])
    t0 = time.time()
    ex = shap.TreeExplainer(clf)
    sv = ex.shap_values(X, check_additivity=True)
    sv = sv[1] if isinstance(sv, list) else sv[..., 1]          # anomaly class
    base = float(np.atleast_1d(ex.expected_value)[1])
    np.savez(path, values=sv, base=base)
    e.log.info("SHAP %s: %s in %.1fs", cache_name, sv.shape, time.time() - t0)
    return sv, base


def sample_sessions(e, seed, n_normal, test_idx=None):
    """All anomalous test sessions + `n_normal` normal test sessions drawn with default_rng(seed)."""
    _, te = get_split(e.cfg1, seed)
    y = e.s.labels[te]
    rng = np.random.default_rng(seed)
    pos = te[y == 1]
    neg = np.sort(rng.choice(te[y == 0], min(n_normal, int((y == 0).sum())), replace=False))
    return np.concatenate([pos, neg])


def rank_desc(v):
    return pd.Series(v).rank(ascending=False, method="min").astype(int).values


# ---------------------------------------------------------------------------- stages
def stage_global(e, seed, n_normal):
    clf, cols = load_rf(e, "rf", seed)
    idx = sample_sessions(e, seed, n_normal)
    sub = e.s.subset(idx)
    pd.DataFrame(dict(session_idx=idx, session_id=e.s.keys[idx].astype(str), label=sub.labels)).to_csv(
        e.out / f"sample_seed{seed}.csv", index=False)
    X = e.X(sub, cols)
    sv, base = shap_values(e, clf, X, f"shap_rf_seed{seed}")
    y = sub.labels
    names = [e.label(t) for t in cols]
    df = pd.DataFrame(dict(template_id=cols, template=[e.text[int(t)] for t in cols],
                           mean_abs_shap=np.abs(sv).mean(0),
                           mean_shap_anomalous=sv[y == 1].mean(0), mean_shap_normal=sv[y == 0].mean(0),
                           mean_count_anomalous=X[y == 1].mean(0), mean_count_normal=X[y == 0].mean(0),
                           rf_impurity_importance=clf.feature_importances_))
    df["rank"] = rank_desc(df.mean_abs_shap)
    df = df.sort_values("rank")
    df.to_csv(e.out / "shap_global.csv", index=False)
    write_json(e.out / f"sample_seed{seed}_info.json",
               dict(seed=seed, n_anomalous=int(y.sum()), n_normal=int((y == 0).sum()), n_normal_requested=n_normal,
                    sampling="all anomalous test sessions + normal test sessions drawn with numpy default_rng(seed)",
                    base_value=base, n_features=len(cols)))
    _plot_global(e, sv, X, names, base)
    e.log.info("global: %d sessions (%d anomalous); top-5 %s", len(idx), int(y.sum()),
               df.template_id.head(5).tolist())
    return idx, X, sv, cols


def _plot_global(e, sv, X, names, base, top=20):
    import shap
    order = np.argsort(-np.abs(sv).mean(0))[:top]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(order) + 1.2))
    ax.barh(range(len(order))[::-1], np.abs(sv).mean(0)[order], color="#3b6ea5")
    ax.set_yticks(range(len(order))[::-1], [names[i] for i in order], fontsize=7)
    ax.set_xlabel("mean |SHAP value| (impact on P(anomaly))")
    ax.set_title(f"RF + MCV, seed 0: global importance (top {len(order)} of {len(names)} templates)")
    fig.tight_layout()
    fig.savefig(e.out / "shap_bar.png", dpi=170)
    plt.close(fig)
    expl = shap.Explanation(values=sv, base_values=np.full(len(sv), base), data=X, feature_names=names)
    plt.figure()
    shap.plots.beeswarm(expl, max_display=top, show=False)
    plt.gcf().set_size_inches(10, 0.32 * top + 1.5)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.gcf().suptitle("RF + MCV, seed 0: SHAP value per session (colour = template count)", fontsize=10)
    plt.savefig(e.out / "shap_beeswarm.png", dpi=170)
    plt.close("all")


def stage_grouped(e, seed, sv, cols, X):
    clusters = get_clusters(e, seed)
    G = group_matrix(clusters, cols)
    gsv = sv @ G                                      # SHAP values are additive
    imp = np.abs(gsv).mean(0)
    sum_member = np.abs(sv).mean(0) @ G
    rows = []
    for k, c in enumerate(clusters):
        rows.append(dict(group=c["name"], representative=c["rep"], n_members=len(c["members"]),
                         members=" ".join(map(str, c["members"])),
                         member_templates=" | ".join(e.text[t] for t in c["members"]),
                         mean_abs_grouped_shap=imp[k], sum_member_mean_abs_shap=sum_member[k]))
    df = pd.DataFrame(rows)
    df["rank"] = rank_desc(df.mean_abs_grouped_shap)
    df = df.sort_values("rank")
    df.to_csv(e.out / "shap_grouped.csv", index=False)
    write_json(e.out / f"clusters_seed{seed}.json", clusters)
    _check_clusters_vs_part1(e, seed, clusters)
    order = np.argsort(-imp)[:20]
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(order) + 1.2))
    lab = [f"{clusters[k]['name']} ({len(clusters[k]['members'])}): " + e.label(clusters[k]["rep"], 38).split(": ", 1)[1]
           for k in order]
    ax.barh(range(len(order))[::-1], imp[order], color="#3b6ea5")
    ax.set_yticks(range(len(order))[::-1], lab, fontsize=7)
    ax.set_xlabel("mean |grouped SHAP value| (sum over the members of a correlation cluster)")
    ax.set_title("RF + MCV, seed 0: importance per step-3a correlation cluster (n members)")
    fig.tight_layout()
    fig.savefig(e.out / "shap_grouped_bar.png", dpi=170)
    plt.close(fig)
    e.log.info("grouped: %d clusters (%d with >1 member); top-5 %s", len(clusters),
               sum(len(c["members"]) > 1 for c in clusters), df.group.head(5).tolist())
    return clusters, gsv


def _check_clusters_vs_part1(e, seed, clusters):
    """The clusters must reproduce Part 1's step-3a correlation removals for this seed."""
    f = PART2_ROOT.parent / "part1" / "results" / f"feature_selection_hdfs_random_seed{seed}.csv"
    if not f.exists():
        e.log.warning("no Part 1 feature-selection report for seed %d; cluster check skipped", seed)
        return
    rep = pd.read_csv(f)
    rep = rep[rep.step == "correlation"]
    removed = {int(t): int(r.split("template ")[1].split(" ")[0]) for t, r in zip(rep.template_id, rep.reason)}
    ours = {t: c["rep"] for c in clusters for t in c["members"] if t != c["rep"]}
    assert removed == ours, f"clusters differ from Part 1's step 3a (seed {seed})"
    e.log.info("clusters reproduce Part 1's step-3a correlation removals for seed %d (%d removed)", seed, len(ours))


def stage_crosscheck(e, seed, idx, X, sv, cols, clusters, gsv):
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import f1_score
    from p1.models.rf import RFMCV
    sub = e.s.subset(idx)
    y = sub.labels
    # --- RF + MCV + step 3a, same seed and sample
    fs_path = e.cache / f"rf_fs_seed{seed}.joblib"
    if not fs_path.exists():
        tr, _ = get_split(e.cfg1, seed)
        t = pd.DataFrame(dict(TemplateId=list(e.text), EventTemplate=list(e.text.values())))
        m = RFMCV(t, e.cfg1["rf"], seed, e.cfg1["feature_selection"], None, e.log).fit(e.s.subset(tr))
        m.save(fs_path)
    d = joblib.load(fs_path)
    clf_fs, cols_fs = d["clf"], np.asarray(d["columns"])
    sv_fs, _ = shap_values(e, clf_fs, e.X(sub, cols_fs), f"shap_rf_fs_seed{seed}")
    imp_fs = dict(zip(cols_fs.tolist(), np.abs(sv_fs).mean(0)))
    g_of = {t: c["name"] for c in clusters for t in c["members"]}
    gimp = dict(zip([c["name"] for c in clusters], np.abs(gsv).mean(0)))
    # --- permutation importance on F1 (main model), per feature and per cluster (columns permuted jointly)
    clf, _ = load_rf(e, "rf", seed)
    Xdf = pd.DataFrame(X, columns=clf.feature_names_in_)
    t0 = time.time()
    pi = permutation_importance(clf, Xdf, y, scoring="f1", n_repeats=10, random_state=seed, n_jobs=1)
    base_f1 = f1_score(y, clf.predict(Xdf))
    rng = np.random.default_rng(seed)
    gpi = {}
    for c in clusters:
        pos = [i for i, t in enumerate(cols) if t in c["members"]]
        drops = []
        for _ in range(10):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, pos] = X[perm][:, pos]
            drops.append(base_f1 - f1_score(y, clf.predict(pd.DataFrame(Xp, columns=clf.feature_names_in_))))
        gpi[c["name"]] = (float(np.mean(drops)), float(np.std(drops)))
    e.log.info("permutation importance done in %.1fs (F1 on the sample = %.4f)", time.time() - t0, base_f1)
    pdf = pd.DataFrame(dict(template_id=cols, template=[e.text[int(t)] for t in cols],
                            perm_importance_f1_mean=pi.importances_mean, perm_importance_f1_std=pi.importances_std,
                            mean_abs_shap=np.abs(sv).mean(0), group=[g_of[int(t)] for t in cols]))
    pdf["rank_perm"] = rank_desc(pdf.perm_importance_f1_mean)
    pdf["rank_shap"] = rank_desc(pdf.mean_abs_shap)
    pdf.sort_values("rank_perm").to_csv(e.out / "permutation_importance.csv", index=False)

    # --- detail table per cluster
    det = []
    for c in clusters:
        kept = [t for t in c["members"] if t in imp_fs]
        det.append(dict(group=c["name"], members=" ".join(map(str, c["members"])),
                        grouped_shap=gimp[c["name"]],
                        rf_fs_feature=kept[0] if kept else None,
                        rf_fs_shap=imp_fs[kept[0]] if kept else np.nan,
                        grouped_perm_f1=gpi[c["name"]][0], grouped_perm_f1_std=gpi[c["name"]][1]))
    det = pd.DataFrame(det)
    det["rank_grouped_shap"] = rank_desc(det.grouped_shap)
    det.to_csv(e.out / "shap_crosscheck_detail.csv", index=False)

    def compare(name, a, b, keys, note):
        a, b = np.asarray(a, float), np.asarray(b, float)
        tau, p = kendalltau(a, b)
        ta = [keys[i] for i in np.argsort(-a, kind="stable")[:5]]
        tb = [keys[i] for i in np.argsort(-b, kind="stable")[:5]]
        return dict(comparison=name, n_items=len(keys), kendall_tau=tau, p_value=p,
                    top5_overlap=len(set(ta) & set(tb)), top5_a=" ".join(map(str, ta)), top5_b=" ".join(map(str, tb)),
                    note=note)

    both = det[det.rf_fs_feature.notna()]
    rows = [compare("grouped SHAP (RF) vs SHAP (RF + step 3a)", both.grouped_shap, both.rf_fs_shap, both.group.tolist(),
                    "clusters whose representative survived the VIF step; RF + 3a has one feature per such cluster"),
            compare("grouped SHAP (RF) vs grouped permutation importance (F1)", det.grouped_shap, det.grouped_perm_f1,
                    det.group.tolist(), "all clusters; member columns permuted jointly, 10 repeats"),
            compare("SHAP vs permutation importance (F1), per feature", pdf.mean_abs_shap, pdf.perm_importance_f1_mean,
                    [int(t) for t in pdf.template_id], "main model, all features, sklearn permutation_importance, 10 repeats")]
    cc = pd.DataFrame(rows)
    cc.to_csv(e.out / "shap_crosscheck.csv", index=False)
    write_json(e.out / "crosscheck_info.json",
               dict(rf_fs_n_features=len(cols_fs), rf_fs_kept=cols_fs.tolist(),
                    clusters_without_rf_fs_feature=det[det.rf_fs_feature.isna()].group.tolist(),
                    f1_on_sample=base_f1, fs_info=d.get("fs_info")))
    e.log.info("cross-check:\n%s", cc[["comparison", "n_items", "kendall_tau", "top5_overlap"]].to_string())


def stage_local(e, seed, clusters):
    import shap
    clf, cols = load_rf(e, "rf", seed)
    sc = pd.read_parquet(e.scores / f"rf_seed{seed}.parquet")
    tp = sc[(sc.label == 1) & (sc.prediction == 1)]
    fn = sc[(sc.label == 1) & (sc.prediction == 0)].sort_values("score", ascending=False)
    fp = sc[(sc.label == 0) & (sc.prediction == 1)].sort_values("score", ascending=False).head(5)
    rng = np.random.default_rng(seed)
    tp_pick = pd.concat([tp.iloc[[int(rng.integers(len(tp)))]], tp.sort_values("score").head(1)])
    cases = [("TP", r, "random true positive (seeded)" if i == 0 else "true positive with the lowest score")
             for i, (_, r) in enumerate(tp_pick.iterrows())]
    cases += [("FN", r, "false negative") for _, r in fn.iterrows()]
    cases += [("FP", r, "false positive (highest scores first)") for _, r in fp.iterrows()]
    idx = np.array([int(r.session_idx) for _, r, _ in cases])
    sub = e.s.subset(idx)
    X = e.X(sub, cols)
    sv, base = shap_values(e, clf, X, f"shap_rf_local_seed{seed}")
    # normal reference: median count per template over normal training sessions
    tr, _ = get_split(e.cfg1, seed)
    trs = e.s.subset(tr)
    Xn = e.X(trs.subset(np.flatnonzero(trs.labels == 0)), cols)
    med_n = np.median(Xn, 0)
    pres_n = (Xn > 0).mean(0)
    names = [e.label(t, 38) for t in cols]
    figdir = e.out / "local"
    figdir.mkdir(exist_ok=True)
    md = [f"# Local explanations: RF + MCV (without step 3a), seed {seed}", "",
          f"Test set of seed {seed}: {len(sc)} sessions; TP {len(tp)}, FN {len(fn) if len(fn) else 0}, "
          f"FP {int(((sc.label == 0) & (sc.prediction == 1)).sum())}. Cases: 2 TP, every FN, up to 5 FP.",
          f"SHAP values are for P(anomaly); base value (mean prediction of the forest) = {base:.4f}; decision: P ≥ 0.5.",
          "'normal median' = median count of the template in normal training sessions; "
          "'in x% of normal' = share of normal training sessions that contain it.", ""]
    for k, ((kind, r, why), x, s) in enumerate(zip(cases, X, sv)):
        sid = str(r.session_id)
        expl = shap.Explanation(values=s, base_values=base, data=x, feature_names=names)
        plt.figure()
        shap.plots.waterfall(expl, max_display=10, show=False)
        plt.gcf().set_size_inches(9, 5.5)
        plt.tight_layout(rect=(0, 0, 1, 0.93))
        plt.gcf().suptitle(f"{kind} {sid}: P(anomaly) = {r.score:.3f}, label = {int(r.label)}", fontsize=9)
        fname = f"waterfall_{k:02d}_{kind}_{sid}.png"
        plt.savefig(figdir / fname, dpi=150)
        plt.close("all")
        seq = sub.seq(k)
        top = np.argsort(-np.abs(s))[:5]
        md += [f"## {k + 1}. {kind}: {sid} ({why})", "",
               f"- Label {'anomaly' if r.label else 'normal'}; P(anomaly) = {r.score:.4f} → predicted "
               f"{'anomaly' if r.prediction else 'normal'}. Figure: `local/{fname}`",
               f"- Sequence ({len(seq)} events, consecutive repeats collapsed): " + _seq_text(e, seq), "",
               "| template | count here | normal median | share of normal sessions with it | SHAP |", "|---|---|---|---|---|"]
        for j in top:
            md.append(f"| {names[j]} | {int(x[j])} | {med_n[j]:g} | {_pct(pres_n[j])} | {s[j]:+.4f} |")
        md += ["", "- Reading: " + _reading(kind, r, x, s, top, cols, med_n, pres_n), ""]
    (e.out / "local_cases.md").write_text("\n".join(md))
    e.log.info("local: %d cases (%d FN, %d FP) -> local_cases.md", len(cases), len(fn), len(fp))


def _seq_text(e, seq):
    out, i = [], 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        n = j - i + 1
        out.append(f"T{int(seq[i])}" + (f"×{n}" if n > 1 else ""))
        i = j + 1
    return " → ".join(out)


def _pct(p):
    return "0%" if p == 0 else (f"{100 * p:.2g}%" if p < 0.01 else f"{100 * p:.1f}%")


def _reading(kind, r, x, s, top, cols, med_n, pres_n):
    """Factual one-line reading generated from the numbers above (no hand-written interpretation)."""
    parts = []
    for j in top[:3]:
        direction = "towards anomaly" if s[j] > 0 else "towards normal"
        if x[j] == 0 and pres_n[j] > 0.5:
            what = f"T{int(cols[j])} is absent although {100 * pres_n[j]:.0f}% of normal sessions contain it"
        elif x[j] > 0 and pres_n[j] == 0:
            what = f"T{int(cols[j])} occurs ({int(x[j])}×) although no normal training session contains it"
        elif x[j] > 0 and pres_n[j] < 0.05:
            what = f"T{int(cols[j])} occurs ({int(x[j])}×) although only {_pct(pres_n[j])} of normal sessions contain it"
        elif x[j] != med_n[j]:
            what = f"T{int(cols[j])} occurs {int(x[j])}× (normal median {med_n[j]:g})"
        else:
            what = f"T{int(cols[j])} occurs {int(x[j])}× (= normal median)"
        parts.append(f"{what} ({direction}, {s[j]:+.3f})")
    verdict = {"TP": "correctly flagged", "FN": "missed", "FP": "wrongly flagged"}[kind]
    return f"{verdict} — " + "; ".join(parts) + "."


def stage_stability(e, seeds, n_normal):
    clusters0 = get_clusters(e, 0)
    names = [c["name"] for c in clusters0]
    rows, same, own_rows, ari = [], {}, [], {}
    for seed in seeds:
        mp = e.models / f"rf_seed{seed}.joblib"
        if not mp.exists():
            e.log.warning("stability: no RF model for seed %d; skipped", seed)
            continue
        clf, cols = load_rf(e, "rf", seed)
        idx = sample_sessions(e, seed, n_normal)
        X = e.X(e.s.subset(idx), cols)
        sv, _ = shap_values(e, clf, X, f"shap_rf_seed{seed}" if seed == 0 else f"shap_rf_stab_seed{seed}")
        # the grouping is fixed to seed 0's clusters so that groups are comparable across resamples;
        # whether each seed's own clusters are identical is recorded
        cl_s = get_clusters(e, seed)
        same[seed] = sorted(map(tuple, (c["members"] for c in cl_s))) == sorted(map(tuple, (c["members"] for c in clusters0)))
        cl_fixed = [dict(c, members=[t for t in c["members"] if t in set(cols.tolist())]) for c in clusters0]
        cl_fixed = [c for c in cl_fixed if c["members"]]
        extra = sorted(set(cols.tolist()) - {t for c in cl_fixed for t in c["members"]})
        cl_fixed += [dict(name=f"G{t}", rep=t, members=[t]) for t in extra]   # template unseen in seed 0's train
        imp = np.abs(sv @ group_matrix(cl_fixed, cols)).mean(0)
        # robustness: the same quantity with this seed's own clusters (Part 1's procedure on its own train split)
        imp_own = np.abs(sv @ group_matrix(cl_s, cols)).mean(0)
        rk = rank_desc(imp_own)
        for c, v, k in zip(cl_s, imp_own, rk):
            own_rows.append(dict(seed=seed, members=" ".join(map(str, c["members"])), importance=v, rank=int(k)))
        lab0 = {t: i for i, c in enumerate(clusters0) for t in c["members"]}
        labs = {t: i for i, c in enumerate(cl_s) for t in c["members"]}
        common = sorted(set(lab0) & set(labs))
        ari[seed] = float(adjusted_rand_score([lab0[t] for t in common], [labs[t] for t in common]))
        r = pd.Series(imp, index=[c["name"] for c in cl_fixed]).reindex(sorted(set(names) | {c["name"] for c in cl_fixed}))
        rows.append(pd.DataFrame(dict(seed=seed, group=r.index, mean_abs_grouped_shap=r.values)))
    per = pd.concat(rows, ignore_index=True)
    per["rank"] = per.groupby("seed").mean_abs_grouped_shap.rank(ascending=False, method="min")
    per.to_csv(e.out / "stability_per_seed.csv", index=False)
    n_seeds = per.seed.nunique()
    st = per.groupby("group").agg(mean_importance=("mean_abs_grouped_shap", "mean"),
                                  std_importance=("mean_abs_grouped_shap", "std"),
                                  median_rank=("rank", "median"), best_rank=("rank", "min"), worst_rank=("rank", "max"),
                                  n_top5=("rank", lambda v: int((v <= 5).sum()))).reset_index()
    st["share_top5"] = st.n_top5 / n_seeds
    mem = {c["name"]: " ".join(map(str, c["members"])) for c in clusters0}
    st["members"] = st.group.map(mem)
    st["representative_template"] = st.group.map({c["name"]: e.text[c["rep"]] for c in clusters0})
    st = st.sort_values(["mean_importance"], ascending=False)
    st.to_csv(e.out / "stability.csv", index=False)
    own = pd.DataFrame(own_rows)
    own_st = own.groupby("members").agg(n_seeds_present=("seed", "nunique"), mean_importance=("importance", "mean"),
                                        median_rank=("rank", "median"), n_top5=("rank", lambda v: int((v <= 5).sum())),
                                        seeds=("seed", lambda v: " ".join(map(str, sorted(v))))).reset_index()
    own_st["member_templates"] = own_st.members.map(lambda m: " | ".join(e.text[int(t)] for t in m.split()))
    own_st = own_st.sort_values(["n_top5", "mean_importance"], ascending=False)
    own_st.to_csv(e.out / "stability_own_clusters.csv", index=False)
    own.to_csv(e.out / "stability_own_clusters_per_seed.csv", index=False)
    write_json(e.out / "stability_info.json", dict(seeds=sorted(per.seed.unique().tolist()), n_normal_per_seed=n_normal,
                                                   grouping="seed-0 correlation clusters for every seed",
                                                   own_clusters_identical_to_seed0={str(k): v for k, v in same.items()},
                                                   adjusted_rand_index_vs_seed0={str(k): v for k, v in ari.items()},
                                                   robustness_file="stability_own_clusters.csv (each seed's own clusters)"))
    # optional: Scott-Knott ESD (np) on the per-resample importance of each cluster
    wide = per.pivot(index="seed", columns="group", values="mean_abs_grouped_shap").fillna(0.0)
    wide = wide[[g for g in st.group if g in wide.columns]]
    wide.to_csv(e.out / "sk_feature_importance_input.csv")
    if n_seeds >= 3:
        r = subprocess.run(["Rscript", str(PART2_ROOT / "src" / "sk_features.R"), str(e.out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            e.log.warning("sk_features.R failed: %s", r.stderr[-500:])
    e.log.info("stability over %d seeds:\n%s", n_seeds, st.head(8)[["group", "mean_importance", "median_rank", "n_top5"]].to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part2_hdfs.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-normal", type=int, default=5000)
    ap.add_argument("--stage", default="all", choices=["all", "global", "grouped", "crosscheck", "local", "stability"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = get_logger("part2.explain", p2path(cfg, "logs_dir") / "explain.log")
    e = E(cfg, log)
    st = a.stage
    idx, X, sv, cols = stage_global(e, a.seed, a.n_normal)
    clusters, gsv = stage_grouped(e, a.seed, sv, cols, X)
    if st in ("all", "crosscheck"):
        stage_crosscheck(e, a.seed, idx, X, sv, cols, clusters, gsv)
    if st in ("all", "local"):
        stage_local(e, a.seed, clusters)
    if st in ("all", "stability"):
        stage_stability(e, cfg["seeds"], a.n_normal)


if __name__ == "__main__":
    main()
