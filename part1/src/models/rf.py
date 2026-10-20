"""Random Forest + MCV, with optional step 3a (correlation + VIF feature selection).

Uniform interface (Part 2): fit(train_sessions), predict_scores(sessions) -> P(anomaly) per session.
Hyper-parameters follow the replication package (models/traditional.py):
RandomForestClassifier(n_estimators=10, max_features='auto'(=sqrt), max_depth=None,
min_samples_split=2, bootstrap=True), no class weighting.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..feature_selection import select_features
from ..represent_mcv import MCVVectorizer


class RFMCV:
    name = "RF"
    representation = "MCV"

    def __init__(self, templates, params, seed=42, feature_selection=None, fig_path=None, log=None):
        """templates: DataFrame with TemplateId, EventTemplate. feature_selection: dict or None."""
        self.names = dict(zip(templates.TemplateId.astype(int), templates.EventTemplate))
        self.n_templates = len(templates)
        self.params = dict(params)
        self.seed = seed
        self.fs_cfg = feature_selection
        self.fig_path = fig_path
        self.log = log
        self.vectorizer = None
        self.columns = None          # template ids used as features
        self.fs_report = None
        self.fs_info = None
        self.clf = None

    def _frame(self, X, ids):
        return pd.DataFrame(X, columns=[self.names[int(i)] for i in ids])

    def fit(self, train_sessions):
        self.vectorizer = MCVVectorizer(self.n_templates).fit(train_sessions)
        X = self.vectorizer.transform(train_sessions)
        ids = self.vectorizer.vocab
        if self.fs_cfg is not None:
            kept, self.fs_report, self.fs_info = select_features(
                X, ids, self.names, self.vectorizer.train_occurrences,
                corr_cut=self.fs_cfg["corr_cut"], vif_threshold=self.fs_cfg["vif_threshold"],
                fig_path=self.fig_path, log=self.log)
            pos = np.searchsorted(ids, kept)
            X, ids = X[:, pos], kept
        self.columns = np.asarray(ids)
        self.clf = RandomForestClassifier(random_state=self.seed, n_jobs=-1, **self.params)
        # fit on a DataFrame so that feature_names_in_ == template texts (for SHAP in Part 2)
        self.clf.fit(self._frame(X, self.columns), train_sessions.labels)
        return self

    def _X(self, sessions, stats=None):
        X = self.vectorizer.transform(sessions, stats)
        pos = np.searchsorted(self.vectorizer.vocab, self.columns)
        return self._frame(X[:, pos], self.columns)

    def predict_scores(self, sessions, stats=None):
        proba = self.clf.predict_proba(self._X(sessions, stats))
        classes = list(self.clf.classes_)
        return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(sessions))

    @property
    def n_features(self):
        return len(self.columns)

    def save(self, path):
        joblib.dump(dict(clf=self.clf, columns=self.columns, vocab=self.vectorizer.vocab,
                         feature_names=[self.names[int(i)] for i in self.columns],
                         params=self.params, seed=self.seed, fs_info=self.fs_info), path, compress=3)
