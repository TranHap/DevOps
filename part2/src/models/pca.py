"""PCA anomaly detector (Xu et al., SOSP 2009), ported from loglizer.

Source: https://github.com/logpai/loglizer (commit ac67f97), files loglizer/models/PCA.py and
loglizer/preprocessing.py (FeatureExtractor). Copyright (c) 2018 LogPAI, MIT License (see
LICENSE_loglizer in this folder). Changes in this port:
  - FeatureExtractor works on Part 1's message count vector (vocabulary = templates seen in the
    training split, unseen test templates dropped == loglizer with oov=False) instead of raw
    event-sequence lists; the TF-IDF and zero-mean formulas are unchanged.
  - SPE is computed for all rows at once (same value as loglizer's per-row loop) and returned as
    a continuous score; the binary decision is loglizer's `SPE > threshold`.
Defaults verified in loglizer's source: n_components=0.95, threshold=None (Q-statistic),
c_alpha=3.2905 (alpha = 0.001); the HDFS demo uses term_weighting='tf-idf', normalization='zero-mean'.
"""
import numpy as np

from .. import p1  # noqa: F401  (registers Part 1 as `p1`)
from p1.represent_mcv import MCVVectorizer


class FeatureExtractor:
    """loglizer.preprocessing.FeatureExtractor (oov=False), on a dense count matrix."""

    def __init__(self, term_weighting="tf-idf", normalization="zero-mean"):
        self.term_weighting = term_weighting
        self.normalization = normalization
        self.idf_vec = None
        self.mean_vec = None

    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        num_instance, num_event = X.shape
        if self.term_weighting == "tf-idf":
            df_vec = np.sum(X > 0, axis=0)
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            X = X * self.idf_vec[None, :]
        if self.normalization == "zero-mean":
            self.mean_vec = X.mean(axis=0).reshape(1, num_event)
            X = X - self.mean_vec
        elif self.normalization == "sigmoid":
            from scipy.special import expit
            X[X != 0] = expit(X[X != 0])
        return X

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.term_weighting == "tf-idf":
            X = X * self.idf_vec[None, :]
        if self.normalization == "zero-mean":
            X = X - self.mean_vec
        elif self.normalization == "sigmoid":
            from scipy.special import expit
            X[X != 0] = expit(X[X != 0])
        return X


class PCA:
    """loglizer.models.PCA: residual-subspace projection + Q-statistic threshold."""

    def __init__(self, n_components=0.95, threshold=None, c_alpha=3.2905):
        self.proj_C = None
        self.components = None
        self.n_components = n_components
        self.n_components_ = None
        self.threshold = threshold
        self.c_alpha = c_alpha
        self.explained = None

    def fit(self, X):
        num_instances, num_events = X.shape
        X_cov = np.dot(X.T, X) / float(num_instances)
        U, sigma, V = np.linalg.svd(X_cov)
        n_components = self.n_components
        if n_components < 1:
            total_variance = np.sum(sigma)
            variance = 0
            for i in range(num_events):
                variance += sigma[i]
                if variance / total_variance >= n_components:
                    break
            n_components = i + 1
        P = U[:, :n_components]
        I = np.identity(num_events, int)
        self.components = P
        self.proj_C = I - np.dot(P, P.T)
        self.n_components_ = int(n_components)
        self.explained = float(np.sum(sigma[:n_components]) / np.sum(sigma))
        if not self.threshold:
            # Q-statistic (Jackson & Mudholkar 1979; Lakhina et al. SIGCOMM 2004), as in loglizer
            phi = np.zeros(3)
            for i in range(3):
                for j in range(n_components, num_events):
                    phi[i] += np.power(sigma[j], i + 1)
            h0 = 1.0 - 2 * phi[0] * phi[2] / (3.0 * phi[1] * phi[1])
            self.threshold = phi[0] * np.power(self.c_alpha * np.sqrt(2 * phi[1] * h0 * h0) / phi[0]
                                               + 1.0 + phi[1] * h0 * (h0 - 1) / (phi[0] * phi[0]),
                                               1.0 / h0)
        return self

    def spe(self, X):
        Y = X @ self.proj_C.T          # row i: proj_C @ x_i
        return np.einsum("ij,ij->i", Y, Y)

    def predict(self, X):
        return (self.spe(X) > self.threshold).astype(int)


class PCAMCV:
    """Unsupervised PCA on MCV -> TF-IDF -> zero-mean. Labels are never read."""
    name = "PCA"

    def __init__(self, n_templates, params, seed=0, log=None):
        self.n_templates = n_templates
        self.p = dict(params)
        self.seed = seed            # PCA is deterministic; kept for the uniform interface
        self.log = log
        self.vec = self.fe = self.pca = None

    def fit(self, train_sessions):
        self.vec = MCVVectorizer(self.n_templates).fit(train_sessions)
        self.fe = FeatureExtractor(self.p.get("term_weighting", "tf-idf"), self.p.get("normalization", "zero-mean"))
        X = self.fe.fit_transform(self.vec.transform(train_sessions))
        self.pca = PCA(self.p.get("n_components", 0.95), None, self.p.get("c_alpha", 3.2905)).fit(X)
        if self.log:
            self.log.info("PCA: %d features, %d components (%.4f of variance), SPE threshold %.4f",
                          X.shape[1], self.pca.n_components_, self.pca.explained, self.pca.threshold)
        return self

    @property
    def threshold(self):
        return float(self.pca.threshold)

    def info(self):
        return dict(n_features=int(len(self.vec.vocab)), n_components=self.pca.n_components_,
                    explained_variance=self.pca.explained, threshold=self.threshold,
                    c_alpha=self.pca.c_alpha, n_components_param=self.p.get("n_components", 0.95))

    def predict_scores(self, sessions):
        return self.pca.spe(self.fe.transform(self.vec.transform(sessions)))
