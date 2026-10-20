"""Message Count Vector (MCV): one count per template id, per session.

The vocabulary is fitted on training sessions only (replication package
logrep/MCV_generator.py: columns = events seen in train; unseen test events are dropped).
"""
import numpy as np
import scipy.sparse as sp


def count_matrix(sessions, n_cols):
    """CSR [n_sessions, n_cols] of event counts; column j = template id j."""
    data = np.ones(len(sessions.events), dtype=np.float32)
    m = sp.csr_matrix((data, sessions.events.astype(np.int64), sessions.offsets), shape=(len(sessions), n_cols))
    m.sum_duplicates()
    return m


class MCVVectorizer:
    def __init__(self, n_templates):
        self.n_cols = n_templates + 1
        self.vocab = None            # template ids kept as columns
        self.train_occurrences = None

    def fit(self, train_sessions):
        m = count_matrix(train_sessions, self.n_cols)
        occ = np.asarray(m.sum(axis=0)).ravel()
        self.vocab = np.flatnonzero(occ > 0)
        self.vocab = self.vocab[self.vocab != 0]
        self.train_occurrences = occ[self.vocab]
        return self

    def transform(self, sessions, stats=None):
        m = count_matrix(sessions, self.n_cols)
        if stats is not None:
            total = len(sessions.events)
            kept = float(m[:, self.vocab].sum())
            stats.update(n_events=int(total), n_events_unseen_in_train=int(total - kept),
                         n_sessions_with_unseen=int(((m.sum(axis=1).A1 - m[:, self.vocab].sum(axis=1).A1) > 0).sum()))
        return m[:, self.vocab].toarray().astype(np.float32)
