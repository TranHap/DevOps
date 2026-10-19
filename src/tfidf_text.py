"""TF-IDF (Text) event-level representation, as in Wu et al.'s replication package (TT_TFIDF_generator.py):
fit TfidfVectorizer on the TRAIN sessions (one document = the template texts of a session), then represent each
event by the TF-IDF vector of its template text. Features = template tokens (e.g. "block", "exception").
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_templates(ds):
    t = pd.read_csv(os.path.join(ROOT, "data", "parsed", ds, f"{ds}_templates.csv"), keep_default_na=False)
    return dict(zip(t["EventId"], t["EventTemplate"]))


def event_vectors(ds, sessions, train_idx):
    """Returns (E, names): E[event_id] = TF-IDF vector of its template (row max_id+1 = zero padding)."""
    templates = load_templates(ds)
    train_docs = [" ".join(templates[e] for e in seq) for seq in sessions["events"].to_numpy()[train_idx]]
    vec = TfidfVectorizer(lowercase=True)
    vec.fit(train_docs)
    max_id = max(templates)
    E = np.zeros((max_id + 2, len(vec.vocabulary_)), dtype=np.float32)
    ids = sorted(templates)
    E[ids] = vec.transform([templates[i] for i in ids]).toarray()
    return E, np.array(vec.get_feature_names_out())
