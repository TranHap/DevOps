"""BERT template embeddings (event-level representation for the LSTM).

Follows the replication package (logrep/BERT_generator_server.py + bert-as-service):
- checkpoint uncased_L-12_H-768_A-12 (Google BERT-base uncased) == HF `bert-base-uncased`
- input text = the raw EventTemplate (placeholders kept), max_seq_len = 50
- sentence vector = mean over all non-padding tokens (incl. [CLS]/[SEP], i.e. bert-as-service
  REDUCE_MEAN with input_mask) of the second-to-last encoder layer (paper §4.2), 768-d.

Each unique template is embedded exactly once -> embeddings.npy [n_templates + 1, 768],
row 0 = zeros (padding).
"""
import re

import numpy as np

from .common import data_dir, get_device, write_json


def template_text(t, keep_placeholders=True):
    if keep_placeholders:
        return t
    s = re.sub(r"\s+", " ", t.replace("<*>", " ")).strip()
    return s or t


def compute_embeddings(cfg, templates, log, force=False):
    out = data_dir(cfg, "features") / "bert_embeddings.npy"
    if out.exists() and not force:
        log.info("BERT embedding cache hit: %s", out)
        return np.load(out)
    import torch
    from transformers import AutoModel, AutoTokenizer
    bc = cfg["bert"]
    device = get_device(log)
    tok = AutoTokenizer.from_pretrained(bc["checkpoint"])
    model = AutoModel.from_pretrained(bc["checkpoint"], output_hidden_states=True).to(device).eval()
    texts = [template_text(t, bc.get("keep_placeholders", True)) for t in templates.EventTemplate]
    E = np.zeros((len(templates) + 1, model.config.hidden_size), dtype=np.float32)
    rows = templates.TemplateId.values
    with torch.no_grad():
        for a in range(0, len(texts), bc.get("batch_size", 64)):
            batch = texts[a:a + bc.get("batch_size", 64)]
            enc = tok(batch, padding=True, truncation=True, max_length=bc["max_seq_len"], return_tensors="pt").to(device)
            hs = model(**enc).hidden_states[bc["layer"]]           # [B, T, 768]
            m = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            vec = (hs * m).sum(1) / m.sum(1)
            E[rows[a:a + len(batch)]] = vec.float().cpu().numpy()
    np.save(out, E)
    n_trunc = sum(len(tok(t)["input_ids"]) > bc["max_seq_len"] for t in texts)
    write_json(data_dir(cfg, "features") / "bert_meta.json",
               dict(checkpoint=bc["checkpoint"], layer=bc["layer"], max_seq_len=bc["max_seq_len"],
                    keep_placeholders=bc.get("keep_placeholders", True), pooling="mean over attention mask (incl. CLS/SEP)",
                    n_templates=len(templates), dim=int(E.shape[1]), n_truncated_templates=int(n_trunc)))
    log.info("BERT embeddings %s saved (%d templates truncated at %d tokens)", E.shape, n_trunc, bc["max_seq_len"])
    return E


def embedding_rank(E):
    """Rank evidence for not applying VIF to the 768 BERT dimensions (padding row excluded)."""
    M = E[1:].astype(np.float64)
    sv = np.linalg.svd(M, compute_uv=False)
    rank = int(np.linalg.matrix_rank(M))
    Mc = M - M.mean(0)              # rank of the centred matrix = what VIF/correlation "see"
    rank_c = int(np.linalg.matrix_rank(Mc))
    return dict(n_templates=int(M.shape[0]), embedding_dim=int(M.shape[1]), rank=rank,
                rank_centered=rank_c, max_possible_rank=int(min(M.shape)),
                n_dims_linearly_dependent=int(M.shape[1] - rank),
                singular_values_top5=[float(x) for x in sv[:5]],
                singular_values_bottom5=[float(x) for x in sv[-5:]],
                condition_number=float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"))
