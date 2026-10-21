"""CLI: embed stays.chiefcomplaint -> text_emb.parquet (stay_id + emb_0..emb_767).

Primary encoder: frozen Bio_ClinicalBERT (config.text.encoder), mean-free
[CLS] token of the last hidden state, batched inference on cuda.
An explicit text.encoder=hashing uses the local encoder without network access.
Otherwise ANY exception (network unreachable, model load failure, OOM, ...) falls back
to a deterministic hashing bag-of-words embedding (md5-stable across runs,
768-dim, signed, L2-normalized) and prints a warning. Empty complaints always
get the zero vector.

Usage:
    PYTHONPATH=src python -m wjem_ed_audit.data.text_embed --config configs/smoke.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .api import load_config, load_stays

EMB_DIM = 768
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ------------------------------------------------------------- hashing fallback
def _hash_embed(texts: list[str], dim: int = EMB_DIM) -> np.ndarray:
    """Stable signed hashing bag-of-words, L2-normalized; empty text -> zeros."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            continue
        vec = np.zeros(dim, dtype=np.float64)
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            out[i] = (vec / norm).astype(np.float32)
    return out


# ------------------------------------------------------------------ BERT encoder
def _bert_embed(texts: list[str], cfg: dict) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    text_cfg = cfg.get("text", {})
    name = text_cfg.get("encoder", "emilyalsentzer/Bio_ClinicalBERT")
    max_len = int(text_cfg.get("max_len", 32))
    batch_size = int(text_cfg.get("batch_size", 64))

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    for p in model.parameters():  # frozen encoder
        p.requires_grad_(False)

    n = len(texts)
    out = np.zeros((n, EMB_DIM), dtype=np.float32)
    nonempty = [i for i, t in enumerate(texts) if (t or "").strip()]
    with torch.no_grad():
        for lo in range(0, len(nonempty), batch_size):
            idx = nonempty[lo:lo + batch_size]
            batch = tok([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
            cls = model(**batch).last_hidden_state[:, 0]  # [CLS]
            out[np.asarray(idx)] = cls.float().cpu().numpy()
    return out


def embed_texts(texts: list[str], cfg: dict) -> tuple[np.ndarray, str]:
    """Honor explicit local hashing; retain the historical BERT fallback otherwise."""
    if cfg.get("text", {}).get("encoder") == "hashing":
        return _hash_embed(texts), "hashing_explicit"
    try:
        emb = _bert_embed(texts, cfg)
        encoder_used = cfg.get("text", {}).get("encoder", "bert")
        print(f"[text_embed] encoder = Bio_ClinicalBERT ({encoder_used})")
        return emb, encoder_used
    except Exception as e:  # Historical behavior: warn and fall back.
        print(f"[text_embed] WARNING: BERT encoder failed ({type(e).__name__}: {e}); "
              "falling back to hashing embeddings.")
        return _hash_embed(texts), "hashing_fallback"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_dir = Path(cfg.get("data", {}).get("cache_dir", "data/cache"))
    stays = load_stays(cache_dir)
    texts = stays["chiefcomplaint"].fillna("").astype(str).tolist()

    emb, encoder_used = embed_texts(texts, cfg)

    zero_rows = int((~np.array([bool((t or "").strip()) for t in texts])).sum())
    df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(EMB_DIM)])
    df.insert(0, "stay_id", stays["stay_id"].to_numpy().astype(np.int64))
    out_path = cache_dir / "text_emb.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[text_embed] wrote {out_path} rows={len(df)} dim={EMB_DIM} "
          f"encoder={encoder_used} zero_vectors(empty complaint)={zero_rows}")


if __name__ == "__main__":
    main()
