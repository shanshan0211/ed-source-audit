"""TLQ-T: Triage Latent-Query Transformer (SPEC §6).

Tokenisation
------------
- ``hourly`` (B,T,10): per-feature linear embedding (token_t = sum_f x_tf * W_f,
  W in R^{10xd}) plus a learnable time embedding (one vector per hour).
- ``text`` (B,768): linear projection to a single token.
- ``tab_C`` (B,7) / ``tab_J`` (B,1) / ``tab_P`` (B,Pp) / ``tab_D`` (B,Pd): each
  linearly projected to one token.  ``tab_J``'s token is multiplied by
  ``mask_J`` so a missing acuity contributes an all-zero token.  The tab_P /
  tab_D projections are ``nn.LazyLinear`` because their input widths depend on
  the fitted one-hot encoders; they materialise on the first forward pass, so
  callers must run one batch through the model *before* creating an optimizer.

Latent-query pooler
-------------------
n_query learnable queries are partitioned by source group (default
{C:16, J:4, P:4, D:4}).  Each group owns an independent stack of `n_layers`
cross-attention blocks (cross-attn + FFN, pre-norm) that pools *only* that
group's source tokens:

    C -> hourly tokens + text token + tab_C token
    J -> tab_J token      P -> tab_P token      D -> tab_D token

The 28 pooled queries are concatenated, flattened and fed to an MLP head
producing the logit.  ``suppress_groups=["J", ...]`` zeroes the corresponding
group's query outputs before the head (intervention, SPEC §5 E3);
``suppress_groups=["random"]`` zeroes a random J-sized subset of the C queries
as a control.  Hourly tokens padded by the dataset are excluded via
``hourly_mask``.

Risk curve
----------
``return_risk_curve=True`` additionally returns a (B,T) per-hour risk curve.
Simplified implementation (per SPEC): the C-group queries attend with a
causal (prefix) mask in a *single* auxiliary cross-attention layer (not the
full 4-layer stack), the per-hour pooled representations are mean-reduced
over queries and mapped by a linear risk head.  Keys are the hourly tokens
within the causal prefix plus static tokens (text, tab_C, tab_J, tab_P,
tab_D) that the implementation exposes at every cursor. This does not verify
that the source fields were available at triage; see docs/REPRODUCIBILITY.md.

bf16 compatibility: all heavy ops run under ``torch.autocast``; attention
masks are boolean and filled with ``-inf`` in float32 before softmax.

Parameter counts depend on materialized categorical input widths.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP_ORDER = ("C", "J", "P", "D")


# ------------------------------------------------------------------- blocks
class CrossBlock(nn.Module):
    """Pre-norm cross-attention (queries attend to source tokens) + FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        a, _ = self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv),
                         key_padding_mask=key_padding_mask, need_weights=False)
        q = q + self.drop(a)
        q = q + self.drop(self.ffn(self.ln2(q)))
        return q


class GroupPooler(nn.Module):
    """Independent per-group latent-query cross-attention pooler."""

    def __init__(self, n_query: int, d_model: int, n_heads: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_query, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [CrossBlock(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, kv: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.queries.unsqueeze(0).expand(kv.size(0), -1, -1)
        for layer in self.layers:
            q = layer(q, kv, key_padding_mask)
        return q  # (B, n_query, d)


# ------------------------------------------------------------------- model
class TLQT(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        tc = cfg.get("tlqt", cfg)  # accept full config or the tlqt sub-dict
        d = int(tc.get("d_model", 128))
        n_layers = int(tc.get("n_layers", 4))
        n_heads = int(tc.get("n_heads", 4))
        dropout = float(tc.get("dropout", 0.1))
        horizon = int(tc.get("horizon_h", 24))
        qg = dict(tc.get("query_groups", {"C": 16, "J": 4, "P": 4, "D": 4}))
        self.d_model, self.query_groups = d, {g: int(qg[g]) for g in GROUP_ORDER}
        self.n_hourly_feat = 10

        # tokenisers
        self.hourly_feat_emb = nn.Parameter(torch.randn(self.n_hourly_feat, d) * 0.02)
        self.hourly_bias = nn.Parameter(torch.zeros(d))
        self.time_emb = nn.Parameter(torch.randn(horizon, d) * 0.02)
        self.text_proj = nn.Linear(768, d)
        self.tab_c_proj = nn.Linear(7, d)
        self.tab_j_proj = nn.Linear(1, d)
        self.tab_p_proj = nn.LazyLinear(d)  # width depends on fitted encoder
        self.tab_d_proj = nn.LazyLinear(d)

        # per-group poolers
        self.poolers = nn.ModuleDict({
            g: GroupPooler(self.query_groups[g], d, n_heads, n_layers, dropout)
            for g in GROUP_ORDER})

        n_q = sum(self.query_groups.values())
        self.head = nn.Sequential(nn.LayerNorm(n_q * d),
                                  nn.Linear(n_q * d, 2 * d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(2 * d, 1))

        # auxiliary single-layer causal pooler for the hourly risk curve
        self.rc_q = nn.Linear(d, d)
        self.rc_k = nn.Linear(d, d)
        self.rc_v = nn.Linear(d, d)
        self.rc_ln = nn.LayerNorm(d)
        self.rc_head = nn.Linear(d, 1)

    # --------------------------------------------------------- tokenisation
    def _tokens(self, batch: dict) -> dict[str, torch.Tensor]:
        hourly = batch["hourly"].float()               # (B,T,10)
        B, T, _ = hourly.shape
        if T > self.time_emb.size(0):
            raise ValueError(f"batch T={T} exceeds horizon {self.time_emb.size(0)}")
        tok_h = torch.einsum("btf,fd->btd", hourly, self.hourly_feat_emb) \
            + self.hourly_bias + self.time_emb[:T].unsqueeze(0)
        tok_text = self.text_proj(batch["text"].float()).unsqueeze(1)      # (B,1,d)
        tok_c = self.tab_c_proj(batch["tab_C"].float()).unsqueeze(1)       # (B,1,d)
        mask_j = batch.get("mask_J")
        tj = self.tab_j_proj(batch["tab_J"].float()).unsqueeze(1)          # (B,1,d)
        if mask_j is not None:
            tj = tj * mask_j.float().unsqueeze(-1)
        tok_p = self.tab_p_proj(batch["tab_P"].float()).unsqueeze(1)
        tok_d = self.tab_d_proj(batch["tab_D"].float()).unsqueeze(1)
        return {"C": torch.cat([tok_h, tok_text, tok_c], dim=1),
                "J": tj, "P": tok_p, "D": tok_d}

    def _kv_mask(self, batch: dict, n_c_tokens: int, T: int,
                 device) -> torch.Tensor | None:
        """key_padding_mask for the C pooler (True = ignore)."""
        hm = batch.get("hourly_mask")
        if hm is None:
            return None
        valid = torch.cat([hm.bool(),
                           torch.ones(hm.size(0), n_c_tokens - T, dtype=torch.bool,
                                      device=device)], dim=1)
        return ~valid

    # ------------------------------------------------------------- forward
    def forward(self, batch: dict, suppress_groups: list[str] | None = None,
                return_risk_curve: bool = False, return_group_repr: bool = False):
        tokens = self._tokens(batch)
        T = batch["hourly"].shape[1]
        pooled: dict[str, torch.Tensor] = {}
        for g in GROUP_ORDER:
            kpm = self._kv_mask(batch, tokens["C"].size(1), T,
                                tokens["C"].device) if g == "C" else None
            pooled[g] = self.poolers[g](tokens[g], kpm)  # (B, n_g, d)

        group_repr = {g: pooled[g].mean(dim=1) for g in GROUP_ORDER}  # (B,d)

        if suppress_groups:
            for g in suppress_groups:
                if g == "random":  # control: zero a random J-sized subset of C
                    n = self.query_groups["J"]
                    idx = torch.randperm(self.query_groups["C"],
                                         device=pooled["C"].device)[:n]
                    pooled["C"] = pooled["C"].clone()
                    pooled["C"][:, idx, :] = 0.0
                elif g in pooled:
                    pooled[g] = torch.zeros_like(pooled[g])

        flat = torch.cat([pooled[g] for g in GROUP_ORDER], dim=1).flatten(1)
        logits = self.head(flat).squeeze(-1)  # (B,)

        outs: list = [logits]
        if return_risk_curve:
            outs.append(self._risk_curve(batch))
        if return_group_repr:
            outs.append(group_repr)
        return outs[0] if len(outs) == 1 else tuple(outs)

    # ---------------------------------------------------------- risk curve
    def _risk_curve(self, batch: dict) -> torch.Tensor:
        """Per-hour prefix risk, (B,T). Causal-masked single cross-attention.

        Keys = static triage-time tokens (text, tab_C, tab_J, tab_P, tab_D --
        all known at h=0, visible at every cursor) + hourly tokens restricted
        to the causal prefix 0..t.  Hourly data is sparse (median ~3 observed
        hours/stay) and ~37% of stays have no hour-0 row, so an hourly-only
        curve degenerates to a constant on empty prefixes and caps far below
        the main head; the static tokens give every cursor a valid prediction.
        """
        hourly = batch["hourly"].float()
        B, T, _ = hourly.shape
        tok_h = torch.einsum("btf,fd->btd", hourly, self.hourly_feat_emb) \
            + self.hourly_bias + self.time_emb[:T].unsqueeze(0)
        tok_text = self.text_proj(batch["text"].float()).unsqueeze(1)  # (B,1,d)
        tok_c = self.tab_c_proj(batch["tab_C"].float()).unsqueeze(1)
        tj = self.tab_j_proj(batch["tab_J"].float()).unsqueeze(1)
        mask_j = batch.get("mask_J")
        if mask_j is not None:
            tj = tj * mask_j.float().unsqueeze(-1)
        tok_p = self.tab_p_proj(batch["tab_P"].float()).unsqueeze(1)
        tok_d = self.tab_d_proj(batch["tab_D"].float()).unsqueeze(1)
        kv = torch.cat([tok_h, tok_text, tok_c, tj, tok_p, tok_d], dim=1)  # (B,T+5,d)
        S = kv.size(1)

        q = self.rc_q(self.poolers["C"].queries).unsqueeze(0).expand(B, -1, -1).float()
        k, v = self.rc_k(kv).float(), self.rc_v(kv).float()
        scores = torch.einsum("bqd,bsd->bqs", q, k) / math.sqrt(self.d_model)
        scores = scores.unsqueeze(2).expand(B, q.size(1), T, S)  # (B,Q,t,s)
        causal = torch.arange(T, device=hourly.device)
        mask = torch.zeros(T, S, dtype=torch.bool, device=hourly.device)
        mask[:, :T] = causal[None, :] > causal[:, None]  # hourly key s>t -> ignore
        scores = scores.masked_fill(mask, float("-inf"))
        hm = batch.get("hourly_mask")
        if hm is not None:
            pad = torch.cat([~hm.bool(), torch.zeros(B, S - T, dtype=torch.bool,
                                                     device=hourly.device)], dim=1)
            scores = scores.masked_fill(pad[:, None, None, :], float("-inf"))
        probs = torch.nan_to_num(F.softmax(scores, dim=-1))
        out = torch.einsum("bqts,bsd->bqtd", probs, v)    # (B,Q,T,d)
        pooled_t = self.rc_ln(out.mean(dim=1))                     # (B,T,d)
        return self.rc_head(pooled_t).squeeze(-1)                  # (B,T)


def build_model(cfg: dict) -> nn.Module:
    """Build a TLQ-T from the full config dict (or its ``tlqt`` sub-dict)."""
    return TLQT(cfg)
