"""ESI probe: can the TLQ-T C-group representation reconstruct acuity (ESI)?

Approach (chosen over forward hooks for simplicity): ``TLQT.forward`` supports
``return_group_repr=True``, which additionally returns a dict of per-group
pooled representations (mean over the group's queries, before the head).
This probe runs the frozen model over a dataset, collects the **C-group**
representation, and fits two probes against the recorded acuity
(``tab_J`` in the batch, with ``mask_J == 1`` rows only):

- Ridge regression on the raw acuity value -> R^2
- 4-class logistic probe (ESI levels 1 / 2 / 3 / 4+, i.e. levels 4 and 5
  merged) -> one-vs-rest macro AUC

Returns ``{"probe_auc": .., "probe_r2": ..}``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

_MODEL_KEYS = ("hourly", "hourly_mask", "text", "tab_C", "tab_J", "mask_J",
               "tab_P", "tab_D")


@torch.no_grad()
def _collect(model, dataset, device, batch_size: int = 256):
    model.eval()
    reprs, acuity, mask = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size):
        b = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in batch.items() if k in _MODEL_KEYS}
        _, group_repr = model(b, return_group_repr=True)
        reprs.append(group_repr["C"].float().cpu().numpy())
        acuity.append(batch["tab_J"].numpy().reshape(-1))
        mask.append(batch["mask_J"].numpy().reshape(-1))
    X = np.concatenate(reprs)
    y = np.concatenate(acuity)
    m = np.concatenate(mask) > 0.5
    return X[m], y[m]


def train_esi_probe(model, dataset, device) -> dict:
    """Freeze `model` (no gradients are taken), probe C-group repr -> acuity."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    device = torch.device(device)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    X, y = _collect(model, dataset, device)
    out = {"probe_auc": float("nan"), "probe_r2": float("nan")}
    if len(X) < 10:
        return out
    strat = np.clip(np.round(y), 1, 4).astype(int)  # 4-class: 1/2/3/4+
    Xtr, Xte, ytr, yte, ctr, cte = train_test_split(
        X, y, strat, test_size=0.3, random_state=0,
        stratify=strat if np.unique(strat).size > 1 else None)

    out["probe_r2"] = float(r2_score(
        yte, make_pipeline(StandardScaler(), Ridge()).fit(Xtr, ytr).predict(Xte)))

    classes = np.unique(ctr)
    if classes.size >= 2:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000)).fit(Xtr, ctr)
        proba = clf.predict_proba(Xte)
        keep = [i for i, c in enumerate(classes) if c in set(cte)]
        if len(keep) >= 2:
            out["probe_auc"] = float(roc_auc_score(
                cte, proba[:, keep], multi_class="ovr", average="macro",
                labels=classes[keep]))
    return out
