"""Time-cursor risk-curve evaluation for TLQ-T (SPEC §6 return_risk_curve).

Loads a trained TLQ-T run (``results/runs/{task}/tlqt/{tag}/seed{k}/model.pt``)
and scores the random-test set with ``return_risk_curve=True`` -> (B,T) per-hour
prefix risk.  For each time cursor h the AUC of the risk value at hour h (which
only attends to hours 0..h via the causal mask) is computed, yielding the
"AUC vs information time" curve that mirrors HCNet's real-time updating卖点.

CLI:
    python -m wjem_ed_audit.eval.riskcurve \
        --config configs/smoke.yaml --task critical --config_tag full --seed 0

Writes ``{results_root}/figures/fig6_riskcurve.png`` and
``{results_root}/tables/table4_riskcurve.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wjem_ed_audit.data import api

FIG_DPI = 300

# batch keys consumed by TLQT.forward (mirrors models.trainer._MODEL_KEYS)
MODEL_KEYS = ("hourly", "hourly_mask", "text", "tab_C", "tab_J", "mask_J",
              "tab_P", "tab_D")


def _pick_device(cfg: dict) -> str:
    want = str(cfg.get("device", "cpu"))
    if want.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
        except ImportError:
            return "cpu"
    return want


def load_tlqt_run(results_root, task: str, config_tag: str, seed: int,
                  cfg: dict, device: str):
    """build_model(cfg) + load model.pt from a SPEC §4 run dir."""
    import torch
    from wjem_ed_audit.models.tlqt import build_model
    run_dir = Path(results_root) / "runs" / task / "tlqt" / config_tag / f"seed{seed}"
    if not (run_dir / "model.pt").is_file():
        raise FileNotFoundError(f"TLQ-T run not found (need model.pt): {run_dir}")
    model = build_model(cfg)
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def risk_curve_auc(model, dataset, device, cursors=(0, 1, 2, 3, 4, 5),
                   batch_size: int = 256) -> pd.DataFrame:
    """AUC at each time cursor h from the (B,T) causal prefix risk curve."""
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ys, curves = [], []
    with torch.no_grad():
        for batch in loader:
            b = {k: batch[k].to(device) for k in MODEL_KEYS if k in batch}
            _, risk = model(b, return_risk_curve=True)  # (B,T) logits
            ys.append(batch["label"].detach().cpu().numpy().ravel())
            curves.append(risk.detach().float().cpu().numpy())
    y = np.concatenate(ys).astype(int)
    curve = np.concatenate(curves, axis=0)  # (N,T)

    rows = []
    for h in cursors:
        if h >= curve.shape[1]:
            continue
        score = curve[:, h]
        try:
            auc = float(roc_auc_score(y, score))
        except ValueError:
            auc = float("nan")
        rows.append({"cursor_h": int(h), "auc": auc, "n": int(len(y))})
    return pd.DataFrame(rows, columns=["cursor_h", "auc", "n"])


def plot_risk_curve(df: pd.DataFrame, out_png) -> Path:
    """AUC vs ED-arrival-relative time; horizontal reference is the h=0 prefix."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["cursor_h"], df["auc"], "o-", color="#4C72B0", lw=1.8,
            label="TLQ-T prefix risk AUC")
    base = float(df.loc[df["cursor_h"] == 0, "auc"].iloc[0]) \
        if (df["cursor_h"] == 0).any() else float(df["auc"].iloc[0])
    ax.axhline(base, color="#C44E52", ls="--", lw=1.2,
               label=f"h=0 prefix reference AUC={base:.3f}")
    ax.set_xlabel("time cursor h (hours since ED arrival)")
    ax.set_ylabel("AUROC")
    ax.set_xticks(df["cursor_h"].tolist())
    ax.set_ylim(min(0.5, df["auc"].min() - 0.02), 1.0)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TLQ-T time-cursor risk-curve eval")
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default=None, help="default: cfg main_task")
    ap.add_argument("--config_tag", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cursors", default=None,
                    help="comma list, default 0..horizon_h-1")
    args = ap.parse_args(argv)

    cfg = api.load_config(args.config)
    task = args.task or cfg.get("main_task", "critical")
    results_root = Path(cfg.get("results_root", "results"))
    cache_dir = cfg.get("data", {}).get("cache_dir", "data/cache")
    horizon = int(cfg.get("tlqt", {}).get("horizon_h", 6))
    cursors = (tuple(int(x) for x in args.cursors.split(",")) if args.cursors
               else tuple(range(horizon)))
    device = _pick_device(cfg)

    from wjem_ed_audit.data.dataset import make_tlqt_dataset
    ds = make_tlqt_dataset(cache_dir, task, "split_random", "test",
                           ("C", "J", "P", "D"), horizon, cfg)
    model = load_tlqt_run(results_root, task, args.config_tag, args.seed,
                          cfg, device)

    df = risk_curve_auc(model, ds, device, cursors=cursors,
                        batch_size=int(cfg.get("tlqt", {}).get("batch_size", 256)))
    df.insert(0, "task", task)
    df["config_tag"] = args.config_tag
    df["seed"] = args.seed

    tables_dir = results_root / "tables"
    figures_dir = results_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "table4_riskcurve.csv"
    out_png = figures_dir / "fig6_riskcurve.png"
    df.to_csv(out_csv, index=False)
    plot_risk_curve(df, out_png)

    for _, r in df.iterrows():
        print(f"[riskcurve] h={int(r['cursor_h'])}  AUC={r['auc']:.4f}  "
              f"(n={int(r['n'])})")
    print(f"[riskcurve] table -> {out_csv}")
    print(f"[riskcurve] figure -> {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
