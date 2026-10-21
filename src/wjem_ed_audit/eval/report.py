"""Final report generation: aggregate run metrics, benchmark table, figures, REPORT.md.

CLI:
    python -m wjem_ed_audit.eval.report --config configs/smoke.yaml

Consumes SPEC §4 run dirs (results/runs/) and SPEC §5 audit CSVs (results/audit/);
writes results/tables/{table1_performance,table2_benchmark,table3_subgroups}.csv,
results/figures/*.png and results/REPORT.md. Missing upstream artifacts are
skipped with warnings so the report works on partial results.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from wjem_ed_audit.data import api
from wjem_ed_audit.eval import calibration as cal_mod
from wjem_ed_audit.eval import dca as dca_mod
from wjem_ed_audit.eval import figures as fig_mod
from wjem_ed_audit.eval import subgroups as sg_mod

CLF_METRICS = ["auroc", "auprc", "brier", "ece"]
REG_METRICS = ["mae", "rmse", "r2"]
ALL_METRICS = CLF_METRICS + REG_METRICS

_SRC = "引自《WJEM 课题A 因果审计研究计划.md》综述的已发表数字"
PUBLISHED_BENCHMARKS = [
    ("ED 危重结局（12h 内 ICU/院内死亡）预测 AUROC", 0.894, ("critical", "auroc", "max"),
     f"MIMIC-IV-ED 危重结局预测基准 0.894（{_SRC}）"),
    ("脓毒症早期预测 AUROC", 0.92, ("critical", "auroc", "max"),
     f"脓毒症预测文献代表值 0.92（{_SRC}；本研究以 critical 任务作近似对照）"),
    ("ED 滞留时间（EDLOS）回归 R²", 0.48, ("edlos", "r2", "max"),
     f"EDLOS 回归文献代表值 R²=0.48（{_SRC}）"),
    ("HCNet 分诊预测 AUROC 区间", "0.941–0.913", ("__range__", "auroc", None),
     f"HCNet 论文报告区间 0.941–0.913（{_SRC}）"),
]


# ------------------------------------------------------------------ runs I/O
def collect_runs(results_root) -> pd.DataFrame:
    """Walk results/runs/{task}/{model}/{tag}/seed{k}[/temporal]/metrics.json."""
    runs_dir = Path(results_root) / "runs"
    rows = []
    if not runs_dir.exists():
        return pd.DataFrame()
    for mj in sorted(runs_dir.rglob("metrics.json")):
        rel = mj.relative_to(runs_dir).parts
        temporal = False
        if len(rel) == 6 and rel[4] == "temporal":
            temporal = True
        elif len(rel) != 5:
            continue
        with open(mj) as f:
            m = json.load(f)
        row = {"task": rel[0], "model": rel[1], "config_tag": rel[2],
               "seed": int(rel[3].replace("seed", "")), "temporal": temporal,
               "n_test": m.get("n_test"), "split": m.get("split")}
        for k in ALL_METRICS:
            row[k] = m.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(mean, sd) -> str:
    if pd.isna(mean):
        return ""
    sd = 0.0 if pd.isna(sd) else sd
    return f"{mean:.3f}±{sd:.3f}"


def build_table1(runs: pd.DataFrame) -> pd.DataFrame:
    """All models × tasks: metric mean±sd across seeds (random test split only)."""
    if runs.empty:
        return pd.DataFrame()
    df = runs[~runs["temporal"]]
    rows = []
    for (task, model, tag), g in df.groupby(["task", "model", "config_tag"]):
        row = {"task": task, "model": model, "config_tag": tag, "n_seeds": len(g)}
        for k in ALL_METRICS:
            v = g[k].dropna()
            row[f"{k}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else (0.0 if len(v) == 1 else np.nan)
        if task == "edlos":
            row["headline"] = _fmt(row["r2_mean"], row["r2_sd"])
        else:
            row["headline"] = _fmt(row["auroc_mean"], row["auroc_sd"])
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values(["task", "model", "config_tag"])
            .reset_index(drop=True))


def build_table2(table1: pd.DataFrame) -> pd.DataFrame:
    """Benchmark against published numbers (hardcoded, sources in `note`)."""
    def best(task, metric, how):
        if table1.empty:
            return ""
        sub = table1[table1["task"] == task]
        col = f"{metric}_mean"
        if sub.empty or sub[col].isna().all():
            return ""
        i = sub[col].idxmax() if how == "max" else sub[col].idxmin()
        r = sub.loc[i]
        return f"{r[col]:.3f} ({r['model']}/{r['config_tag']}, {int(r['n_seeds'])} seeds)"

    rows = []
    for item, published, (task, metric, how), note in PUBLISHED_BENCHMARKS:
        if task == "__range__":
            vals = table1.loc[table1["task"] != "edlos", "auroc_mean"].dropna() \
                if not table1.empty else pd.Series(dtype=float)
            ours = f"{vals.min():.3f}–{vals.max():.3f}" if len(vals) else ""
        else:
            ours = best(task, metric, how)
        rows.append({"item": item, "ours": ours, "published": published, "note": note})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ preds I/O
def load_preds_by_model(results_root, task) -> dict:
    """{model: preds DataFrame} for `task`, preferring config_tag 'full', seeds pooled."""
    base = Path(results_root) / "runs" / task
    out = {}
    if not base.exists():
        return out
    for mdir in sorted(p for p in base.iterdir() if p.is_dir()):
        tags = sorted(p.name for p in mdir.iterdir() if p.is_dir())
        if not tags:
            continue
        tag = "full" if "full" in tags else tags[0]
        frames = [pd.read_parquet(pp)
                  for pp in sorted((mdir / tag).glob("seed*/preds.parquet"))]
        if frames:
            out[mdir.name] = pd.concat(frames, ignore_index=True)
    return out


def _pooled_xy(preds: pd.DataFrame):
    """Pool seeds: mean score per stay_id."""
    g = preds.groupby("stay_id").agg(y_true=("y_true", "first"),
                                     y_score=("y_score", "mean"))
    return g["y_true"].to_numpy(), g["y_score"].to_numpy(), g.index.to_numpy()


# ------------------------------------------------------------------ markdown
def _md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _riskcurve_section(results_root: Path) -> str | None:
    """Section on the TLQ-T time-cursor risk curve; None when table4 is absent."""
    csv = Path(results_root) / "tables" / "table4_riskcurve.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    d = df.dropna(subset=["auc"]).sort_values("cursor_h")
    if d.empty:
        return None
    h0 = d[d["cursor_h"] == 0]
    first = h0.iloc[0] if not h0.empty else d.iloc[0]
    last = d.iloc[-1]
    fig_rel = "figures/fig6_riskcurve.png"
    lines = [
        "## 7. 时间游标风险曲线（TLQ-T 实时更新）\n",
        f"图：`{fig_rel}`（fig6）；数据：`tables/table4_riskcurve.csv`（table4）\n",
        f"- 分诊单点（h={int(first['cursor_h'])}）AUC = {first['auc']:.3f}",
        f"- 末点（h={int(last['cursor_h'])}）AUC = {last['auc']:.3f}"
        f"（Δ = {last['auc'] - first['auc']:+.3f}，n={int(last['n'])}）",
        "- 各游标：\n",
    ]
    lines += [f"  - h={int(r['cursor_h'])}: AUC={r['auc']:.3f}"
              for _, r in d.iterrows()]
    return "\n".join(lines)


LIMITATIONS_TEMPLATE = """## 8. 局限性（模板，投稿前须补全）

1. **【数据来源】** 数据版本与来源见配置（本报告由所给 config 自动生成）；若含合成烟测
   阶段产物，须与真实数据结果严格区分，禁止混入假设检验。
2. **【单中心回顾性】** 单一学术医疗中心、回顾性数据，外推至其他中心/前瞻性场景需再验证。
3. **【因果解释边界】** 分支遮蔽/表示干预反映的是模型*依赖结构*，非严格因果效应；
   J 组（acuity）与 C 组存在信息冗余，归因须结合 SDS/CSR 联合解读。
4. **【缺失机制】** acuity 约 20% 缺失，以缺失指示变量处理；若缺失非随机可能引入偏差。
5. **【文本嵌入】** 若 HuggingFace 不可达则降级为 hashing 嵌入，C 组文本信息保真度下降。
6. **【时间外推】** 2017–2019 外推段分诊习惯漂移既来自 DGP 设定也来自真实漂移，
   衰减量不宜全部归因于 J 组依赖。
"""


def _audit_highlights(results_root: Path, main_task: str) -> str:
    audit = results_root / "audit"
    out = []
    e1 = audit / "E1_branch_mask.csv"
    if e1.exists():
        df = pd.read_csv(e1)
        # tolerate both the audit engine schema (delta_vs_full + model column
        # + seed="ALL" summary rows) and the legacy schema (delta_auc).
        if "delta_auc" not in df.columns and "delta_vs_full" in df.columns:
            df = df.rename(columns={"delta_vs_full": "delta_auc"})
        d = df[df["task"] == main_task] if "task" in df.columns else df
        if "seed" in d.columns and (d["seed"].astype(str) == "ALL").any():
            d = d[d["seed"].astype(str) == "ALL"]
        if "model" in d.columns and d["model"].nunique() > 1:
            d = d.assign(combo=d["combo"].astype(str) + "/"
                         + d["model"].astype(str))
        g = d.groupby("combo")["delta_auc"].mean().sort_values()
        out.append("### E1 分支遮蔽（主任务 ΔAUC vs full，种子均值）\n")
        out += [f"- `{k}`: {v:+.4f}" for k, v in g.items()]
    else:
        out.append("### E1 分支遮蔽\n\n- 未运行（缺 E1_branch_mask.csv）")
    e5 = audit / "E5_transfer.csv"
    if e5.exists():
        df = pd.read_csv(e5)
        lines = ["### E5 时间外推衰减（combo=full，跨种子汇总）\n"]
        if "auc_random" in df.columns:
            # audit engine schema: use the decay_summary rows (mean + t CI)
            d = df[(df["kind"] == "decay_summary") & (df["combo"] == "full")]
            for _, r in d.iterrows():
                lines.append(
                    f"- {r['task']}/{r['model']}: 衰减={r['decay']:+.3f} "
                    f"[{r['lo']:+.3f}, {r['hi']:+.3f}], p={r['p']:.3g}")
        else:
            # legacy long schema (split/auc rows)
            if "status" in df.columns:
                df = df[df["status"].isna() | (df["status"] == "ok")]
            d = df[df["combo"] == "full"] if "combo" in df.columns else df
            for t, g in d.groupby("task"):
                piv = g.groupby("split")["auc"].mean()
                r, tt = piv.get("random_test", np.nan), piv.get("temporal_test", np.nan)
                lines.append(f"- {t}: random={r:.3f}, temporal={tt:.3f}, 衰减={r - tt:+.3f}")
        out.append("\n".join(lines))
    else:
        out.append("### E5 时间外推衰减\n\n- 未运行（缺 E5_transfer.csv）")
    sds = audit / "sds_csr.csv"
    if sds.exists():
        df = pd.read_csv(sds)
        out.append("### SDS / CSR（bootstrap 95%CI）\n\n" + _md_table(df))
    else:
        out.append("### SDS / CSR\n\n- 未运行（缺 sds_csr.csv）")
    return "\n\n".join(out)


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WJEM ED-Audit final report")
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-root", default=None, help="override cfg results_root")
    args = ap.parse_args(argv)

    cfg = api.load_config(args.config)
    results_root = Path(args.results_root or cfg.get("results_root", "results"))
    tables_dir = results_root / "tables"
    figures_dir = results_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    main_task = cfg.get("main_task", "critical")

    # ---- tables
    runs = collect_runs(results_root)
    if runs.empty:
        warnings.warn(f"[report] no runs found under {results_root}/runs")
    table1 = build_table1(runs)
    table1.to_csv(tables_dir / "table1_performance.csv", index=False)
    table2 = build_table2(table1)
    table2.to_csv(tables_dir / "table2_benchmark.csv", index=False)

    # ---- figures: audit figs
    made = fig_mod.make_all_figures(results_root, figures_dir)

    # ---- figures + subgroups on main task
    sg_table = pd.DataFrame()
    if main_task != "edlos":
        preds_by_model = load_preds_by_model(results_root, main_task)
        if preds_by_model:
            cal_input = {m: _pooled_xy(p)[:2] for m, p in preds_by_model.items()}
            made.append(cal_mod.plot_calibration(
                cal_input, figures_dir / f"calibration_{main_task}.png"))
            lo, hi, steps = cfg.get("eval", {}).get("dca_thresholds", [0.01, 0.5, 50])
            made.append(dca_mod.plot_dca(
                cal_input, np.linspace(lo, hi, int(steps)),
                figures_dir / f"dca_{main_task}.png"))
            # subgroups on best model by mean AUROC
            cand = table1[table1["task"] == main_task] if not table1.empty else table1
            if not cand.empty and cand["auroc_mean"].notna().any():
                best_model = cand.loc[cand["auroc_mean"].idxmax(), "model"]
                yt, ys, ids = _pooled_xy(preds_by_model[best_model])
                cache_dir = cfg.get("data", {}).get("cache_dir", "data/cache")
                try:
                    stays = api.load_stays(cache_dir)
                    sub = stays.set_index("stay_id").reindex(ids).reset_index()
                    sg_table = sg_mod.subgroup_performance(
                        sub, yt, ys, cfg.get("eval", {}).get("subgroups", []))
                    sg_table.to_csv(tables_dir / "table3_subgroups.csv", index=False)
                except Exception as e:  # cache may not exist yet
                    warnings.warn(f"[report] subgroup step skipped: {e}")
        else:
            warnings.warn(f"[report] no preds for main task {main_task!r}; "
                          "calibration/DCA/subgroups skipped")

    # ---- REPORT.md
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        f"# WJEM ED-Audit 总结报告\n\n生成时间：{now}  \n"
        f"配置：`{args.config}`  \n数据：`{cfg.get('data', {}).get('cache_dir', 'data/cache')}`"
        f"（source={cfg.get('data', {}).get('source', '?')}）  \n主任务：`{main_task}`\n",
        "## 1. 表1 模型性能（random test，跨种子 mean±sd）\n\n"
        + (_md_table(table1) if not table1.empty else "_无运行产物_"),
        "## 2. 表2 与已发表基准对照\n\n" + _md_table(table2),
        "## 3. 审计实验关键数字（E1–E5 / SDS-CSR）\n\n"
        + _audit_highlights(results_root, main_task),
        "## 4. 亚组性能\n\n"
        + (_md_table(sg_table) if not sg_table.empty else "_未生成（见 table3_subgroups.csv 缺失原因警告）_"),
        "## 5. 图表清单\n\n" + ("\n".join(f"- `{p}`" for p in made) or "_无_"),
        "## 6. 复现\n\n```bash\n./scripts/run_all.sh configs/smoke.yaml\n"
        "# 或单步：python -m wjem_ed_audit.eval.report "
        f"--config {args.config}\n```",
        LIMITATIONS_TEMPLATE,
    ]
    rc_section = _riskcurve_section(results_root)
    if rc_section is not None:
        parts.insert(-1, rc_section)
    report_md = results_root / "REPORT.md"
    report_md.write_text("\n\n".join(parts), encoding="utf-8")

    print(f"[report] tables -> {tables_dir}")
    print(f"[report] figures -> {figures_dir} ({len(made)} figs)")
    print(f"[report] report -> {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

