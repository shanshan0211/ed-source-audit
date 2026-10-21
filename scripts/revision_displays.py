"""Build manuscript displays from stored real-data audit summaries only.

Uses Python's standard library and TeX Live/PGFPlots; fits no models, reads no
patient records, and estimates no new confidence intervals. Original results
remain untouched. The manifest records the exact aggregate inputs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results_revision/figures_20261021"
SOURCES = {
    name: ROOT / f"results_real/audit/{name}.csv"
    for name in (
        "E1_branch_mask", "E3_repr_intervention", "E5_transfer", "E5_eicu_real"
    )
}


def read_rows(name):
    with SOURCES[name].open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def only(rows, **filters):
    selected = [row for row in rows if all(row[key] == value for key, value in filters.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected[0]


def coordinates(points):
    return " ".join(f"({x:.10f},{y:.3f})" for x, y in points)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    e1 = [row for row in read_rows("E1_branch_mask") if row["task"] == "critical"]
    e3 = [row for row in read_rows("E3_repr_intervention") if row["task"] == "critical"]
    parts = [r"""\documentclass[border=5pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\definecolor{lgbm}{HTML}{236A9C}
\definecolor{tlqt}{HTML}{C2602C}
\definecolor{control}{HTML}{64727A}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
group style={group size=1 by 2,vertical sep=2.0cm},
width=10.8cm,height=4.9cm,scale only axis,
xmin=-0.002,xmax=0.060,ymin=0.5,ymax=4.55,
ytick={1,2,3,4},xtick={0,0.01,0.02,0.03,0.04,0.05,0.06},
scaled x ticks=false,
xticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2},
tick label style={font=\small},label style={font=\small},
title style={font=\bfseries,at={(0,1.05)},anchor=south west},
axis lines=left,xmajorgrids,grid style={gray!18},clip=false,
every axis plot/.append style={line width=0.8pt},
]
\nextgroupplot[
title={A. Retrained input ablation (E1)},
yticklabels={Remove C (J+P+D),Remove J+P (C+D),Remove P (C+J+D),Remove J (C+P+D)},
xlabel={AUROC loss after retraining},
legend style={at={(0.99,0.99)},anchor=north east,font=\small,draw=none,fill=white},
]
"""]
    combos = ("meta_only", "clin_only", "noP", "noJ")
    for model, color, offset, marker in (
        ("lgbm", "lgbm", -0.13, "square*"),
        ("tlqt", "tlqt", 0.13, "*"),
    ):
        means = []
        for position, combo in enumerate(combos, 1):
            summary = only(e1, model=model, combo=combo, seed="ALL")
            losses = []
            for seed in range(5):
                fitted = only(e1, model=model, combo=combo, seed=str(seed))
                full = only(e1, model=model, combo="full", seed=str(seed))
                loss = float(full["auroc"]) - float(fitted["auroc"])
                assert abs(loss + float(fitted["delta_vs_full"])) < 1e-10
                losses.append(loss)
            mean = statistics.fmean(losses)
            assert abs(mean + float(summary["delta_vs_full"])) < 1e-10
            y = position + offset
            parts.append(
                rf"\addplot[only marks,mark={marker},mark size=1.5pt,"
                rf"color={color}!50,forget plot] coordinates {{"
                + coordinates((loss, y + (seed - 2) * 0.025) for seed, loss in enumerate(losses))
                + "};\n"
            )
            means.append((mean, y))
            parts.append(
                rf"\node[anchor=west,font=\scriptsize,text={color}] "
                rf"at (axis cs:{max(losses) + 0.0011:.10f},{y:.3f}) {{{mean:.4f}}};" + "\n"
            )
        parts.append(
            rf"\addplot[only marks,mark={marker},mark size=2.4pt,color={color}] coordinates {{"
            + coordinates(means) + "};\n"
        )
        parts.append(r"\addlegendentry{" + ("LightGBM" if model == "lgbm" else "TLQ-T") + "}\n")
    parts.append(r"""\nextgroupplot[
title={B. Fixed TLQ-T query suppression (E3)},
yticklabels={Random C queries,Suppress D,Suppress P,Suppress J},
xlabel={AUROC loss without retraining},
]
""")
    for position, intervention in enumerate(
        ("suppress_random_C", "suppress_D", "suppress_P", "suppress_J"), 1
    ):
        summary = only(e3, intervention=intervention, seed="ALL")
        values = [float(only(e3, intervention=intervention, seed=str(seed))["delta_vs_none"])
                  for seed in range(5)]
        mean, low, high = (float(summary[key]) for key in ("delta_vs_none", "lo", "hi"))
        assert abs(mean - statistics.fmean(values)) < 1e-10
        color = "control" if intervention == "suppress_random_C" else "tlqt"
        parts.append(
            rf"\addplot[only marks,mark=*,mark size=1.5pt,color={color}!50] coordinates {{"
            + coordinates((value, position + (seed - 2) * 0.055) for seed, value in enumerate(values))
            + "};\n"
        )
        parts.append(
            rf"\addplot[color={color},mark=*,mark size=2.4pt,"
            rf"error bars/.cd,x dir=both,x explicit] coordinates "
            rf"{{({mean:.10f},{position}) += ({high-mean:.10f},0) -= ({mean-low:.10f},0)}};" + "\n"
        )
        parts.append(
            rf"\node[anchor=west,font=\scriptsize,text={color}] "
            rf"at (axis cs:{high + 0.0011:.10f},{position}) {{{mean:.4f}}};" + "\n"
        )
    parts.append("\\end{groupplot}\n\\end{tikzpicture}\n\\end{document}\n")
    figure_tex = OUTPUT / "core_source_audit.tex"
    figure_tex.write_text("".join(parts), encoding="utf-8")

    # Transparent arithmetic summaries used to typeset the main E5 table.
    e5 = read_rows("E5_transfer")
    external = read_rows("E5_eicu_real")
    display_rows = []
    for model in ("tlqt", "lgbm"):
        for combo in ("full", "clin_only", "meta_only"):
            records = [row for row in e5 if row["task"] == "critical" and row["model"] == model
                       and row["combo"] == combo and row["kind"] == "detail"]
            assert len(records) == 5
            reference = statistics.fmean(float(row["auc_random"]) for row in records)
            comparison = statistics.fmean(float(row["auc_temporal"]) for row in records)
            loss = reference - comparison
            stored = only(e5, kind="decay_summary", task="critical", model=model, combo=combo, seed="ALL")
            assert abs(loss - float(stored["decay"])) < 1e-10
            display_rows.append({"setting": "anchor_period", "model": model, "combo": combo,
                                 "reference_auroc": reference, "comparison_auroc": comparison,
                                 "loss": loss, "comparison_ci_lo": "", "comparison_ci_hi": ""})
    for model in ("lr", "lgbm"):
        row = only(external, task="death", model=model, seed="0")
        display_rows.append({"setting": "icu_anchored_eicu", "model": model, "combo": "clin_only",
                             "reference_auroc": float(row["auc_random_mimic"]),
                             "comparison_auroc": float(row["auc_eicu_real"]), "loss": float(row["decay"]),
                             "comparison_ci_lo": float(row["lo"]), "comparison_ci_hi": float(row["hi"])})
    with (OUTPUT / "E5_display_values.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(display_rows[0]))
        writer.writeheader()
        writer.writerows(display_rows)
    manifest = {
        "purpose": "Main-paper displays of stored real-data aggregate results; no model rerun",
        "figure": "A: five per-seed E1 losses and means, no confidence intervals; "
                  "B: five per-seed E3 losses, means and archived seed-t intervals (df=4)",
        "uncertainty": "E3 intervals describe seed variability, not patient sampling. "
                       "E5 eICU intervals are the archived seed-0 stay-bootstrap intervals.",
        "inputs": [{"file": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in SOURCES.values()],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    completed = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", figure_tex.name],
        cwd=OUTPUT, capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout[-6000:] + completed.stderr)
    print(f"Figure: {figure_tex.with_suffix('.pdf')}")
    for row in display_rows:
        print(f"{row['setting']:18} {row['model']:5} {row['combo']:10} "
              f"{row['reference_auroc']:.4f} {row['comparison_auroc']:.4f} {row['loss']:+.5f}")


if __name__ == "__main__":
    main()
