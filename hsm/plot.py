"""Draw the README charts from the saved results. Nothing is re-run.

    python -m hsm.plot [--results-dir results] [--out-dir results/figures]

Reads results/results.json for each arm's mean scores and the paired-test
p-values, and results/per_query.csv (when present) for a 95% confidence
interval on each arm's mean. Writes two PNGs:

    ndcg10.png    nDCG@10 per arm, with 95% confidence intervals
    metrics.png   nDCG@10, Recall@10 and MRR@10 per arm, grouped
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARMS = ["bm25", "dense", "rrf", "rrf_rerank"]
LABELS = {
    "bm25": "BM25 full-text",
    "dense": "Dense vectors (bge-small)",
    "rrf": "Hybrid: RRF fusion",
    "rrf_rerank": "Hybrid + rerank (experiment)",
}
GROUPED = [("ndcg@10", "nDCG@10"), ("recall@10", "Recall@10"), ("mrr@10", "MRR@10")]

# Light theme. Series colours are the first three slots of a palette checked
# for colour-vision-deficiency separation; text never wears a series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
WHISKER = "#0d366b"
DPI = 200            # an 8 inch figure is 1600 px, sharp when shown at 800 px
WIDTH_IN = 8.0


def load_results(results_dir: Path) -> dict:
    return json.loads((results_dir / "results.json").read_text())


def arm_intervals(csv_path: Path, means: dict) -> dict:
    """95% t-interval on each arm's mean, from the saved per-query scores.

    Returns {} when the per-query file is absent (the charts then omit the
    whiskers). Raises if the file disagrees with results.json, so a stale
    per-query file can never draw intervals around the wrong means."""
    if not csv_path.exists():
        return {}
    from scipy import stats

    scores: dict = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            arm = row["arm"]
            for metric in means.get(arm, {}):
                scores.setdefault(arm, {}).setdefault(metric, []).append(float(row[metric]))
    out: dict = {}
    for arm, by_metric in scores.items():
        for metric, xs in by_metric.items():
            n = len(xs)
            mean = sum(xs) / n
            if abs(mean - means[arm][metric]) > 1e-5:
                raise ValueError(
                    f"{csv_path} gives {arm} {metric} = {mean:.6f} but results.json "
                    f"has {means[arm][metric]:.6f}; the files come from different runs"
                )
            sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
            half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
            out.setdefault(arm, {})[metric] = (mean - half, mean + half)
    return out


def fmt_p(p: float) -> str:
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"


def comparison(res: dict, a: str, b: str, metric: str) -> dict:
    for c in res["significance"]["comparisons"]:
        if c["a"] == a and c["b"] == b and c["metric"] == metric:
            return c
    raise KeyError(f"no paired test for {a} vs {b} on {metric} in results.json")


def verdict(res: dict, a: str, b: str, metric: str, alpha: float = 0.05) -> str:
    """One clause per paired test, phrased from the sign and the p-value."""
    c = comparison(res, a, b, metric)
    name = {"bm25": "BM25", "dense": "dense", "rrf": "hybrid", "rrf_rerank": "rerank"}
    if c["p_value"] >= alpha:
        return f"{name[a]} vs {name[b]} not significant ({fmt_p(c['p_value'])})"
    win, lose = (a, b) if c["diff"] > 0 else (b, a)
    return f"{name[win]} beat {name[lose]} ({fmt_p(c['p_value'])})"


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "svg.hashsalt": "hsm",
    })


def frame(ax, value_axis: str) -> None:
    """Recessive frame: hairline grid on the value axis, one baseline, no box."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    base = "left" if value_axis == "x" else "bottom"
    ax.spines[base].set_visible(True)
    ax.spines[base].set_color(BASELINE)
    ax.spines[base].set_linewidth(0.8)
    ax.grid(axis=value_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)


def header(fig, title: str, subtitle: str, top: float) -> None:
    fig.text(0.02, top, title, fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.02, top - 0.30 / fig.get_figheight(), subtitle,  # 0.3 inch below the title
             fontsize=9.5, color=INK_2, va="top")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # No software or date stamp, so re-rendering the same results gives the same file.
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def plot_ndcg10(res: dict, ci: dict, path: Path) -> None:
    means = [res["metrics"][a]["ndcg@10"] for a in ARMS]
    n = res["dataset"]["n_queries"]
    has_ci = all("ndcg@10" in ci.get(a, {}) for a in ARMS)

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.7))
    fig.subplots_adjust(left=0.29, right=0.95, top=0.77, bottom=0.26)
    y = list(range(len(ARMS)))
    ax.barh(y, means, height=0.5, color=SERIES[0], zorder=2)
    tips = means
    if has_ci:
        lo = [ci[a]["ndcg@10"][0] for a in ARMS]
        hi = [ci[a]["ndcg@10"][1] for a in ARMS]
        ax.errorbar(means, y, xerr=[[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]],
                    fmt="none", ecolor=WHISKER, elinewidth=1.3, capsize=4, capthick=1.3, zorder=3)
        tips = hi
    for yi, tip, m in zip(y, tips, means):
        ax.text(tip + 0.008, yi, f"{m:.4f}", va="center", ha="left",
                fontsize=10.5, fontweight="bold", color=INK)
    ax.set_yticks(y, [LABELS[a] for a in ARMS], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.5)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.tick_params(axis="x", labelsize=9)
    frame(ax, "x")

    whiskers = " Whiskers: 95% confidence interval of each mean." if has_ci else ""
    header(fig, "nDCG@10 on BEIR FiQA-2018",
           f"Mean over {n} test queries, higher is better.{whiskers}", top=0.955)
    note = (f"Overlapping whiskers are not a test. Paired t-tests over the same {n} queries: "
            + verdict(res, "rrf", "bm25", "ndcg@10") + ",\n"
            + verdict(res, "rrf", "dense", "ndcg@10") + ", "
            + verdict(res, "rrf_rerank", "rrf", "ndcg@10") + ".")
    fig.text(0.02, 0.035, note, fontsize=8.5, color=INK_2, va="bottom", linespacing=1.45)
    save(fig, path)


def plot_metrics(res: dict, path: Path) -> None:
    n = res["dataset"]["n_queries"]
    k = len(GROUPED)
    bar, gap, group_gap = 0.26, 0.03, 0.42
    step = k * bar + (k - 1) * gap + group_gap

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.0))
    fig.subplots_adjust(left=0.29, right=0.95, top=0.76, bottom=0.08)
    centres = []
    for i, arm in enumerate(ARMS):
        top = i * step
        centres.append(top + (k * bar + (k - 1) * gap) / 2)
        for j, (metric, _) in enumerate(GROUPED):
            yj = top + j * (bar + gap) + bar / 2
            v = res["metrics"][arm][metric]
            ax.barh(yj, v, height=bar, color=SERIES[j], zorder=2)
            ax.text(v + 0.006, yj, f"{v:.4f}", va="center", ha="left", fontsize=8.5, color=INK_2)
    ax.set_yticks(centres, [LABELS[a] for a in ARMS], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.55)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.tick_params(axis="x", labelsize=9)
    frame(ax, "x")

    header(fig, "Three ranking metrics per arm",
           f"Mean over {n} BEIR FiQA-2018 test queries, higher is better.", top=0.965)
    handles = [Patch(facecolor=c, label=lbl) for c, (_, lbl) in zip(SERIES, GROUPED)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.865), ncol=k,
               frameon=False, fontsize=9.5, handlelength=1.0, handleheight=1.0,
               columnspacing=1.6, labelcolor=INK)
    save(fig, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hsm.plot",
                                 description="Draw the README charts from saved results.")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out-dir", default=None, help="default: <results-dir>/figures")
    args = ap.parse_args(argv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"

    res = load_results(results_dir)
    ci = arm_intervals(results_dir / "per_query.csv", res["metrics"])
    if not ci:
        print(f"[plot] {results_dir / 'per_query.csv'} not found; drawing without intervals",
              file=sys.stderr)
    style()
    plot_ndcg10(res, ci, out_dir / "ndcg10.png")
    plot_metrics(res, out_dir / "metrics.png")
    for name in ("ndcg10.png", "metrics.png"):
        print(f"[plot] wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
