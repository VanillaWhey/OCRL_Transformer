"""IQM bar plot with 95% bootstrap confidence intervals (rliable) from the CSV of eval_box_csv.py.

Every run gives one value: its mean catches per episode (--metric catches, the default), or its normalized score
(mean catches - random) / (chase - random) (--metric score: 0 = random, 1 = chase). Per model and frame, rliable
computes the interquartile mean over the runs (seeds) and its 95% stratified-bootstrap confidence interval.
Bars are grouped by model, colored by frame; the Δ above a group is IQM(TO) - IQM(FROM) of --delta FROM TO
(default: agent frame minus absolute frame), --no-delta leaves it out.

    python -m cleanrl_experiments.eval.plot_box_iqm results/results_box.csv -o results/box_iqm.pdf
"""
import argparse
import os

# Load torch's OpenMP runtime first: in the OCRL_Transformer env the MKL one (libiomp5md.dll) is renamed
# (workaround for "OMP: Error #15"), and numpy.linalg, which matplotlib uses, crashes without it.
import torch  # noqa: F401
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics
from tueplots import bundles

MODEL_LABELS = {"ORBiT": "OCT/ORBiT", "PPObj": "PPObj"}
FRAME_LABELS = {"abs": "Absolute", "agent": "Player (ORCA)", "goal": "Target", "distractor": "Distractor",
                "landmark": "Landmark"}
FRAME_COLORS = {"abs": "#6087AA", "agent": "#B83232", "goal": "#D69A3C", "distractor": "#8A6FA9",
                "landmark": "#4F9B8B"}

# Style of the paper's figures (Eval/scripts/csv/plot_aggregated.py): TMLR bundle (6.5 in text width, Computer
# Modern via LaTeX), thin axes and grid, bars without edges at alpha 0.82, thin near-black CI whiskers.
_BUNDLE = bundles.tmlr2023()
_FIG_W = _BUNDLE["figure.figsize"][0] * 1.2 / 0.92 / 2  # as plot_first.py: one panel
_FIG_H = _BUNDLE["figure.figsize"][0] * 0.75 * 0.75
_ECOLOR = "#222222"  # CI whiskers
_ELW = 0.8  # CI whisker line width


def _apply_style():
    plt.rcParams.update(_BUNDLE)
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.4,
        "savefig.pad_inches": 0.015,
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

def run_scores(df):
    """One score per run: mean catches over its episodes, normalized by random and chase."""
    per_run = df.groupby(["model", "frame", "run_id"]).agg(catches=("catches", "mean"), random=("random", "first"),
                                                          chase=("chase", "first")).reset_index()
    per_run["score"] = (per_run["catches"] - per_run["random"]) / (per_run["chase"] - per_run["random"])
    return per_run


def iqm_with_ci(per_run, metric, reps):
    """{(model, frame): (iqm, lower, upper, number of runs)} of the per-run ``metric`` via rliable."""
    score_dict = {f"{m}|{f}": g[metric].to_numpy()[:, None] for (m, f), g in per_run.groupby(["model", "frame"])}
    point, cis = rly.get_interval_estimates(score_dict, lambda x: np.array([metrics.aggregate_iqm(x)]), reps=reps)
    return {tuple(k.split("|")): (point[k][0], cis[k][0, 0], cis[k][1, 0], len(score_dict[k])) for k in score_dict}


def plot(stats, models, frames, delta, metric, out, legend_cols=None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), layout="none")
    width, gap = 0.42, 0.11  # bar width and gap between the bars of a group
    n_bars = max(sum((m, f) in stats for f in frames) for m in models)
    group_step = n_bars * (width + gap) + 0.3  # distance between group centers
    pad = 0.04 * max(hi for _, _, hi, _ in stats.values())  # vertical spacing of the Δ brackets

    centers = np.arange(len(models)) * group_step
    top = max(hi for _, _, hi, _ in stats.values())
    for model, xc in zip(models, centers):
        present = [f for f in frames if (model, f) in stats]
        xs = {f: xc + (j - (len(present) - 1) / 2) * (width + gap) for j, f in enumerate(present)}
        for f, x in xs.items():
            iqm, lo, hi, _ = stats[(model, f)]
            ax.bar(x, iqm, width, color=FRAME_COLORS.get(f, "#333333"), alpha=0.82, zorder=4)
            ax.errorbar(x, iqm, yerr=[[iqm - lo], [hi - iqm]], capsize=4, ecolor=_ECOLOR, ls="", lw=_ELW, mew=0.8,
                        zorder=5)
        a, b = delta or (None, None)
        if a in xs and b in xs:  # Δ bracket over the two bars, as in plot_first.py
            y = max(stats[(model, a)][2], stats[(model, b)][2]) + pad
            d = stats[(model, b)][0] - stats[(model, a)][0]
            ax.plot([xs[a] - width / 2, xs[b] + width / 2], [y, y], color="black", lw=1.4, solid_capstyle="round",
                    zorder=7)
            ax.text((xs[a] + xs[b]) / 2, y + pad / 3, rf"$\Delta\,{d:+.{1 if metric == 'catches' else 2}f}$",
                    ha="center", va="bottom", color="black", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))
            top = max(top, y + 4 * pad)  # room for the label

    ax.set_xticks(centers)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], ha="center")
    ax.set_xlim(centers[0] - group_step / 2, centers[-1] + group_step / 2)
    ax.set_ylim(0, top * 1.03)
    ax.set_ylabel("Catches IQM" if metric == "catches" else "Score IQM")

    used = [f for f in frames if any((m, f) in stats for m in models)]
    handles = [mpl.patches.Patch(facecolor=FRAME_COLORS.get(f, "#333333"), alpha=0.82, label=FRAME_LABELS.get(f, f))
               for f in used]
    legend_kw = dict(handles=handles, loc="upper center", ncol=legend_cols or len(handles), frameon=False,
                     handlelength=0.9, handletextpad=0.4, columnspacing=0.8, borderaxespad=0)
    # Resize the figure so that the plot including its y labels is exactly as wide as the legend (with a single
    # legend row the legend is wider, with several rows the plot may be narrowed), then put the legend centred
    # under it, right below the x labels.
    legend = ax.legend(**legend_kw)
    renderer = fig.canvas.get_renderer()
    legend_w = legend.get_window_extent(renderer).width
    legend.remove()
    for _ in range(5):
        fig.tight_layout(pad=0.2)
        renderer = fig.canvas.get_renderer()
        diff = legend_w - ax.get_tightbbox(renderer).width  # pixels
        if abs(diff) < 1:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + diff / fig.dpi, h)
    renderer = fig.canvas.get_renderer()
    full = ax.get_tightbbox(renderer)  # plot area + tick labels + axis labels
    gap = 4 / 72 * fig.dpi  # 4 pt between the x labels and the legend
    anchor = fig.transFigure.inverted().transform(((full.x0 + full.x1) / 2, ax.xaxis.get_tightbbox(renderer).y0 - gap))
    fig.legend(bbox_to_anchor=anchor, bbox_transform=fig.transFigure, **legend_kw)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="results/results_box.csv")
    parser.add_argument("-o", "--out", default="results/box_iqm.pdf")
    parser.add_argument("--n-distractors", type=int, default=5, help="which evaluation setting to plot")
    parser.add_argument("--n-landmarks", type=int, default=2, help="which evaluation setting to plot")
    parser.add_argument("--models", nargs="*", default=["ORBiT", "PPObj"])
    parser.add_argument("--frames", nargs="*", default=["abs", "agent", "goal", "distractor", "landmark"])
    parser.add_argument("--delta", nargs=2, default=["abs", "agent"], metavar=("FROM", "TO"),
                        help="the Δ above each group is IQM(TO) - IQM(FROM)")
    parser.add_argument("--no-delta", action="store_const", const=None, dest="delta", help="no Δ brackets")
    parser.add_argument("--metric", choices=["catches", "score"], default="catches",
                        help="catches per episode, or normalized score (0 = random, 1 = chase)")
    parser.add_argument("--reps", type=int, default=50_000, help="bootstrap repetitions")
    parser.add_argument("--legend-cols", type=int, default=None, help="legend columns (default: all in one row)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "n_landmarks" not in df:  # CSV from before landmarks could be varied
        df["n_landmarks"] = 2
    df = df[(df["n_distractors"] == args.n_distractors) & (df["n_landmarks"] == args.n_landmarks)
            & df["model"].isin(args.models) & df["frame"].isin(args.frames)]
    stats = iqm_with_ci(run_scores(df), args.metric, args.reps)
    print(f"{'model':8s} {'frame':11s} {'runs':>4s} {args.metric + ' IQM':>12s}   95% CI")
    for m in args.models:
        for f in args.frames:
            if (m, f) in stats:
                iqm, lo, hi, n = stats[(m, f)]
                print(f"{m:8s} {f:11s} {n:4d} {iqm:12.3f}   [{lo:.3f}, {hi:.3f}]")
    plot(stats, args.models, args.frames, args.delta, args.metric, args.out, args.legend_cols)
