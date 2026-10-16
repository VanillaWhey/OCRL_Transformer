"""plot_box_iqm.py with consistent sizes in the paper (same rules as Eval/scripts/csv/scaled.py).

Same statistics and style as plot_box_iqm.py, but everything drawn (plot, y label, legend) is exactly
``--scale x textwidth`` wide in the paper (TMLR: 6.5 in): the base font is ``--font-size`` pt (default 8), line widths,
whiskers and caps scale with it, every bar is ``--bar-width`` inches wide, every legend handle ``--handle-length`` pt
long, and the legend uses as many columns as fit into the width. The figure is drawn magnified with the base font at
``--render-font`` pt (default 10) and scaled down by LaTeX, so include it with an explicit width:
``\includegraphics[width=<scale>\linewidth]{...}``.

    python -m cleanrl_experiments.eval.plot_box_iqm_scaled --scale 0.4 --no-delta
    python -m cleanrl_experiments.eval.plot_box_iqm_scaled --scale 0.5 --models ORBiT -o results/box_iqm_orbit.pdf
"""
import argparse
import os

import torch  # noqa: F401  (loads the OpenMP runtime numpy.linalg needs in the OCRL_Transformer env)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cleanrl_experiments.eval.plot_box_iqm import (FRAME_COLORS, FRAME_LABELS, MODEL_LABELS, _ECOLOR, _ELW, _FIG_H,
                                                   _FIG_W, _apply_style, iqm_with_ci, run_scores)

TEXTWIDTH = 6.5  # in, TMLR
BASE_FONT = 14  # font.size of _apply_style
PAD = 0.01  # in, white margin around the cropped figure
_FONT_KEYS = ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize",
              "legend.fontsize")
_LINE_KEYS = ("axes.linewidth", "grid.linewidth", "xtick.major.width", "ytick.major.width", "xtick.major.size",
              "ytick.major.size", "xtick.major.pad", "ytick.major.pad", "axes.labelpad")


def _apply_scaled_style(f):
    _apply_style()
    for key in _FONT_KEYS + _LINE_KEYS:
        mpl.rcParams[key] = mpl.rcParams[key] * f


def plot(stats, models, frames, delta, metric, out, scale, font_size, bar_width, height, legend_cols=None,
         render_font=10.0, handle_length=12.0):
    m = render_font / font_size  # magnification, undone by LaTeX
    f = render_font / BASE_FONT
    _apply_scaled_style(f)
    target_w = (scale * TEXTWIDTH - 2 * PAD) * m  # width of everything drawn; savefig adds the margins
    target_h = target_w * (height or _FIG_H / _FIG_W)
    fig, ax = plt.subplots(figsize=(scale * TEXTWIDTH * m, scale * TEXTWIDTH * m * (height or _FIG_H / _FIG_W)),
                           layout="none")
    gap = 0.11  # bar positions as in plot_box_iqm.py (data units); the bar width is set physically below
    n_bars = max(sum((m, f_) in stats for f_ in frames) for m in models)
    group_step = n_bars * (0.42 + gap) + 0.3
    centers = np.arange(len(models)) * group_step
    xs, bars = {}, {}
    for model, xc in zip(models, centers):
        present = [fr for fr in frames if (model, fr) in stats]
        for j, fr in enumerate(present):
            x = xc + (j - (len(present) - 1) / 2) * (0.42 + gap)
            iqm, lo, hi, _ = stats[(model, fr)]
            xs[(model, fr)] = x
            bars[(model, fr)] = ax.bar(x, iqm, 0.42, color=FRAME_COLORS.get(fr, "#333333"), alpha=0.82,
                                       zorder=4)[0]
            ax.errorbar(x, iqm, yerr=[[iqm - lo], [hi - iqm]], capsize=4 * f, ecolor=_ECOLOR, ls="", lw=_ELW * f,
                        mew=0.8 * f, zorder=5)
    pad = 0.04 * max(hi for _, _, hi, _ in stats.values())
    top = max(hi for _, _, hi, _ in stats.values())
    a, b = delta or (None, None)
    if delta:
        for model in models:
            if (model, a) in xs and (model, b) in xs:
                top = max(top, max(stats[(model, a)][2], stats[(model, b)][2]) + 5 * pad)
    ax.set_xticks(centers)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], ha="center")
    ax.set_xlim(centers[0] - group_step / 2, centers[-1] + group_step / 2)
    ax.set_ylim(0, top * 1.03)
    ax.set_ylabel("Catches IQM" if metric == "catches" else "Score IQM")

    used = [fr for fr in frames if any((m, fr) in stats for m in models)]
    handles = [mpl.patches.Patch(facecolor=FRAME_COLORS.get(fr, "#333333"), alpha=0.82, label=FRAME_LABELS.get(fr, fr))
               for fr in used]
    renderer = fig.canvas.get_renderer()
    ncol = legend_cols or len(handles)  # as many columns as fit into the width
    while True:
        legend_kw = dict(handles=handles, loc="upper center", ncol=ncol, frameon=False,
                         handlelength=handle_length * m / mpl.rcParams["legend.fontsize"],
                         handletextpad=0.4, columnspacing=0.8, borderaxespad=0)
        probe = fig.legend(**legend_kw)
        fits = probe.get_window_extent(renderer).width / fig.dpi <= target_w
        probe.remove()
        if fits or ncol == 1 or legend_cols:
            break
        ncol -= 1

    legend = None
    w, h = fig.get_size_inches()
    for _ in range(10):  # resize until plot + labels + legend have the target size
        fig.set_size_inches(w, h)
        if legend is not None:
            legend.remove()
        fig.tight_layout(pad=0.2)
        renderer = fig.canvas.get_renderer()
        legend_h = fig.legend(**legend_kw).get_window_extent(renderer).height
        fig.legends[-1].remove()
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + (legend_h + 4 * m / 72 * fig.dpi) / fig.bbox.height)
        full = ax.get_tightbbox(renderer)
        anchor = fig.transFigure.inverted().transform(((full.x0 + full.x1) / 2,
                                                      ax.xaxis.get_tightbbox(renderer).y0 - 4 * m / 72 * fig.dpi))
        legend = fig.legend(bbox_to_anchor=anchor, bbox_transform=fig.transFigure, **legend_kw)
        bb = fig.get_tightbbox(renderer)
        if abs(bb.width - target_w) < 0.002 and abs(bb.height - target_h) < 0.005:
            break
        w, h = w + target_w - bb.width, h + target_h - bb.height

    # bars of the same physical width (at most 92% of the distance between neighbouring bars)
    per_inch = np.diff(ax.get_xlim())[0] / (ax.get_position().width * fig.get_figwidth())
    width = min(bar_width * m * per_inch, 0.92 * (0.42 + gap))
    for key, bar in bars.items():
        bar.set_x(xs[key] - width / 2)
        bar.set_width(width)
    if delta:  # Δ brackets over the two bars, as in plot_box_iqm.py
        for model in models:
            if (model, a) in xs and (model, b) in xs:
                y = max(stats[(model, a)][2], stats[(model, b)][2]) + pad
                d = stats[(model, b)][0] - stats[(model, a)][0]
                ax.plot([xs[(model, a)] - width / 2, xs[(model, b)] + width / 2], [y, y], color="black",
                        lw=1.4 * f, solid_capstyle="round", zorder=7)
                ax.text((xs[(model, a)] + xs[(model, b)]) / 2, y + pad / 3,
                        rf"$\Delta\,{d:+.{1 if metric == 'catches' else 2}f}$", ha="center", va="bottom",
                        color="black", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=PAD * m)
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    print(f"saved {out}: {bb.width / m + 2 * PAD:.2f} x {bb.height / m + 2 * PAD:.2f} in in the paper ({scale} x "
          f"{TEXTWIDTH} in), drawn {m:.2f} x larger, legend in {ncol} column(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="results/results_box.csv")
    parser.add_argument("-o", "--out", default="results/box_iqm_scaled.pdf")
    parser.add_argument("--scale", type=float, required=True, help="width in LaTeX as a fraction of \\linewidth")
    parser.add_argument("--font-size", type=float, default=8.0, help="base font size in the paper (pt)")
    parser.add_argument("--render-font", type=float, default=10.0,
                        help="pt the base font is drawn at before LaTeX scales the figure down")
    parser.add_argument("--handle-length", type=float, default=12.0, help="legend handle length in the paper (pt)")
    parser.add_argument("--bar-width", type=float, default=0.235, help="bar width in the paper in inches")
    parser.add_argument("--height", type=float, default=None, help="height / width (default: as plot_box_iqm.py)")
    parser.add_argument("--n-distractors", type=int, default=5, help="which evaluation setting to plot")
    parser.add_argument("--n-landmarks", type=int, default=2, help="which evaluation setting to plot")
    parser.add_argument("--models", nargs="*", default=["ORBiT", "PPObj"])
    parser.add_argument("--frames", nargs="*", default=["abs", "agent", "goal", "distractor", "landmark"])
    parser.add_argument("--delta", nargs=2, default=["abs", "agent"], metavar=("FROM", "TO"),
                        help="the Δ above each group is IQM(TO) - IQM(FROM)")
    parser.add_argument("--no-delta", action="store_const", const=None, dest="delta", help="no Δ brackets")
    parser.add_argument("--metric", choices=["catches", "score"], default="catches")
    parser.add_argument("--reps", type=int, default=50_000, help="bootstrap repetitions")
    parser.add_argument("--legend-cols", type=int, default=None, help="legend columns (default: as many as fit)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "n_landmarks" not in df:
        df["n_landmarks"] = 2
    df = df[(df["n_distractors"] == args.n_distractors) & (df["n_landmarks"] == args.n_landmarks)
            & df["model"].isin(args.models) & df["frame"].isin(args.frames)]
    stats = iqm_with_ci(run_scores(df), args.metric, args.reps)
    plot(stats, args.models, args.frames, args.delta, args.metric, args.out, args.scale, args.font_size,
         args.bar_width, args.height, args.legend_cols, args.render_font, args.handle_length)
