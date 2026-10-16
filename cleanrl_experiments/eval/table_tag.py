"""Results table for Tag (box): IQM and 95% CI per configuration, from the CSV of eval_box_csv.py.

Columns: ORCA (anchored frames) with ORBiT and PPObj, absolute coordinates with OCT (= the ORBiT architecture on
absolute coordinates) and PPObj. Rows: the base setting (5 distractors, 2 landmarks, player anchor), the
player-anchored agents evaluated with other numbers of distractors or landmarks, and the other anchors. PPObj cannot be evaluated with another
number of objects than in training, and absolute coordinates have no anchor; those cells are "---".

--level run (default, as plot_box_iqm.py) averages the 100 episodes of every seed and takes rliable's IQM and
95% stratified-bootstrap CI over the 5 seed means. --level episode pools all episodes of all seeds (a seeds x
episodes matrix); rliable then treats the episodes like tasks, which gives much narrower CIs that understate the
spread between seeds.

    python -m cleanrl_experiments.eval.table_tag
    python -m cleanrl_experiments.eval.table_tag --metric score
"""
import argparse

import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics

# (row label, evaluated numbers of distractors and landmarks, frame of the ORCA columns,
#  whether the PPObj / absolute-coordinates columns apply)
ROWS = [
    (r"Base ($m = 5$, Anchor: Player)", 5, 2, "agent", True, True),
    (r"$m' = 0$ distractors", 0, 2, "agent", False, True),
    (r"$m' = 1$ distractors", 1, 2, "agent", False, True),
    (r"$m' = 10$ distractors", 10, 2, "agent", False, True),
    (r"No landmarks", 5, 0, "agent", False, True),
    (r"4 landmarks", 5, 4, "agent", False, True),
    (r"Anchor: Target", 5, 2, "goal", True, False),
    (r"Anchor: Distractor", 5, 2, "distractor", True, False),
    (r"Anchor: Landmark", 5, 2, "landmark", True, False),
]
# (column header, model, frame: None = the row's ORCA frame, "abs" = absolute coordinates)
COLUMNS = [("ORBiT", "ORBiT", None), ("PPObj", "PPObj", None), ("OCT", "ORBiT", "abs"), ("PPObj", "PPObj", "abs")]


def cell_values(df, model, frame, n, landmarks, metric):
    """Score matrix (seeds x episodes) of one cell, or None if there is no data."""
    d = df[(df["model"] == model) & (df["frame"] == frame) & (df["n_distractors"] == n)
           & (df["n_landmarks"] == landmarks)]
    if d.empty:
        return None
    if metric == "score":
        d = d.assign(catches=(d["catches"] - d["random"]) / (d["chase"] - d["random"]))
    return d.pivot_table(index="run_id", columns="episode", values="catches").to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="results/results_box.csv")
    parser.add_argument("--metric", choices=["catches", "score"], default="catches",
                        help="catches per episode, or normalized score (0 = random, 1 = chase)")
    parser.add_argument("--level", choices=["run", "episode"], default="run",
                        help="IQM over the per-seed means, or over all episodes of all seeds")
    parser.add_argument("--reps", type=int, default=50_000, help="bootstrap repetitions")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "n_landmarks" not in df:  # CSV from before landmarks could be varied
        df["n_landmarks"] = 2
    cells, matrices = {}, {}
    for r, (_, n, landmarks, orca_frame, ppobj_ok, abs_ok) in enumerate(ROWS):
        for c, (_, model, frame) in enumerate(COLUMNS):
            if (model == "PPObj" and not ppobj_ok) or (frame == "abs" and not abs_ok):
                continue
            m = cell_values(df, model, frame or orca_frame, n, landmarks, args.metric)
            if m is not None:
                matrices[f"{r}|{c}"] = m if args.level == "episode" else m.mean(axis=1, keepdims=True)
    point, cis = rly.get_interval_estimates(matrices, lambda x: np.array([metrics.aggregate_iqm(x)]),
                                            reps=args.reps)
    for key in matrices:
        r, c = map(int, key.split("|"))
        cells[(r, c)] = (point[key][0], cis[key][0, 0], cis[key][1, 0], matrices[key].shape[0])

    digits = 1 if args.metric == "catches" else 2
    fmt = lambda v: f"{v:.{digits}f}"  # noqa: E731

    # readable table
    header = ["Configuration"] + [f"{'ORCA' if f is None else 'Abs'} {h}" for h, _, f in COLUMNS]
    widths = [32] + [22] * len(COLUMNS)
    print(f"{args.metric} IQM [95% CI], level={args.level}, seeds per cell in ()")
    print("".join(h.ljust(w) for h, w in zip(header, widths)))
    for r, (label, *_rest) in enumerate(ROWS):
        row = [label]
        for c in range(len(COLUMNS)):
            if (r, c) in cells:
                v, lo, hi, n = cells[(r, c)]
                row.append(f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}] ({n})")
            else:
                row.append("---")
        print("".join(s.ljust(w) for s, w in zip(row, widths)))

    # LaTeX rows for the table body
    print("\n% LaTeX rows")
    for r, (label, *_rest) in enumerate(ROWS):
        entries = []
        for c in range(len(COLUMNS)):
            if (r, c) in cells:
                v, lo, hi, _ = cells[(r, c)]
                entries.append(f"{fmt(v)}\\,{{\\scriptsize[{fmt(lo)}, {fmt(hi)}]}}")
            else:
                entries.append("---")
        print(f"    {label:<34s}& " + " & ".join(entries) + r" \\")


if __name__ == "__main__":
    main()
