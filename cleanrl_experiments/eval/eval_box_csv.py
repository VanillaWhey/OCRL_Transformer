"""Evaluate the box runs of W&B projects and store the raw results in a CSV file (nothing is uploaded).

For every finished run trained on tag/box-v0, the latest model artifact is downloaded and evaluated in the box
(deterministic policy, full 500-step re-spawning episodes). One row per episode:

    project, run_id, run_name, model, frame, seed, n_distractors, n_landmarks, episode, catches, random, chase

model is ORBiT or PPObj, frame the reference frame (abs, agent, goal, distractor, landmark), catches the catches
in that episode, and random / chase the mean catches per episode of the reference policies in the same setting
(from tag/baselines.json), so that score = (catches - random) / (chase - random). Every combination of -d
(distractors) and -l (landmarks) is evaluated; both default to the training numbers. Settings already in the CSV
are skipped, so an interrupted evaluation can be resumed; plot_box_iqm.py and table_tag.py read the CSV.

    python -m cleanrl_experiments.eval.eval_box_csv -p OCRL_Transformer PPObj-v2 -f config.total_timesteps=3000000
    python -m cleanrl_experiments.eval.eval_box_csv -p OCRL_Transformer -f config.total_timesteps=3000000 -l 0 4

Outputs (CSV, downloaded models) go to results/ in the repository root, which git ignores.
"""
import argparse
import csv
import itertools
import os

import torch
import wandb
from rtpt import RTPT

from cleanrl_experiments.eval.eval_tag import load_agent
from cleanrl_experiments.eval.eval_wandb_tag import ParseKwargs, download_model
from cleanrl_experiments.tag import policies
from cleanrl_experiments.tag.tag_setup import WORLDS, baseline, world_setting, wrap

COLUMNS = ["project", "run_id", "run_name", "model", "frame", "seed", "n_distractors", "n_landmarks", "episode",
           "catches", "random", "chase"]
TRAIN_LANDMARKS = world_setting("box", "n_landmarks")  # the training scripts always use the world's landmarks


def read_done(path):
    """(run_id, n_distractors, n_landmarks) already in the CSV; adds the n_landmarks column to an older CSV."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows, fields = list(reader), reader.fieldnames
    if "n_landmarks" not in fields:  # written before landmarks could be varied: all with the training number
        for row in rows:
            row["n_landmarks"] = TRAIN_LANDMARKS
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"added the n_landmarks column ({TRAIN_LANDMARKS}) to {path}")
    return {(row["run_id"], int(row["n_distractors"]), int(row["n_landmarks"])) for row in rows}


if __name__ == '__main__':
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument('-e', '--wandb_entity', type=str, default=None, help="default: your W&B default entity")
    argsparser.add_argument('-p', '--projects', action="extend", nargs='*', type=str)
    argsparser.add_argument('-f', '--filters', nargs='*', action=ParseKwargs, default={})
    argsparser.add_argument('-d', '--distractors', nargs='*', type=int, default=None,
                            help="numbers of distractors to evaluate (default: the training number only)")
    argsparser.add_argument('-l', '--landmarks', nargs='*', type=int, default=None,
                            help=f"numbers of landmarks to evaluate (default: the training number, {TRAIN_LANDMARKS})")
    argsparser.add_argument('-n', '--num_episodes', type=int, default=100)
    argsparser.add_argument('--seed', type=int, default=12345)
    argsparser.add_argument('--limit', type=int, default=None, help="evaluate at most this many runs per project")
    argsparser.add_argument('-o', '--out', type=str, default="results/results_box.csv")
    argsparser.add_argument('--base_dir', type=str, default="results", help="downloaded models go to <base_dir>/models")
    argsparser.add_argument('--device', type=str, default="cpu")
    args = argsparser.parse_args()
    print(args)

    api = wandb.Api()
    entity = args.wandb_entity or api.default_entity
    model_dir = f"{args.base_dir}/models"
    os.makedirs(model_dir, exist_ok=True)
    indentation = " " * 2

    done = read_done(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new_file = not os.path.exists(args.out)
    out = open(args.out, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=COLUMNS)
    if new_file:
        writer.writeheader()

    w = WORLDS["box"]
    for project in args.projects:
        filters = {"state": "finished", "config.env_id": "tag/box-v0"} | args.filters
        runs = list(api.runs(f"{entity}/{project}", filters=filters))[:args.limit]
        print(f"found {len(runs)} runs in {entity}/{project}!")

        rtpt = RTPT(name_initials="CD", experiment_name=f"EvalBox_{project}", max_iterations=max(len(runs), 1))
        rtpt.start()
        for run in runs:
            print(f"Processing run: {run.name}")
            n_train = run.config.get("num_distractors") or world_setting("box", "n_distractors")
            settings = list(itertools.product(args.distractors or [n_train], args.landmarks or [TRAIN_LANDMARKS]))
            if all((run.id, n, l) in done for n, l in settings):  # skip before downloading the model
                print(f"{indentation}already in {args.out}")
                rtpt.step()
                continue
            model_path = download_model(run, model_dir, indentation)
            if model_path is None:
                rtpt.step()
                continue
            agent, world, condition, n_train, fixed_objects = load_agent(model_path, args.device)
            model = "PPObj" if fixed_objects else "ORBiT"
            frame = condition.split("_")[0]  # abs, agent, goal, distractor, landmark
            seed = torch.load(model_path, map_location="cpu", weights_only=False)["args"]["seed"]

            for n, l in settings:
                setting = f"{n} distractors / {l} landmarks"
                if (run.id, n, l) in done:
                    print(f"{indentation}{setting}: already in {args.out}")
                    continue
                if fixed_objects and (n, l) != (n_train, TRAIN_LANDMARKS):
                    print(f"{indentation}{setting}: skipped (PPObj needs {n_train} distractors, "
                          f"{TRAIN_LANDMARKS} landmarks)")
                    continue
                if (frame == "distractor" and n < 1) or (frame == "landmark" and l < 1):
                    print(f"{indentation}{setting}: skipped (no {frame} to anchor the frame to)")
                    continue
                kw = dict(w["env_kwargs"], n_distractors=n, n_landmarks=l)
                with torch.no_grad():
                    catches = policies.evaluate_episodes(lambda obs, env, rng: agent.predict(obs)[0],
                                                         args.num_episodes, args.seed,
                                                         wrap=lambda e: wrap(e, condition, w["fixed_throttle"]),
                                                         env_cls=w["env_cls"], **kw)
                random, chase = (baseline(name, args.num_episodes, args.seed, env_cls=w["env_cls"], **kw)
                                 for name in ("random", "chase"))
                writer.writerows({"project": project, "run_id": run.id, "run_name": run.name, "model": model,
                                  "frame": frame, "seed": seed, "n_distractors": n, "n_landmarks": l,
                                  "episode": i, "catches": c, "random": random, "chase": chase}
                                 for i, c in enumerate(catches))
                out.flush()
                print(f"{indentation}{model} / {frame} / seed {seed} / {setting}: {catches.mean():.1f} catches, "
                      f"score {policies.score(catches.mean(), random, chase):+.2f}")
            rtpt.step()
    out.close()
