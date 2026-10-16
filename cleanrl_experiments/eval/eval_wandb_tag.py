"""Evaluate the tag runs of W&B projects and write the results into each run's summary.

For every finished run on a tag env, the latest model artifact is downloaded and evaluated (deterministic policy,
full 500-step re-spawning episodes) with several numbers of distractors, next to the random and chase policies:

    <prefix>/<n>_distractors/{agent,score}    for every evaluated number of distractors n
    <prefix>/{agent,score}                    the training setting

With --eval-worlds the model is also evaluated in other worlds (transfer), each with one number of distractors:

    <prefix>/<world>/<n>_distractors/{agent,score}

There, n and the goals' fleeing are the world's own settings, except in box_torus_like, which takes both from the
training world (so only the walls differ from training on the torus). Worlds with a different action space
(torus: steering only) are skipped.

agent is catches per episode, score = (agent - random) / (chase - random); random and chase do not depend on the
model, so they are only printed (they are stored in tag/baselines.json, not in the runs). PPObj flattens a fixed
number of objects, so it is only evaluated with its training number of distractors.

    python -m cleanrl_experiments.eval.eval_wandb_tag -p OCRL_Transformer PPObj-v2 -d 1 3 5 8 10 --eval-worlds box torus_2d box_torus_like --test
"""
import argparse
import os

import torch
import wandb
import yaml
from rtpt import RTPT

from cleanrl_experiments.eval.eval_tag import load_agent
from cleanrl_experiments.tag.tag_setup import WORLDS, evaluate, world_setting

# Results written to the W&B runs (random and chase are the same for every model)
UPLOADED = ("agent", "score")


def download_model(run, model_dir, indentation="  "):
    """Path of the run's latest model artifact, downloaded to <model_dir>/<run id>.cleanrl_model if not there yet;
    None if the run has no model artifact."""
    new_path = os.path.join(model_dir, f"{run.id}.cleanrl_model")
    if os.path.exists(new_path):
        return new_path
    # Latest model artifact of the run (the final model is logged last)
    artifacts = sorted((a for a in run.logged_artifacts() if a.type == "model"),
                       key=lambda a: a.updated_at, reverse=True)
    if not artifacts:
        print(f"{indentation}No model artifact found for run {run.name}")
        return None
    artifact = artifacts[0]
    print(f"{indentation}Downloading artifact: {artifact.name}")

    # Download artifact to the output directory and rename the model file
    artifact_dir = artifact.download(root=os.path.join(model_dir, run.id))
    os.replace(os.path.join(artifact_dir, list(artifact.manifest.entries.keys())[0]), new_path)
    print(f"{indentation}Saved to: {new_path}")
    return new_path


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = yaml.safe_load(value)


if __name__ == '__main__':
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument('-e', '--wandb_entity', type=str, default=None, help="default: your W&B default entity")
    argsparser.add_argument('-p', '--projects', action="extend", nargs='*', type=str)
    argsparser.add_argument('-f', '--filters', nargs='*', action=ParseKwargs, default={})
    argsparser.add_argument('-d', '--distractors', nargs='*', type=int, default=[1, 3, 5, 8, 10],
                            help="numbers of distractors to evaluate (the training number is always added)")
    argsparser.add_argument('-w', '--eval-worlds', nargs='*', default=[], choices=list(WORLDS), dest="eval_worlds",
                            help="other worlds to evaluate every model in (the training world is skipped)")
    argsparser.add_argument('-n', '--num_episodes', type=int, default=100)
    argsparser.add_argument('--seed', type=int, default=12345)
    argsparser.add_argument('--limit', type=int, default=None, help="evaluate at most this many runs per project")
    argsparser.add_argument('--test', action='store_true', help="evaluate, but do not upload to W&B")
    argsparser.add_argument('--no-re-eval', action='store_false', dest="re_eval",
                            help="skip runs whose model was already downloaded")
    argsparser.add_argument('--prefix', type=str, default="eval")
    argsparser.add_argument('--base_dir', type=str, default="results", help="downloaded models go to <base_dir>/models")
    argsparser.add_argument('--device', type=str, default="cpu")
    args = argsparser.parse_args()
    print(args)

    # Initialize the wandb API
    api = wandb.Api()
    entity = args.wandb_entity or api.default_entity

    for project in args.projects:

        filters = {
            "state": "finished",
            "config.env_id": {"$in": [f"tag/{w}-v0" for w in WORLDS]},  # W&B does not support $regex here
        } | args.filters

        # Get all important runs in the project
        runs = list(api.runs(f"{entity}/{project}", filters=filters))[:args.limit]

        print(f"found {len(runs)} runs in {entity}/{project}!")

        rtpt = RTPT(name_initials="CD", experiment_name=f"Eval_{project}",
                    max_iterations=max(len(runs), 1))

        # Directory to save the models
        model_dir = f"{args.base_dir}/models"
        os.makedirs(model_dir, exist_ok=True)

        indentation = " " * 2

        # Start the RTPT tracking
        rtpt.start()
        # Iterate through runs to download models
        for run in runs:
            print(f"Processing run: {run.name}")

            # skip if already done
            if os.path.exists(os.path.join(model_dir, f"{run.id}.cleanrl_model")) and not args.re_eval:
                print(f"{indentation}Model of {run.id} already downloaded. Skipping it!")
                rtpt.step()
                continue
            new_path = download_model(run, model_dir, indentation)
            if new_path is None:
                rtpt.step()
                continue

            agent, world, condition, n_train, fixed_objects = load_agent(new_path, args.device)

            def run_eval(eval_world, n, note="", **env_kwargs):
                """Evaluates the agent in eval_world with n distractors; None if the agent cannot be run there."""
                if fixed_objects and n != n_train:
                    print(f"{indentation}{eval_world} / {n} distractors: skipped (PPObj needs {n_train} distractors)")
                    return None
                if n < 1 and condition.startswith("distractor"):  # the anchor must exist
                    return None
                with torch.no_grad():
                    res = evaluate(lambda obs: agent.predict(obs)[0], eval_world, condition, episodes=args.num_episodes,
                                   seed=args.seed, n_distractors=n, **env_kwargs)
                print(f"{indentation}{eval_world} / {condition} / {n} distractors{note}: "
                      f"agent {res['agent']:.1f}, random {res['random']:.1f}, chase {res['chase']:.1f} "
                      f"-> score {res['score']:+.2f}")
                return res

            summary = {}
            # training world, several numbers of distractors
            for n in sorted(set(args.distractors) | {n_train}):
                res = run_eval(world, n, " (training)" if n == n_train else "")
                for key, value in (res or {}).items():
                    if key not in UPLOADED:
                        continue
                    summary[f"{args.prefix}/{n}_distractors/{key}"] = value
                    if n == n_train:
                        summary[f"{args.prefix}/{key}"] = value

            # other worlds (transfer)
            for eval_world in args.eval_worlds:
                if eval_world == world:
                    continue
                if (WORLDS[eval_world]["fixed_throttle"] is None) != (WORLDS[world]["fixed_throttle"] is None):
                    print(f"{indentation}{eval_world}: skipped (different action space)")
                    continue
                if eval_world == "box_torus_like":  # walls, but distractors and fleeing as in training
                    n, env_kwargs = n_train, {"npc_flee": world_setting(world, "npc_flee")}
                else:
                    n, env_kwargs = world_setting(eval_world, "n_distractors"), {}
                res = run_eval(eval_world, n, **env_kwargs)
                for key, value in (res or {}).items():
                    if key in UPLOADED:
                        summary[f"{args.prefix}/{eval_world}/{n}_distractors/{key}"] = value

            if not args.test:
                for key, value in summary.items():
                    run.summary[key] = value
                run.summary.update()
                print(f"{indentation}Uploaded evaluation to W&B!")
            else:
                print(f"{indentation}Only testing, not uploading!")

            rtpt.step()
