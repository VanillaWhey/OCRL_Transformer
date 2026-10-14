"""Evaluate a trained ppo_atari_orbit.py model on its tag environment.

Runs the deterministic policy (action mean) on 100 full 500-step re-spawning episodes and reports catches per
episode next to the random and chase (expert) policies, and score = (agent - random) / (chase - random).

    python -m cleanrl_experiments.cleanrl.eval_tag <path/to/model.cleanrl_model>
"""
import sys
from types import SimpleNamespace

import torch

import cleanrl_experiments.cleanrl.ppo_atari_orbit as orbit
from cleanrl_experiments.tag.tag_env import N_CATEGORIES
from cleanrl_experiments.tag.tag_setup import evaluate, make_vec_env
from cleanrl_experiments.tag.wrappers import TagVecEnvAdapter


def load_agent(model_path, device="cpu"):
    data = torch.load(model_path, map_location=device, weights_only=False)
    args = SimpleNamespace(**data["args"])
    orbit.args = args  # PPOAgent reads dropout, pooling_type and num_post_layers from the module-level args
    world = args.env_id.split("/")[1].split("-v")[0]
    condition = f"{args.player_name}_kinematic" if args.player_name else "abs"
    env_kwargs = {} if args.num_distractors is None else {"n_distractors": args.num_distractors}
    envs = TagVecEnvAdapter(make_vec_env(world, condition, 1, **env_kwargs))  # only for the spaces
    agent = orbit.PPOAgent(envs, args.emb_dim, args.num_heads, args.num_blocks, N_CATEGORIES, torch.device(device))
    agent.load_state_dict(data["model_weights"])
    agent.eval()
    return agent, world, condition, env_kwargs


if __name__ == "__main__":
    agent, world, condition, env_kwargs = load_agent(sys.argv[1])
    res = evaluate(lambda obs: agent.predict(obs)[0], world, condition, **env_kwargs)
    print(f"{world} / {condition}: agent {res['agent']:.1f}, random {res['random']:.1f}, "
          f"chase {res['chase']:.1f} catches per episode -> score {res['score']:+.2f}")
