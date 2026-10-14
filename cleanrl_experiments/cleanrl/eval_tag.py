"""Evaluate a trained ppo_atari_orbit.py or ppo_atari_obj.py model on its tag environment.

Runs the deterministic policy (action mean) on 100 full 500-step re-spawning episodes and reports catches per
episode next to the random and chase (expert) policies, and score = (agent - random) / (chase - random).

    python -m cleanrl_experiments.cleanrl.eval_tag <path/to/model.cleanrl_model> [n_distractors]
"""
import sys
from types import SimpleNamespace

import torch

import cleanrl_experiments.cleanrl.ppo_atari_obj as obj
import cleanrl_experiments.cleanrl.ppo_atari_orbit as orbit
from cleanrl_experiments.tag.tag_env import N_CATEGORIES
from cleanrl_experiments.tag.tag_setup import evaluate, make_vec_env, world_setting
from cleanrl_experiments.tag.wrappers import TagVecEnvAdapter


def load_agent(model_path, device="cpu"):
    """Returns (agent, world, condition, training n_distractors, fixed_objects). ``fixed_objects``: the agent only
    accepts the number of objects it was trained with (PPObj flattens them); ORBiT takes any number."""
    data = torch.load(model_path, map_location=device, weights_only=False)
    args = SimpleNamespace(**data["args"])
    world = args.env_id.split("/")[1].split("-v")[0]
    condition = f"{args.player_name}_kinematic" if args.player_name else "abs"
    n_distractors = getattr(args, "num_distractors", None)
    if n_distractors is None:  # the world's default
        n_distractors = world_setting(world, "n_distractors")
    envs = TagVecEnvAdapter(make_vec_env(world, condition, 1, n_distractors=n_distractors))  # only for the spaces
    if hasattr(args, "emb_dim"):  # ORBiT
        orbit.args = args  # PPOAgent reads dropout, pooling_type and num_post_layers from the module-level args
        agent = orbit.PPOAgent(envs, args.emb_dim, args.num_heads, args.num_blocks, N_CATEGORIES, torch.device(device))
        fixed_objects = False
    else:  # PPObj
        agent = obj.PPOAgent(envs, args.encoder_dims, args.decoder_dims, True, torch.device(device))
        fixed_objects = True
    agent.load_state_dict(data["model_weights"])
    agent.eval()
    return agent, world, condition, n_distractors, fixed_objects


if __name__ == "__main__":
    agent, world, condition, n_distractors, _ = load_agent(sys.argv[1])
    n_distractors = int(sys.argv[2]) if len(sys.argv) > 2 else n_distractors
    res = evaluate(lambda obs: agent.predict(obs)[0], world, condition, n_distractors=n_distractors)
    print(f"{world} / {condition} / {n_distractors} distractors: agent {res['agent']:.1f}, random {res['random']:.1f}, "
          f"chase {res['chase']:.1f} catches per episode -> score {res['score']:+.2f}")
