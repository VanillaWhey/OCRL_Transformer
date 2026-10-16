"""The experimental setup of the reference-frame study, for plugging in your own RL algorithm.

    from tag_setup import CONDITIONS, make_vec_env, encode_obs, evaluate

    envs = make_vec_env("box", "agent_rotated", num_envs=256, seed=1)   # training env (episodes end at a catch)
    obs, _ = envs.reset(seed=1)                                         # (256, n_objects, n_features)
    x = encode_obs(obs, envs)                                           # (256, input_dim), what our PPO network sees
    ...
    res = evaluate(lambda obs: my_policy(obs), "box", "agent_rotated")  # deterministic policy, 100 full episodes
    print(res["score"])                                                 # 0 = random, 1 = chase (expert)

Run this file for a quick demo:  python tag_setup.py
"""

import inspect
import json
import os

import numpy as np

from cleanrl_experiments.tag import policies
from cleanrl_experiments.tag.tag_box_env import TagBoxVecEnv
from cleanrl_experiments.tag.tag_env import TagVecEnv
from cleanrl_experiments.tag.wrappers import VecAnchorFrame, VecFixedThrottle, VecKinematicFrame

# The 13 frames of the ablation: name -> (frame, anchor). Anchors are fixed objects: "distractor" is
# always the first distractor (object id 2), "landmark" the first landmark.
CONDITIONS = {
    "abs": ("absolute", None),
    "agent_rotated": ("rotated", "agent"),
    "agent_translated": ("translated", "agent"),
    "agent_kinematic": ("kinematic", "agent"),
    "goal_rotated": ("rotated", "goal"),
    "goal_translated": ("translated", "goal"),
    "goal_kinematic": ("kinematic", "goal"),
    "distractor_rotated": ("rotated", "distractor"),
    "distractor_translated": ("translated", "distractor"),
    "distractor_kinematic": ("kinematic", "distractor"),
    "landmark_rotated": ("rotated", "landmark"),
    "landmark_translated": ("translated", "landmark"),
    "landmark_kinematic": ("kinematic", "landmark"),
}

# Worlds of the report. "box" (= Box-main, the main results): walled box, 5 distractors, goals flee
# with weight 0.5. "torus": wrap-around world, 1 distractor, no fleeing; our torus runs only steered
# (throttle fixed at full). The env defaults are exactly these settings. "torus_2d": the same torus
# with the full 2-d action (throttle, turn), as in the box. "box_torus_like": walls as in the box, but the
# torus' distractors and goal behaviour (1 distractor, no fleeing) - only the topology differs from torus_2d.
# (eval_wandb_tag.py evaluates every model in it with the number of distractors and fleeing of its training world.)
WORLDS = {
    "box": dict(env_cls=TagBoxVecEnv, env_kwargs={}, fixed_throttle=None),
    "box_heldout": dict(env_cls=TagBoxVecEnv, env_kwargs={"n_distractors": 3}, fixed_throttle=None),
    "torus": dict(env_cls=TagVecEnv, env_kwargs={}, fixed_throttle=1.0),
    "torus_2d": dict(env_cls=TagVecEnv, env_kwargs={}, fixed_throttle=None),
    "box_torus_like": dict(env_cls=TagBoxVecEnv, env_kwargs={"n_distractors": 1, "npc_flee": 0.0}, fixed_throttle=None),
}


def world_setting(world, key):
    """Effective value of the env parameter ``key`` (e.g. n_distractors, npc_flee) in ``world``."""
    w = WORLDS[world]
    return w["env_kwargs"].get(key, inspect.signature(w["env_cls"]).parameters[key].default)


# Catches per episode of the reference policies, which do not depend on the agent: computed once per setting
# and stored in this file (keyed by policy, env class, every env parameter, episodes and seed).
BASELINES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines.json")
_NOT_ENV_PARAMS = ("num_envs", "seed", "terminate_on_catch", "autoreset")


def _load_baselines():
    try:
        with open(BASELINES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def baseline(name, episodes=100, seed=12345, env_cls=TagVecEnv, **env_kwargs):
    """Mean catches per full episode of the reference policy ``name`` ("random" or "chase"), read from
    baselines.json, or evaluated and added to it if this setting is not in it yet."""
    params = {k: p.default for k, p in inspect.signature(env_cls).parameters.items() if k not in _NOT_ENV_PARAMS}
    params.update(env_kwargs)
    key = json.dumps({"policy": name, "env": env_cls.__name__, "episodes": episodes, "seed": seed, **params},
                     sort_keys=True)
    cached = _load_baselines()
    if key in cached:
        return cached[key]
    value = policies.evaluate(policies.POLICIES[name], episodes, seed, env_cls=env_cls, **env_kwargs)
    cached = _load_baselines()  # re-read: another process may have added settings meanwhile
    cached[key] = value
    tmp = f"{BASELINES_PATH}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cached, f, indent=1, sort_keys=True)
    os.replace(tmp, BASELINES_PATH)  # atomic, so concurrent readers never see a half-written file
    return value


def anchor_id(anchor, n_distractors):
    """Object id of the anchor: 0 agent, 1 goal, 2 first distractor, 2 + n_distractors first landmark."""
    return {"agent": 0, "goal": 1, "distractor": 2, "landmark": 2 + n_distractors}[anchor]


def wrap(env, condition, fixed_throttle=None):
    """Frame wrapper (and, for the torus, fixed throttle) around a TagVecEnv / TagBoxVecEnv."""
    frame, anchor = CONDITIONS[condition]
    if frame == "rotated":
        env = VecAnchorFrame(env, anchor_id(anchor, env.n_npcs - 1), rotate=True)
    elif frame == "translated":
        env = VecAnchorFrame(env, anchor_id(anchor, env.n_npcs - 1), rotate=False)
    elif frame == "kinematic":
        env = VecKinematicFrame(env, anchor_id(anchor, env.n_npcs - 1))
    if fixed_throttle is not None:
        env = VecFixedThrottle(env, fixed_throttle)
    return env


def make_vec_env(world, condition, num_envs=256, seed=None, terminate_on_catch=True, **env_kwargs):
    """Training env as in our PPO runs: the episode ends at the first catch (reward +1), otherwise it is
    truncated after 500 steps. Extra env_kwargs override the world's settings (e.g. npc_flee=0.25)."""
    w = WORLDS[world]
    env = w["env_cls"](num_envs=num_envs, seed=seed, terminate_on_catch=terminate_on_catch,
                       **{**w["env_kwargs"], **env_kwargs})
    return wrap(env, condition, w["fixed_throttle"])


def encode_obs(obs, env):
    """The network input of our PPO (optional; use your own if you prefer): periodic features (positions
    on the torus, headings) -> (sin, cos), all others scaled by their observation-space bound to [-1, 1];
    then flattened. obs (num_envs, n_objects, n_features) -> (num_envs, input_dim)."""
    space = env.single_observation_space
    periods = np.asarray(env.feature_periods, dtype=np.float64)
    scale = np.maximum(np.abs(space.low), np.abs(space.high))
    scale = np.where(scale > 0, scale, 1.0)
    linear = (obs / scale)[..., periods <= 0]
    angle = 2 * np.pi * obs[..., periods > 0] / periods[periods > 0]
    x = np.concatenate([linear, np.sin(angle), np.cos(angle)], axis=-1)
    return x.reshape(len(obs), -1).astype(np.float32)


def evaluate(act, world, condition, episodes=100, seed=12345, **env_kwargs):
    """Catches per full 500-step re-spawning episode of ``act(obs) -> actions`` (deterministic policy;
    obs as from make_vec_env, actions (episodes, action_dim)), of random and of chase on the same
    episodes, and score = (agent - random) / (chase - random). random and chase come from baselines.json."""
    w = WORLDS[world]
    kw = dict(w["env_kwargs"], **env_kwargs)
    res = {"agent": policies.evaluate(lambda obs, env, rng: act(obs), episodes, seed,
                                      wrap=lambda e: wrap(e, condition, w["fixed_throttle"]), env_cls=w["env_cls"], **kw)}
    for name in ("random", "chase"):
        res[name] = baseline(name, episodes, seed, env_cls=w["env_cls"], **kw)
    res["score"] = policies.score(res["agent"], res["random"], res["chase"])
    return res


if __name__ == "__main__":
    envs = make_vec_env("box", "agent_rotated", num_envs=8, seed=0)
    obs, _ = envs.reset(seed=0)
    print("observation", obs.shape, "-> network input", encode_obs(obs, envs).shape, "| action", envs.single_action_space)
    rng = np.random.default_rng(0)
    for _ in range(600):
        # ended environments are already reset; their last observation is infos["final_obs"][i] where infos["_final_obs"][i]
        obs, reward, terminated, truncated, infos = envs.step(rng.uniform(-1, 1, (8, 2)))
    res = evaluate(lambda obs: rng.uniform(-1, 1, (len(obs), 2)), "box", "agent_rotated", episodes=20)
    print("random policy as 'agent':", {k: round(v, 2) for k, v in res.items()})
