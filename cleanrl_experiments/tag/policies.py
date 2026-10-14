"""Reference policies for the Tag environment and a fast evaluation.

A policy is a function  policy(obs, env, rng) -> actions  for a batch of environments:
    obs   observations of the (possibly wrapped) vector env, (num_envs, n_objects, features)
    env   the underlying tag_env.TagVecEnv (its state: x, y, th, v, w of shape (num_envs, n_objects);
          object 0 = player, 1 = goal)
    rng   numpy random generator
    returns actions (num_envs, 2) = (throttle a_v, turn a_omega), each in [-1, 1]
The reference policies below read the true state and ignore ``obs``.

    import policies
    random_catches = policies.evaluate(policies.random)            # ~4 catches per episode (10x10)
    chase_catches = policies.evaluate(policies.chase)              # ~34
    policies.score(x, random_catches, chase_catches)               # 0 = random, 1 = chase, > 1 = better

Score: (catches - random) / (chase - random), catches per full 500-step episode (goal re-spawns
after every catch). "chase" (steer straight at the goal at full speed) is the oracle used as 1.0;
it is not optimal (e.g. it does not lead the goal), so scores above 1 are possible.
"""

import numpy as np

from cleanrl_experiments.tag.tag_env import TagVecEnv, wrap_angle


def chase(obs, env, rng):
    """Oracle: full throttle, turn towards the goal (as hard as needed, at most max_turn per step)."""
    dx, dy = env.delta(0, 1)
    err = wrap_angle(np.arctan2(dy, dx) - env.th[:, 0])
    return np.stack([np.ones(env.num_envs), np.clip(err / env.max_turn, -1.0, 1.0)], axis=1)


def random(obs, env, rng):
    """Uniformly random throttle and turn."""
    return rng.uniform(-1.0, 1.0, (env.num_envs, 2))


def straight(obs, env, rng):
    """Full speed, never turn."""
    return np.tile([1.0, 0.0], (env.num_envs, 1))


def circle(obs, env, rng):
    """Full speed, always turn left as hard as possible (tight circle)."""
    return np.tile([1.0, 1.0], (env.num_envs, 1))


def idle(obs, env, rng):
    """Brake and stand still."""
    return np.tile([-1.0, 0.0], (env.num_envs, 1))


POLICIES = {"chase": chase, "random": random, "straight": straight, "circle": circle, "idle": idle}


def evaluate(policy, episodes: int = 100, seed: int = 12345, wrap=None, env_cls=TagVecEnv, **env_kwargs) -> float:
    """Mean catches per full episode (re-spawning, ``episode_length`` steps) over ``episodes`` episodes,
    all run in parallel. ``wrap(env)`` adds wrappers (e.g. frames for a learned policy); env_kwargs go
    to TagVecEnv (world_size, n_distractors, ...); env_cls=tag_box_env.TagBoxVecEnv for the walled box.
    Same seed -> same initial states for every policy."""
    base = env_cls(num_envs=episodes, seed=seed, terminate_on_catch=False, **env_kwargs)
    env = wrap(base) if wrap else base
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    catches = np.zeros(episodes)
    for _ in range(base.episode_length):
        obs, reward, _, _, _ = env.step(policy(obs, base, rng))
        catches += reward
    return float(catches.mean())


def score(catches: float, random_catches: float, chase_catches: float) -> float:
    return (catches - random_catches) / (chase_catches - random_catches)


if __name__ == "__main__":
    for size, msd in ((5.0, 2.0), (10.0, 3.0)):
        res = {name: evaluate(p, world_size=size, min_spawn_dist=msd) for name, p in POLICIES.items()}
        print(f"{size:g}x{size:g}: " + "  ".join(f"{n} {c:.1f} (score {score(c, res['random'], res['chase']):+.2f})"
                                               for n, c in res.items()))
