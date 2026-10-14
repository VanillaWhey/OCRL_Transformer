"""Object-centric Tag environment.

An agent tags a goal on a torus as often as it can within a fixed number of steps. The goal
moves exactly like one or more distractors, so the agent has to use the category to know whom to
chase. Every catch gives +1, after which the goal and the distractors re-spawn.

Two interfaces to the same implementation:
    TagVecEnv(num_envs, ...)   gymnasium VectorEnv; all environments are stepped together with numpy
                               array operations (fast; use this for training and evaluation)
    TagEnv(...)                gymnasium Env ("Tag-v0"); a single environment for rendering, videos and
                               step-by-step inspection (internally a TagVecEnv with one environment)

World
    Periodic square (torus) of side ``world_size`` (default 10); leaving one edge re-enters at the
    opposite one and all distances use the shortest wrapped displacement. Coordinates: x right,
    y up, angles in radians (counter-clockwise).

Objects (object id = row of the observation)
        object id                   category
        0                           Player      controlled by the agent
        1                           Goal        random unicycle (the only object of its category)
        2 .. 1+n_distractors        Distractor  same dynamics and spawning as the goal
        2+n_distractors .. n-1      Landmark    static: fixed position and orientation
                                                (default: evenly spaced on the diagonal, headings
                                                evenly spread over the circle)

Dynamics (unicycle; player, goal and distractors)
    control (a_v, a_omega) in [-1, 1]^2:
        v     <- clip(v + accel * a_v, 0, max_speed)
        omega <- max_turn * a_omega
        theta <- wrap(theta + omega)
        x, y  <- (x, y) + v * (cos theta, sin theta)   (mod world_size)
    Player: control = action. Goal and distractors: a smooth random control
    u <- rho * u + sqrt(1 - rho^2) * N(0, npc_noise^2) (rho = npc_smoothness), blended with a
    "steer away from the player at full speed" control:  (1 - npc_flee) * u + npc_flee * flee.

Reward / episode
    +1 when the player is within ``catch_radius`` (default 1) of the goal, else 0. After a catch the
    goal AND all distractors re-spawn at random positions farther than ``min_spawn_dist`` from the
    player (so a jump does not reveal the goal). Episodes are truncated after ``episode_length``
    steps; the return is the number of catches. With ``terminate_on_catch=True`` an episode
    instead ends (terminated) at the first catch, which is how PPO trains (ppo_oc.py); evaluation
    uses the full re-spawning episode.

Observation (current state only)
    float32 array (n_objects, 9), one row per object in object-id order:
        [is_player, is_goal, is_distractor, is_landmark, x, y, theta, v, omega]
    (category one-hot, then the kinematic state; landmarks: v = omega = 0). Column indices: X, Y,
    THETA, V, OMEGA. For the unordered output of an object detector, add wrappers.ShuffleRows.

TagVecEnv: SAME_STEP autoreset (an environment that is done is reset within the same step; its last
observation is in infos["final_obs"][i], infos["_final_obs"][i] is True). At the end of an episode
infos["episode"] = {"r": return, "l": length} with infos["_episode"] marking which envs ended.
The simulation state is public (arrays x, y, th, v, w of shape (num_envs, n_objects)), which is
what policies.py uses.

    envs = TagVecEnv(num_envs=256, seed=0)
    obs, _ = envs.reset()                                              # (256, 5, 9)
    obs, rewards, terminated, truncated, infos = envs.step(actions)    # actions (256, 2)

    env = TagEnv(render_mode="rgb_array")                              # or gym.make("Tag-v0")
    obs, info = env.reset(seed=0)                                      # (5, 9)
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.vector.utils import batch_space

CATEGORY_IDS = {"Player": 0, "Goal": 1, "Distractor": 2, "Landmark": 3}
N_CATEGORIES = len(CATEGORY_IDS)
FEATURES = tuple(f"is_{c.lower()}" for c in CATEGORY_IDS) + ("x", "y", "theta", "v", "omega")
X, Y, THETA, V, OMEGA = range(N_CATEGORIES, len(FEATURES))  # observation columns (the first N_CATEGORIES are the one-hot category)
OBS_FORMAT = "onehot_v2"  # bumped when the meaning of observation features changes (stored in checkpoints)
COLORS = {"Player": (92, 186, 92), "Goal": (214, 92, 92), "Distractor": (200, 72, 200), "Landmark": (128, 128, 128)}


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class TagVecEnv(gym.vector.VectorEnv):
    # metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
    periodic = True  # torus (tag_box_env.TagBoxVecEnv: walled box, periodic = False)

    def __init__(
        self,
        num_envs: int = 1,
        n_distractors: int = 1,
        n_landmarks: int = 2,
        landmark_positions: list[tuple[float, float]] | None = None,
        landmark_orientations: list[float] | None = None,
        world_size: float = 10.0,
        catch_radius: float = 1.0,
        episode_length: int = 500,
        player_max_speed: float = 0.5,
        npc_max_speed: float = 0.5,
        accel: float = 0.05,
        max_turn: float = math.pi / 8,
        npc_smoothness: float = 0.9,
        npc_noise: float = 0.5,
        npc_flee: float = 0.0,
        min_spawn_dist: float = 3.0,
        terminate_on_catch: bool = False,
        autoreset: bool = True,
        seed: int | None = None,
    ):
        """``landmark_positions``: absolute (x, y) per landmark; default evenly spaced on the diagonal.
        ``landmark_orientations``: absolute heading (rad) per landmark; default (i + 0.5) / n * 2 pi.
        ``autoreset=False`` (used by TagEnv): finished environments are not reset automatically."""
        assert 0.0 <= npc_smoothness < 1.0 and 0.0 <= npc_flee <= 1.0
        # on the torus no point is farther than world_size / sqrt(2) from the player
        assert min_spawn_dist < world_size / math.sqrt(2), "no spawn position farther than min_spawn_dist exists"
        self.num_envs = N = num_envs
        self.world_size = L = float(world_size)
        self.catch_radius, self.episode_length = catch_radius, episode_length
        self.player_max_speed, self.npc_max_speed, self.accel, self.max_turn = player_max_speed, npc_max_speed, accel, max_turn
        self.npc_smoothness, self.npc_noise, self.npc_flee = npc_smoothness, npc_noise, npc_flee
        self.min_spawn_dist, self.terminate_on_catch, self.autoreset = min_spawn_dist, terminate_on_catch, autoreset

        if landmark_positions is None:
            landmark_positions = [((i + 0.5) / n_landmarks * L,) * 2 for i in range(n_landmarks)]
        if landmark_orientations is None:
            landmark_orientations = [(i + 0.5) / n_landmarks * 2 * math.pi for i in range(n_landmarks)]
        assert len(landmark_positions) == len(landmark_orientations) == n_landmarks
        self.categories = ["Player", "Goal"] + ["Distractor"] * n_distractors + ["Landmark"] * n_landmarks
        self.n_objects = n = len(self.categories)
        self.n_npcs = 1 + n_distractors  # objects 1 .. n_npcs: goal, distractors
        self.landmark_ids = np.arange(1 + self.n_npcs, n)
        self.landmark_xy = np.array(landmark_positions, dtype=np.float64).reshape(-1, 2) % L
        self.landmark_theta = wrap_angle(np.array(landmark_orientations, dtype=np.float64))
        self.category_id = np.array([CATEGORY_IDS[c] for c in self.categories])

        low = [0.0] * N_CATEGORIES + [0.0, 0.0, -math.pi, 0.0, -max_turn]
        high = [1.0] * N_CATEGORIES + [L, L, math.pi, max(player_max_speed, npc_max_speed), max_turn]
        self.single_observation_space = spaces.Box(np.tile(np.array(low, np.float32), (n, 1)),
                                                   np.tile(np.array(high, np.float32), (n, 1)), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)  # (a_v, a_omega)
        self.observation_space = batch_space(self.single_observation_space, N)
        self.action_space = batch_space(self.single_action_space, N)

        # simulation state, (num_envs, n_objects) in object-id order
        self.x, self.y, self.th, self.v, self.w = (np.zeros((N, n)) for _ in range(5))
        self.ctrl = np.zeros((N, self.n_npcs, 2))  # random control of goal and distractors
        self.steps = np.zeros(N, dtype=np.int64)
        self.ep_return, self.ep_length = np.zeros(N), np.zeros(N, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    @property
    def feature_periods(self) -> list[float]:
        """Period of each observation feature (0 = not periodic), e.g. for sin/cos encodings."""
        return [0.0] * N_CATEGORIES + [self.world_size, self.world_size, 2 * math.pi, 0.0, 0.0]

    def delta(self, a: int, b: int):
        """Shortest displacement (dx, dy) from object a to object b on the torus, per environment."""
        L = self.world_size
        return (self.x[:, b] - self.x[:, a] + L / 2) % L - L / 2, (self.y[:, b] - self.y[:, a] + L / 2) % L - L / 2

    # ------------------------------------------------------------------ simulation
    def _spawn_npcs(self, envs):
        """Re-spawn goal and distractors of the environments ``envs`` (index array) away from the player."""
        k, L, m = len(envs), self.world_size, self.n_npcs
        if k == 0:
            return
        npcs = np.arange(1, 1 + m)
        px, py = self.x[envs, 0][:, None], self.y[envs, 0][:, None]
        xs, ys = self.rng.uniform(0, L, (k, m)), self.rng.uniform(0, L, (k, m))
        for _ in range(100_000):  # vectorised rejection sampling; the valid region can be small in tiny worlds
            dist = np.hypot((xs - px + L / 2) % L - L / 2, (ys - py + L / 2) % L - L / 2)
            bad = dist <= self.min_spawn_dist
            if not bad.any():
                break
            xs[bad], ys[bad] = self.rng.uniform(0, L, bad.sum()), self.rng.uniform(0, L, bad.sum())
        else:
            raise RuntimeError("no spawn position found; min_spawn_dist too close to world_size / sqrt(2)")
        self.x[envs[:, None], npcs], self.y[envs[:, None], npcs] = xs, ys
        self.th[envs[:, None], npcs] = self.rng.uniform(-np.pi, np.pi, (k, m))
        self.v[envs[:, None], npcs] = self.rng.uniform(0, self.npc_max_speed, (k, m))
        self.w[envs[:, None], npcs] = 0.0
        self.ctrl[envs] = self.rng.normal(0.0, self.npc_noise, (k, m, 2))

    def _reset_envs(self, envs):
        k, L = len(envs), self.world_size
        if k == 0:
            return
        lm = self.landmark_ids
        self.x[np.ix_(envs, lm)], self.y[np.ix_(envs, lm)] = self.landmark_xy[:, 0], self.landmark_xy[:, 1]
        self.th[np.ix_(envs, lm)], self.v[np.ix_(envs, lm)], self.w[np.ix_(envs, lm)] = self.landmark_theta, 0.0, 0.0
        self.x[envs, 0], self.y[envs, 0] = self.rng.uniform(0, L, k), self.rng.uniform(0, L, k)
        self.th[envs, 0], self.v[envs, 0], self.w[envs, 0] = self.rng.uniform(-np.pi, np.pi, k), 0.0, 0.0
        self._spawn_npcs(envs)
        self.steps[envs], self.ep_return[envs], self.ep_length[envs] = 0, 0.0, 0

    def _move(self, objs, a_v, a_w, max_speed):
        L = self.world_size
        a_v, a_w = np.clip(a_v, -1.0, 1.0), np.clip(a_w, -1.0, 1.0)
        self.v[:, objs] = np.clip(self.v[:, objs] + self.accel * a_v, 0.0, max_speed)
        self.w[:, objs] = self.max_turn * a_w
        self.th[:, objs] = wrap_angle(self.th[:, objs] + self.w[:, objs])
        self.x[:, objs] = (self.x[:, objs] + self.v[:, objs] * np.cos(self.th[:, objs])) % L
        self.y[:, objs] = (self.y[:, objs] + self.v[:, objs] * np.sin(self.th[:, objs])) % L

    def _obs(self, envs=None) -> np.ndarray:
        envs = slice(None) if envs is None else envs
        n_env = self.x[envs].shape[0]
        out = np.zeros((n_env, self.n_objects, len(FEATURES)), dtype=np.float32)
        out[:, np.arange(self.n_objects), self.category_id] = 1.0
        out[..., X], out[..., Y], out[..., THETA], out[..., V], out[..., OMEGA] = (
            self.x[envs], self.y[envs], self.th[envs], self.v[envs], self.w[envs])
        return out

    # ------------------------------------------------------------------ gymnasium API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_envs(np.arange(self.num_envs))
        return self._obs()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_envs, 2)
        self._move(slice(0, 1), actions[:, :1], actions[:, 1:], self.player_max_speed)
        # goal and distractors: smooth random control (+ fleeing from the player)
        rho, npcs = self.npc_smoothness, slice(1, 1 + self.n_npcs)
        self.ctrl = rho * self.ctrl + math.sqrt(1 - rho**2) * self.rng.normal(0.0, self.npc_noise, self.ctrl.shape)
        ctrl = self.ctrl
        if self.npc_flee != 0.0:
            L = self.world_size
            dx = (self.x[:, npcs] - self.x[:, :1] + L / 2) % L - L / 2
            dy = (self.y[:, npcs] - self.y[:, :1] + L / 2) % L - L / 2
            turn = np.clip(wrap_angle(np.arctan2(dy, dx) - self.th[:, npcs]) / self.max_turn, -1.0, 1.0)
            ctrl = (1 - self.npc_flee) * ctrl + self.npc_flee * np.stack([np.ones_like(turn), turn], axis=-1)
        self._move(npcs, ctrl[..., 0], ctrl[..., 1], self.npc_max_speed)
        self.steps += 1

        distance = np.hypot(*self.delta(0, 1))
        caught = distance <= self.catch_radius
        reward = caught.astype(np.float64)
        terminated = caught & self.terminate_on_catch
        truncated = (self.steps >= self.episode_length) & ~terminated
        self._spawn_npcs(np.flatnonzero(caught & ~terminated))  # re-spawn after a catch
        self.ep_return += reward
        self.ep_length += 1

        obs = self._obs()
        done = terminated | truncated
        infos = {"caught": caught, "distance": distance}
        if done.any():
            infos["episode"] = {"r": np.where(done, self.ep_return, 0.0), "l": np.where(done, self.ep_length, 0)}
            infos["_episode"] = done.copy()
            if self.autoreset:
                ended = np.flatnonzero(done)
                infos["final_obs"] = np.full(self.num_envs, None, dtype=object)
                for i in ended:
                    infos["final_obs"][i] = obs[i].copy()  # copy: obs[ended] is overwritten by the reset below
                infos["_final_obs"] = done.copy()
                self._reset_envs(ended)
                obs[ended] = self._obs(ended)
        return obs, reward, terminated, truncated, infos

    def seed(self, seed: int):
        self.np_random, _ = seeding.np_random(seed)


class GameObject:
    """View on one object of a single-environment TagVecEnv (reads and writes the shared state)."""

    def __init__(self, vec: TagVecEnv, object_id: int):
        self._vec, self.object_id = vec, object_id
        self.category = vec.categories[object_id]
        self.category_id = CATEGORY_IDS[self.category]
        self.rgb, self.size = COLORS[self.category], 1.0

    def _get(name):
        return property(lambda self: float(getattr(self._vec, name)[0, self.object_id]),
                        lambda self, value: getattr(self._vec, name).__setitem__((0, self.object_id), value))

    x, y, theta, v, omega = _get("x"), _get("y"), _get("th"), _get("v"), _get("w")
    del _get

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def __repr__(self):
        return (f"{self.category}#{self.object_id}(x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:+.2f}, "
                f"v={self.v:.2f}, omega={self.omega:+.2f})")


class TagEnv(gym.Env):
    """Single Tag environment (gymnasium Env "Tag-v0"); same arguments as TagVecEnv (without num_envs)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    periodic = True

    def __init__(self, render_mode: str | None = None, **kwargs):
        assert render_mode in (None, "rgb_array")
        self.render_mode = render_mode
        self.vec = vec = TagVecEnv(num_envs=1, autoreset=False, **kwargs)
        self.observation_space, self.action_space = vec.single_observation_space, vec.single_action_space
        self.objects = [GameObject(vec, i) for i in range(vec.n_objects)]
        self.player, self.goal = self.objects[:2]
        self.npcs = self.objects[1:1 + vec.n_npcs]
        self.distractors = self.objects[2:1 + vec.n_npcs]
        self.landmarks = self.objects[1 + vec.n_npcs:]

    def __getattr__(self, name):  # parameters (world_size, max_turn, catch_radius, ...) live in the vector env
        if name == "vec":
            raise AttributeError(name)
        return getattr(self.vec, name)

    @property
    def row_object_ids(self) -> list[int]:
        """Object id of the object in each observation row (always in order; see wrappers.ShuffleRows)."""
        return list(range(len(self.objects)))

    def delta(self, a: GameObject, b: GameObject) -> np.ndarray:
        """Shortest displacement from a to b on the torus."""
        L = self.world_size
        return (b.xy - a.xy + L / 2) % L - L / 2

    def dist(self, a: GameObject, b: GameObject) -> float:
        return float(np.linalg.norm(self.delta(a, b)))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs, _ = self.vec.reset(seed=seed)
        return obs[0]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.vec.step(np.asarray(action, dtype=np.float64)[None])
        return obs[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), {
            "distance": float(info["distance"][0]), "caught": bool(info["caught"][0])}

    def render(self, scale: float | None = None):
        """RGB frame of the torus (y up), ~640 px wide by default (``scale`` = pixels per unit);
        objects crossing an edge are drawn on both sides."""
        if self.render_mode != "rgb_array":
            return None
        L = self.world_size
        scale = scale or 640 / L
        size = int(L * scale)
        px = (np.arange(size) + 0.5) / scale
        xx, yy = np.meshgrid(px, px[::-1])  # row 0 = top = y max
        img = np.full((size, size, 3), 24, dtype=np.uint8)

        def offset(o):
            return (xx - o.x + L / 2) % L - L / 2, (yy - o.y + L / 2) % L - L / 2

        dx, dy = offset(self.goal)
        img[np.abs(np.hypot(dx, dy) - self.catch_radius) < 1 / scale] = (120, 60, 60)  # catch ring
        for o in self.objects[::-1]:  # player on top
            dx, dy = offset(o)
            r = o.size / 2
            along = dx * math.cos(o.theta) + dy * math.sin(o.theta)
            across = -dx * math.sin(o.theta) + dy * math.cos(o.theta)
            if o.category == "Landmark":  # square aligned with its orientation
                img[(np.abs(along) <= r) & (np.abs(across) <= r)] = o.rgb
            else:
                img[np.hypot(dx, dy) <= r] = o.rgb
            # heading marker: a short line from the centre in direction theta
            img[(along > 0) & (along <= r * 1.6) & (np.abs(across) <= 1.2 / scale)] = (255, 255, 255)
        return img

    def seed(self, seed: int):
        self.np_random, _ = seeding.np_random(seed)


gym.register(id="Tag-v0", entry_point=TagEnv)


if __name__ == "__main__":
    env = TagEnv()
    obs, info = env.reset(seed=0)
    print(*env.objects, f"obs shape: {obs.shape}", sep="\n")
    envs = TagVecEnv(num_envs=100, seed=0)
    envs.reset()
    rng = np.random.default_rng(0)
    for _ in range(envs.episode_length):
        _, _, _, _, infos = envs.step(rng.uniform(-1, 1, (100, 2)))
    print(f"random policy: {infos['episode']['r'].mean():.2f} catches per episode (100 episodes)")
