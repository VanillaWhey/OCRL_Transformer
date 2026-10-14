"""Tag in a walled box instead of on a torus (separate implementation of tag_env.py; same API).

Defaults: 5 distractors and goals that flee with npc_flee = 0.5. Chosen by a screening of harder
variants (report, "Choosing a harder box"): with the torus defaults (1 distractor, no fleeing) PPO
reaches the expert's level in every reference frame, so the box could not tell frames apart; with 5
look-alikes the goal is hard to identify, and a fleeing goal can be trapped against the walls, which
lets good policies beat the chase expert.

Differences to tag_env.py:
    * the world is the square [0, L]^2 with walls; distances are plain Euclidean (no wrap-around)
    * player: stops at the walls - its position is clamped to the box, so it slides along a wall;
      its heading is only ever changed by its own actions
    * goal and distractors: bounce off the walls - position mirrored back into the box and heading
      mirrored (theta -> pi - theta at the left/right walls, theta -> -theta at the bottom/top walls),
      speed kept
    * positions are not periodic (feature_periods 0 for x and y); ``periodic = False`` tells the frame
      wrappers not to wrap displacements
Everything else (objects, dynamics, reward, spawning, observation layout, autoreset, infos) is as in
tag_env.TagVecEnv / TagEnv.

    envs = TagBoxVecEnv(num_envs=256, seed=0)
    env = TagBoxEnv(render_mode="rgb_array")
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.vector.utils import batch_space

from atari.tag.tag_env import CATEGORY_IDS, COLORS, FEATURES, N_CATEGORIES, OMEGA, THETA, V, X, Y, GameObject, wrap_angle


class TagBoxVecEnv(gym.vector.VectorEnv):
    # metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
    periodic = False

    def __init__(
        self,
        num_envs: int = 1,
        n_distractors: int = 5,
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
        npc_flee: float = 0.5,
        min_spawn_dist: float = 3.0,
        terminate_on_catch: bool = False,
        autoreset: bool = True,
        seed: int | None = None,
    ):
        assert 0.0 <= npc_smoothness < 1.0 and 0.0 <= npc_flee <= 1.0
        # from any position the farthest corner is >= L/sqrt(2) away; L/2 leaves enough free area for the re-spawn
        assert min_spawn_dist < world_size / 2, "spawn positions farther than min_spawn_dist may not exist"
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
        self.n_npcs = 1 + n_distractors
        self.landmark_ids = np.arange(1 + self.n_npcs, n)
        self.landmark_xy = np.clip(np.array(landmark_positions, dtype=np.float64).reshape(-1, 2), 0.0, L)
        self.landmark_theta = wrap_angle(np.array(landmark_orientations, dtype=np.float64))
        self.category_id = np.array([CATEGORY_IDS[c] for c in self.categories])

        low = [0.0] * N_CATEGORIES + [0.0, 0.0, -math.pi, 0.0, -max_turn]
        high = [1.0] * N_CATEGORIES + [L, L, math.pi, max(player_max_speed, npc_max_speed), max_turn]
        self.single_observation_space = spaces.Box(np.tile(np.array(low, np.float32), (n, 1)),
                                                   np.tile(np.array(high, np.float32), (n, 1)), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, N)
        self.action_space = batch_space(self.single_action_space, N)

        self.x, self.y, self.th, self.v, self.w = (np.zeros((N, n)) for _ in range(5))
        self.ctrl = np.zeros((N, self.n_npcs, 2))
        self.steps = np.zeros(N, dtype=np.int64)
        self.ep_return, self.ep_length = np.zeros(N), np.zeros(N, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    @property
    def feature_periods(self) -> list[float]:
        """Only headings are periodic in the box."""
        return [0.0] * N_CATEGORIES + [0.0, 0.0, 2 * math.pi, 0.0, 0.0]

    def delta(self, a: int, b: int):
        """Displacement (dx, dy) from object a to object b, per environment (no wrap-around)."""
        return self.x[:, b] - self.x[:, a], self.y[:, b] - self.y[:, a]

    # ------------------------------------------------------------------ simulation
    def _spawn_npcs(self, envs):
        k, L, m = len(envs), self.world_size, self.n_npcs
        if k == 0:
            return
        npcs = np.arange(1, 1 + m)
        px, py = self.x[envs, 0][:, None], self.y[envs, 0][:, None]
        xs, ys = self.rng.uniform(0, L, (k, m)), self.rng.uniform(0, L, (k, m))
        for _ in range(100_000):
            bad = np.hypot(xs - px, ys - py) <= self.min_spawn_dist
            if not bad.any():
                break
            xs[bad], ys[bad] = self.rng.uniform(0, L, bad.sum()), self.rng.uniform(0, L, bad.sum())
        else:
            raise RuntimeError("no spawn position found; min_spawn_dist too large for the box")
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

    def _move_player(self, a_v, a_w):
        """Unicycle step; the player stops at the walls (position clamped, heading untouched). Its speed v
        keeps the commanded value (like a wheel-speed reading), also while it is pressed against a wall."""
        L = self.world_size
        a_v, a_w = np.clip(a_v, -1.0, 1.0), np.clip(a_w, -1.0, 1.0)
        self.v[:, 0] = np.clip(self.v[:, 0] + self.accel * a_v, 0.0, self.player_max_speed)
        self.w[:, 0] = self.max_turn * a_w
        self.th[:, 0] = wrap_angle(self.th[:, 0] + self.w[:, 0])
        self.x[:, 0] = np.clip(self.x[:, 0] + self.v[:, 0] * np.cos(self.th[:, 0]), 0.0, L)
        self.y[:, 0] = np.clip(self.y[:, 0] + self.v[:, 0] * np.sin(self.th[:, 0]), 0.0, L)

    def _move_npcs(self, a_v, a_w):
        """Unicycle step; goal and distractors bounce off the walls (position and heading mirrored)."""
        L, s = self.world_size, slice(1, 1 + self.n_npcs)
        a_v, a_w = np.clip(a_v, -1.0, 1.0), np.clip(a_w, -1.0, 1.0)
        self.v[:, s] = np.clip(self.v[:, s] + self.accel * a_v, 0.0, self.npc_max_speed)
        self.w[:, s] = self.max_turn * a_w
        th = wrap_angle(self.th[:, s] + self.w[:, s])
        x = self.x[:, s] + self.v[:, s] * np.cos(th)
        y = self.y[:, s] + self.v[:, s] * np.sin(th)
        hit_x = (x < 0.0) | (x > L)
        x = np.where(x < 0.0, -x, np.where(x > L, 2 * L - x, x))
        th = np.where(hit_x, np.pi - th, th)
        hit_y = (y < 0.0) | (y > L)
        y = np.where(y < 0.0, -y, np.where(y > L, 2 * L - y, y))
        th = np.where(hit_y, -th, th)
        self.x[:, s], self.y[:, s], self.th[:, s] = np.clip(x, 0.0, L), np.clip(y, 0.0, L), wrap_angle(th)

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
        self._move_player(actions[:, 0], actions[:, 1])
        rho, npcs = self.npc_smoothness, slice(1, 1 + self.n_npcs)
        self.ctrl = rho * self.ctrl + math.sqrt(1 - rho**2) * self.rng.normal(0.0, self.npc_noise, self.ctrl.shape)
        ctrl = self.ctrl
        if self.npc_flee != 0.0:
            dx, dy = self.x[:, npcs] - self.x[:, :1], self.y[:, npcs] - self.y[:, :1]
            turn = np.clip(wrap_angle(np.arctan2(dy, dx) - self.th[:, npcs]) / self.max_turn, -1.0, 1.0)
            ctrl = (1 - self.npc_flee) * ctrl + self.npc_flee * np.stack([np.ones_like(turn), turn], axis=-1)
        self._move_npcs(ctrl[..., 0], ctrl[..., 1])
        self.steps += 1

        distance = np.hypot(*self.delta(0, 1))
        caught = distance <= self.catch_radius
        reward = caught.astype(np.float64)
        terminated = caught & self.terminate_on_catch
        truncated = (self.steps >= self.episode_length) & ~terminated
        self._spawn_npcs(np.flatnonzero(caught & ~terminated))
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


class TagBoxEnv(gym.Env):
    """Single Tag-in-a-box environment ("TagBox-v0"); same arguments as TagBoxVecEnv (without num_envs)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    periodic = False

    def __init__(self, render_mode: str | None = None, **kwargs):
        assert render_mode in (None, "rgb_array")
        self.render_mode = render_mode
        self.vec = vec = TagBoxVecEnv(num_envs=1, autoreset=False, **kwargs)
        self.observation_space, self.action_space = vec.single_observation_space, vec.single_action_space
        self.objects = [GameObject(vec, i) for i in range(vec.n_objects)]
        self.player, self.goal = self.objects[:2]
        self.npcs = self.objects[1:1 + vec.n_npcs]
        self.distractors = self.objects[2:1 + vec.n_npcs]
        self.landmarks = self.objects[1 + vec.n_npcs:]

    def __getattr__(self, name):
        if name == "vec":
            raise AttributeError(name)
        return getattr(self.vec, name)

    @property
    def row_object_ids(self) -> list[int]:
        return list(range(len(self.objects)))

    def delta(self, a: GameObject, b: GameObject) -> np.ndarray:
        return b.xy - a.xy

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
        """RGB frame of the box (y up), ~640 px wide by default, with the walls drawn as a border."""
        if self.render_mode != "rgb_array":
            return None
        L = self.world_size
        scale = scale or 640 / L
        size = int(L * scale)
        px = (np.arange(size) + 0.5) / scale
        xx, yy = np.meshgrid(px, px[::-1])
        img = np.full((size, size, 3), 24, dtype=np.uint8)
        img[[0, 1, -2, -1], :] = img[:, [0, 1, -2, -1]] = (200, 200, 200)  # walls
        g = self.goal
        img[np.abs(np.hypot(xx - g.x, yy - g.y) - self.catch_radius) < 1 / scale] = (120, 60, 60)  # catch ring
        for o in self.objects[::-1]:
            dx, dy = xx - o.x, yy - o.y
            r = o.size / 2
            along = dx * math.cos(o.theta) + dy * math.sin(o.theta)
            across = -dx * math.sin(o.theta) + dy * math.cos(o.theta)
            if o.category == "Landmark":
                img[(np.abs(along) <= r) & (np.abs(across) <= r)] = o.rgb
            else:
                img[np.hypot(dx, dy) <= r] = o.rgb
            img[(along > 0) & (along <= r * 1.6) & (np.abs(across) <= 1.2 / scale)] = (255, 255, 255)
        return img

    def seed(self, seed: int):
        self.np_random, _ = seeding.np_random(seed)


gym.register(id="TagBox-v0", entry_point=TagBoxEnv)
