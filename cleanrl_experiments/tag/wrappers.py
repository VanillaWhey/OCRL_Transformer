"""Observation wrappers for TagEnv (kept out of the environment on purpose)."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from gymnasium.utils import seeding
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from cleanrl_experiments.tag.tag_env import FEATURES, N_CATEGORIES, OMEGA, THETA, V, X, Y  # TagEnv observation columns (the first N_CATEGORIES = one-hot category)


def _disp(d, L, periodic=True):
    """Displacement d: shortest wrapped one on the torus (periodic), the plain difference in a box."""
    return (d + L / 2) % L - L / 2 if periodic else d


def _rot_rows(angle: float) -> np.ndarray:
    """Matrix M with (row vectors) @ M = the vectors rotated by -angle (into a frame turned by +angle)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


class AnchorFrame(gym.ObservationWrapper):
    """Express every object in the frame of reference of the object with ``object_id == anchor_id``
    (0 player, 1 goal, 2 first distractor, ..., see TagEnv). Always the same object, also when
    several objects share its category; the wrapper finds its row via ``env.row_object_ids``.

    Positions become the shortest (wrapped) displacement from the anchor; with ``rotate``
    they are additionally rotated so that the anchor's heading points along +x, and all
    headings are taken relative to the anchor's heading. Speeds and turn rates are frame
    independent and stay unchanged. The anchor's own row becomes
    (one-hot category, 0, 0, 0 if rotate else theta, v, omega).

    ``offset`` moves the origin away from the anchor by a fixed vector, given in the anchor's
    (rotated) frame if ``rotate``, else in world axes; the anchor's position then becomes
    -offset (wrapped into the world if not ``rotate``).

    ``mount_angle`` (rad, ``rotate`` only) turns the frame's axes by a further fixed angle
    against the anchor's heading, like a sensor glued to the body at (``offset``, ``mount_angle``):
    positions are additionally rotated by -mount_angle and headings reduced by mount_angle, so the
    anchor's row reads (one-hot, R(-mount_angle) (-offset), -mount_angle, v, omega).
    """

    def __init__(self, env: gym.Env, anchor_id: int, rotate: bool = True, offset=(0.0, 0.0), mount_angle: float = 0.0):
        super().__init__(env)
        n_objects = len(env.unwrapped.objects)
        assert 0 <= anchor_id < n_objects, f"anchor_id must be an object id in [0, {n_objects})"
        self.anchor_id = anchor_id
        assert rotate or mount_angle == 0.0, "a mount angle needs rotate=True"
        self.rotate = rotate
        self.offset = np.asarray(offset, dtype=np.float32)
        self.mount_angle = float(mount_angle)
        self.world_size = L = env.unwrapped.world_size
        self.periodic = periodic = getattr(env.unwrapped, "periodic", True)  # False: walled box (tag_box_env)

        # largest possible |coordinate| per axis (on the torus a translation-only frame re-wraps after the offset)
        if periodic:
            r = L / math.sqrt(2) if rotate else L / 2
            self._bound = r + np.abs(self.offset) if rotate else np.full(2, r, dtype=np.float32)
        else:
            r = L * math.sqrt(2) if rotate else L
            self._bound = r + np.abs(self.offset)
        if self.mount_angle:  # rotating the shifted positions mixes the axes: bound by the norm
            self._bound = np.full(2, r + np.linalg.norm(self.offset), dtype=np.float32)
        low, high = env.observation_space.low.copy(), env.observation_space.high.copy()
        low[:, [X, Y]], high[:, [X, Y]] = -self._bound, self._bound
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # relative positions are only periodic as long as they are not rotated
        periods = list(env.get_wrapper_attr("feature_periods"))
        periods[X] = periods[Y] = L if (periodic and not rotate) else 0.0
        self.feature_periods = periods

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return self.transform(obs, self.env.unwrapped.row_object_ids)

    def transform(self, obs: np.ndarray, row_object_ids) -> np.ndarray:
        """Anchor transform of an observation whose rows hold the objects ``row_object_ids``."""
        L = self.world_size
        anchor = obs[list(row_object_ids).index(self.anchor_id)]
        out = obs.copy()
        if self.rotate:
            d = _disp(obs[:, [X, Y]] - anchor[[X, Y]], L, self.periodic)
            c, s = math.cos(anchor[THETA]), math.sin(anchor[THETA])
            d = d @ np.array([[c, -s], [s, c]], dtype=np.float32) - self.offset  # rotate by -theta_anchor
            if self.mount_angle:
                d = d @ _rot_rows(self.mount_angle).astype(np.float32)
            out[:, THETA] = (obs[:, THETA] - anchor[THETA] - self.mount_angle + math.pi) % (2 * math.pi) - math.pi
        else:
            d = _disp(obs[:, [X, Y]] - anchor[[X, Y]] - self.offset, L, self.periodic)
        out[:, [X, Y]] = np.clip(d, -self._bound, self._bound)  # guard against float rounding at the corners
        return out


class KinematicFrame(gym.ObservationWrapper):
    """Full kinematic change into the co-moving, co-rotating frame of the object with
    ``object_id == anchor_id`` (0 player, 1 goal, 2 first distractor, ..., see TagEnv).

    In this frame the anchor sits at the origin, points along +x and neither moves nor turns:
    its row becomes (one-hot category, 0, 0, 0, 0, 0, 0). For every other object (landmarks
    included: in the world they have a position and orientation but never move), with
    r = shortest wrapped displacement from the anchor and R = rotation by -theta_anchor:
        position  r'      = R r
        heading   theta'  = wrap(theta - theta_anchor)
        velocity  v'      = R (v_vec - v_vec_anchor) - omega_anchor * (-r'_y, r'_x)
                            (v_vec = v (cos theta, sin theta); the last term is the apparent
                            velocity caused by the frame's rotation)
        turn rate omega'  = omega - omega_anchor
    The scalar speed is not enough to describe a velocity that is no longer aligned with the
    heading, so the observation rows become
        [is_player, is_goal, is_distractor, is_landmark, x, y, theta, vx, vy, omega]
    (columns KinematicFrame.X, .Y, .THETA, .VX, .VY, .OMEGA). Velocities and turn rates are
    per env step.
    """

    X, Y, THETA, VX, VY, OMEGA = range(N_CATEGORIES, N_CATEGORIES + 6)

    def __init__(self, env: gym.Env, anchor_id: int):
        super().__init__(env)
        n_objects = len(env.unwrapped.objects)
        assert 0 <= anchor_id < n_objects, f"anchor_id must be an object id in [0, {n_objects})"
        self.anchor_id = anchor_id
        self.world_size = L = env.unwrapped.world_size
        self.periodic = getattr(env.unwrapped, "periodic", True)

        base_high = env.observation_space.high
        self._r_max = r = L / math.sqrt(2) if self.periodic else L * math.sqrt(2)
        v_max = float(base_high[:, V].max())
        w_max = float(np.abs(base_high[:, OMEGA]).max())
        self._v_max = u = 2 * v_max + w_max * r  # |v'| <= |v| + |v_anchor| + |omega_anchor| |r'|
        w = 2 * w_max
        low = np.array([0.0] * N_CATEGORIES + [-r, -r, -math.pi, -u, -u, -w], dtype=np.float32)
        high = np.array([1.0] * N_CATEGORIES + [r, r, math.pi, u, u, w], dtype=np.float32)
        self.observation_space = spaces.Box(np.tile(low, (n_objects, 1)), np.tile(high, (n_objects, 1)), dtype=np.float32)
        self.feature_periods = [0.0] * N_CATEGORIES + [0.0, 0.0, 2 * math.pi, 0.0, 0.0, 0.0]  # rotated positions are not periodic

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return self.transform(obs, self.env.unwrapped.row_object_ids)

    def transform(self, obs: np.ndarray, row_object_ids) -> np.ndarray:
        """Kinematic transform of an observation whose rows hold the objects ``row_object_ids``."""
        L = self.world_size
        obs = obs.astype(np.float64)
        anchor = obs[list(row_object_ids).index(self.anchor_id)]
        c, s = math.cos(anchor[THETA]), math.sin(anchor[THETA])
        rot = np.array([[c, -s], [s, c]])  # row vectors @ rot = rotation by -theta_anchor

        r = _disp(obs[:, [X, Y]] - anchor[[X, Y]], L, self.periodic) @ rot
        theta = (obs[:, THETA] - anchor[THETA] + math.pi) % (2 * math.pi) - math.pi
        v_world = obs[:, [V]] * np.stack([np.cos(obs[:, THETA]), np.sin(obs[:, THETA])], axis=1)
        v_anchor = anchor[V] * np.array([c, s])
        v = (v_world - v_anchor) @ rot - anchor[OMEGA] * np.stack([-r[:, 1], r[:, 0]], axis=1)
        omega = obs[:, OMEGA] - anchor[OMEGA]

        r = np.clip(r, -self._r_max, self._r_max)  # guard against float rounding at the corners
        v = np.clip(v, -self._v_max, self._v_max)
        return np.column_stack([obs[:, :N_CATEGORIES], r, theta, v, omega]).astype(np.float32)


class ShiftedKinematicFrame(KinematicFrame):
    """KinematicFrame whose origin is moved away from the anchor by ``offset``, a fixed vector in
    the anchor's (rotated) frame, e.g. (-3, 2) = 3 behind and 2 to the left of the anchor.

    The new origin is a point attached to the anchor, so it co-moves and co-rotates with the frame:
    in frame coordinates the shift is a constant, and constants have no time derivative. Hence
    only the positions change, x' = R r - offset; headings, velocities and turn rates are exactly
    those of KinematicFrame. (Check: the origin moves with v_anchor + omega_anchor x offset in the
    world, and v' = R (v_vec - v_origin) - omega_anchor x (r' - offset) = R (v_vec - v_anchor) -
    omega_anchor x r'.) The anchor's row becomes (one-hot, -offset_x, -offset_y, 0, 0, 0, 0): it
    sits still at -offset. As in AnchorFrame(rotate=True, offset), every object is shown at its
    copy closest to the anchor (not to the shifted origin), so the shifted and unshifted frames
    show the same copies.

    ``mount_angle`` (rad) additionally turns the frame's axes by a fixed angle against the anchor's
    heading (a sensor glued to the body at pose (``offset``, ``mount_angle``)). A fixed rotation of
    a co-rotating frame is again co-rotating, so this is a constant relabelling of the axes:
    positions and velocities are rotated by -mount_angle, headings reduced by mount_angle, turn
    rates unchanged. The anchor's row becomes (one-hot, R(-mount_angle) (-offset), -mount_angle,
    0, 0, 0).
    """

    def __init__(self, env: gym.Env, anchor_id: int, offset=(0.0, 0.0), mount_angle: float = 0.0):
        super().__init__(env, anchor_id)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.mount_angle = float(mount_angle)
        low, high = self.observation_space.low.copy(), self.observation_space.high.copy()
        # rotating the shifted positions mixes the axes: bound by the norm (velocity bounds are already norm bounds)
        bound = self._r_max + (np.linalg.norm(self.offset) if self.mount_angle else np.abs(self.offset))
        low[:, [self.X, self.Y]], high[:, [self.X, self.Y]] = -bound, bound
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def transform(self, obs: np.ndarray, row_object_ids) -> np.ndarray:
        out = super().transform(obs, row_object_ids).astype(np.float64)
        out[:, [self.X, self.Y]] -= self.offset
        if self.mount_angle:
            m = _rot_rows(self.mount_angle)
            out[:, [self.X, self.Y]] = out[:, [self.X, self.Y]] @ m
            out[:, [self.VX, self.VY]] = out[:, [self.VX, self.VY]] @ m
            out[:, self.THETA] = (out[:, self.THETA] - self.mount_angle + math.pi) % (2 * math.pi) - math.pi
            high = self.observation_space.high[0]  # guard against float rounding at the bounds
            cols = [self.X, self.Y, self.VX, self.VY]
            out[:, cols] = np.clip(out[:, cols], -high[cols], high[cols])
        return out.astype(np.float32)


class FixedThrottle(gym.ActionWrapper):
    """Steering-only control: the agent outputs just the turn command a_omega in [-1, 1]; the
    throttle a_v is fixed to ``throttle`` (1.0 = always accelerate, as the probes and the oracle do)."""

    def __init__(self, env: gym.Env, throttle: float = 1.0):
        super().__init__(env)
        self.throttle = float(throttle)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        return np.array([self.throttle, float(np.asarray(action).reshape(-1)[0])], dtype=np.float32)


class ShuffleRows(gym.ObservationWrapper):
    """Observation rows in a random order, new at every reset (like the unordered output of an
    object detector); ``self.row_object_ids`` says which object is in which row. Put it outermost
    (after any frame wrapper), since the frame wrappers expect rows in object-id order."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.order = np.arange(env.observation_space.shape[0])

    @property
    def row_object_ids(self) -> list[int]:
        return [int(i) for i in self.order]

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset(seed=seed, options=options)
        self.order = self.env.unwrapped.np_random.permutation(len(self.order))  # seeded by env.reset(seed)
        return self.observation(obs)

    def observation(self, obs):
        return obs[self.order]


# ---------------------------------------------------------------------------------------------
# Vector versions for tag_env.TagVecEnv: the same formulas, written out for a batch of
# environments (num_envs, n_objects, features). They also transform infos["final_obs"] (the last
# observation of an episode that ended in this step), which PPO needs for time-limit bootstrapping.
# Rows must be in object-id order, i.e. VecShuffleRows goes outermost.
# ---------------------------------------------------------------------------------------------

if hasattr(gym.vector, "VectorWrapper"):  # gymnasium >= 1.0
    VectorWrapper = gym.vector.VectorWrapper
else:  # gymnasium 0.29

    class VectorWrapper(gym.vector.VectorEnvWrapper):
        """Stand-in for gymnasium 1.0's VectorWrapper on top of 0.29's VectorEnvWrapper.

        0.29's VectorEnvWrapper does not forward reset() / step(): it inherits VectorEnv.reset/step,
        which call self.reset_async/_wait and step_async/_wait, and those go straight to the wrapped
        env's *_async/*_wait, skipping the reset/step overrides of any wrapper in between (and
        TagVecEnv implements reset/step directly). Here reset/step/render call the wrapped env's
        reset/step/render, as in 1.0. Attribute forwarding also skips 0.29's deprecation warning,
        and metadata / render_mode / spec / np_random come from the wrapped env instead of the
        gym.Env class defaults.
        """

        def reset(self, *, seed=None, options=None):
            return self.env.reset(seed=seed, options=options)

        def step(self, actions):
            return self.env.step(actions)

        def render(self):
            return self.env.render()

        def __getattr__(self, name):
            if name.startswith(
                    "_") or name == "env":  # "env" missing = __init__ has not run yet
                raise AttributeError(
                    f"attempted to get missing private attribute '{name}'")
            return getattr(self.env, name)

        @property
        def metadata(self):
            return self.env.metadata

        @property
        def render_mode(self):
            return self.env.render_mode

        @property
        def spec(self):
            return self.env.spec

        @property
        def np_random(self):
            return self.env.np_random

        @np_random.setter
        def np_random(self, value):
            self.env.np_random = value

        def __del__(self):
            if "env" in self.__dict__:  # VectorEnvWrapper.__del__ fails if __init__ raised before self.env
                super().__del__()

        def seed(self, seed):
            self.np_random = seeding.np_random(seed)


class _VecObsWrapper(VectorWrapper):
    """Applies self.transform(batch) to observations and to infos["final_obs"]."""

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset(seed=seed, options=options)
        return self.transform(obs)

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        if "final_obs" in infos:
            infos["final_obs"] = infos["final_obs"].copy()
            for i in np.flatnonzero(infos["_final_obs"]):
                infos["final_obs"][i] = \
                self.transform(infos["final_obs"][i][None])[0]
        return self.transform(obs), rewards, terminated, truncated, infos


class VecAnchorFrame(_VecObsWrapper):
    """Vector version of AnchorFrame (no offset / mount angle): positions relative to the object
    ``anchor_id``; with ``rotate`` also rotated so that the anchor points along +x, headings relative."""

    def __init__(self, env, anchor_id: int, rotate: bool = True):
        super().__init__(env)
        self.anchor_id, self.rotate = anchor_id, rotate
        self.world_size = L = env.unwrapped.world_size
        self.periodic = periodic = getattr(env.unwrapped, "periodic", True)
        self.bound = (L / math.sqrt(2) if rotate else L / 2) if periodic else (
            L * math.sqrt(2) if rotate else L)
        base = env.single_observation_space
        low, high = base.low.copy(), base.high.copy()
        low[:, [X, Y]], high[:, [X, Y]] = -self.bound, self.bound
        self.single_observation_space = spaces.Box(low, high, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space,
                                             env.num_envs)
        periods = list(env.feature_periods)
        periods[X] = periods[Y] = L if (periodic and not rotate) else 0.0
        self.feature_periods = periods

    def transform(self, obs):
        L = self.world_size
        a = obs[:, self.anchor_id]  # (N, F)
        dx = _disp(obs[..., X] - a[:, None, X], L, self.periodic)
        dy = _disp(obs[..., Y] - a[:, None, Y], L, self.periodic)
        out = obs.copy()
        if self.rotate:
            c, s = np.cos(a[:, None, THETA]), np.sin(a[:, None, THETA])
            dx, dy = dx * c + dy * s, -dx * s + dy * c  # rotate by -theta_anchor
            out[..., THETA] = (obs[..., THETA] - a[
                :, None, THETA] + math.pi) % (2 * math.pi) - math.pi
        out[..., X], out[..., Y] = np.clip(dx, -self.bound,
                                           self.bound), np.clip(dy,
                                                                -self.bound,
                                                                self.bound)
        return out


class VecKinematicFrame(_VecObsWrapper):
    """Vector version of KinematicFrame: co-moving, co-rotating frame of the object ``anchor_id``;
    rows become [one-hot, x, y, theta, vx, vy, omega] (columns KinematicFrame.X ... .OMEGA)."""

    def __init__(self, env, anchor_id: int):
        super().__init__(env)
        self.anchor_id = anchor_id
        self.world_size = L = env.unwrapped.world_size
        base = env.single_observation_space
        n = base.shape[0]
        self.periodic = getattr(env.unwrapped, "periodic", True)
        self.r_max = r = L / math.sqrt(2) if self.periodic else L * math.sqrt(
            2)
        v_max, w_max = float(base.high[:, V].max()), float(
            np.abs(base.high[:, OMEGA]).max())
        self.v_max = u = 2 * v_max + w_max * r
        w = 2 * w_max
        low = np.array([0.0] * N_CATEGORIES + [-r, -r, -math.pi, -u, -u, -w],
                       dtype=np.float32)
        high = np.array([1.0] * N_CATEGORIES + [r, r, math.pi, u, u, w],
                        dtype=np.float32)
        self.single_observation_space = spaces.Box(np.tile(low, (n, 1)),
                                                   np.tile(high, (n, 1)),
                                                   dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space,
                                             env.num_envs)
        self.feature_periods = [0.0] * N_CATEGORIES + [0.0, 0.0, 2 * math.pi,
                                                       0.0, 0.0, 0.0]

    def transform(self, obs):
        L = self.world_size
        obs = obs.astype(np.float64)
        a = obs[:, self.anchor_id]
        c, s = np.cos(a[:, None, THETA]), np.sin(a[:, None, THETA])
        dx, dy = _disp(obs[..., X] - a[:, None, X], L, self.periodic), _disp(
            obs[..., Y] - a[:, None, Y], L, self.periodic)
        rx, ry = dx * c + dy * s, -dx * s + dy * c  # R r, rotation by -theta_anchor
        theta = (obs[..., THETA] - a[:, None, THETA] + math.pi) % (
                    2 * math.pi) - math.pi
        vwx, vwy = obs[..., V] * np.cos(obs[..., THETA]), obs[..., V] * np.sin(
            obs[..., THETA])
        vax, vay = a[:, None, V] * c, a[:, None, V] * s
        wa = a[:, None, OMEGA]
        vx = (vwx - vax) * c + (
                    vwy - vay) * s + wa * ry  # R (v - v_anchor) - omega_anchor x r'
        vy = -(vwx - vax) * s + (vwy - vay) * c - wa * rx
        omega = obs[..., OMEGA] - wa
        rx, ry = np.clip(rx, -self.r_max, self.r_max), np.clip(ry, -self.r_max,
                                                               self.r_max)
        vx, vy = np.clip(vx, -self.v_max, self.v_max), np.clip(vy, -self.v_max,
                                                               self.v_max)
        return np.concatenate([obs[..., :N_CATEGORIES],
                               np.stack([rx, ry, theta, vx, vy, omega],
                                        axis=-1)],
                              axis=-1).astype(np.float32)


class VecFixedThrottle(VectorWrapper):
    """Vector version of FixedThrottle: actions (num_envs, 1) = turn command; throttle fixed."""

    def __init__(self, env, throttle: float = 1.0):
        super().__init__(env)
        self.throttle = float(throttle)
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(1,),
                                              dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, env.num_envs)

    @property
    def feature_periods(self):
        return self.env.feature_periods

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_envs,
                                                                1)
        return self.env.step(
            np.concatenate([np.full_like(actions, self.throttle), actions],
                           axis=1))


class VecShuffleRows(VectorWrapper):
    """Vector version of ShuffleRows: every environment gets a new random row order at the start of
    each episode; ``self.order[i]`` = object ids of the rows of environment i. Its own random
    generator (``seed``), unlike ShuffleRows, which draws from the env's np_random, so the two give
    different orders for the same seed."""

    def __init__(self, env, seed: int | None = None):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.order = np.tile(np.arange(env.single_observation_space.shape[0]),
                             (env.num_envs, 1))

    @property
    def feature_periods(self):
        return self.env.feature_periods

    def _new_order(self, envs):
        self.order[envs] = np.argsort(
            self.rng.random((len(envs), self.order.shape[1])), axis=1)

    def _apply(self, obs, envs=slice(None)):
        return np.take_along_axis(obs, self.order[envs][..., None], axis=1)

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self._new_order(np.arange(self.num_envs))
        return self._apply(obs), infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        done = np.flatnonzero(terminated | truncated)
        if "final_obs" in infos:
            infos["final_obs"] = infos["final_obs"].copy()
            for i in np.flatnonzero(infos["_final_obs"]):
                infos["final_obs"][i] = infos["final_obs"][i][
                    self.order[i]]  # old order: same episode
        self._new_order(
            done)  # the observation of a reset env belongs to its new episode
        return self._apply(obs), rewards, terminated, truncated, infos


class SequenzWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert type(env.observation_space) == spaces.Box

        num_features = len(FEATURES)

        self.shape = list(self.env.observation_space.shape)
        tmp = self.shape[-1]
        self.shape[-1] = self.shape[-2] // num_features
        self.shape[-2] = tmp

        self.shape = np.append(self.shape, [num_features])
        self.observation_space = spaces.Box(self.observation(env.observation_space.low), self.observation(env.observation_space.high))


    def observation(self, obs):
        np.swapaxes(obs, -1, -2)
        return np.reshape(obs, self.shape)


class TagVecEnvAdapter:
    """Gives a tag vector env (gymnasium VectorEnv API) the SB3 VecEnv interface used by the training loop:
    single observation/action spaces, reset() -> obs, step() -> (obs, reward, done, list of info dicts)."""

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self._seed = None

    def seed(self, seed):
        self._seed = seed  # used by the next reset

    def reset(self):
        obs = self.env.reset(seed=self._seed)
        self._seed = None
        return obs

    def step(self, actions):
        obs, reward, terminated, truncated, infos = self.env.step(actions)
        done = terminated | truncated
        list_infos = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(infos.get("_episode", [])):
            list_infos[i]["episode"] = {"r": infos["episode"]["r"][i], "l": infos["episode"]["l"][i]}
            list_infos[i]["TimeLimit.truncated"] = bool(truncated[i])
            if "final_obs" in infos:
                list_infos[i]["terminal_observation"] = infos["final_obs"][i]
        return obs, reward, done, list_infos

    def close(self):
        pass
