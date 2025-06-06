import numpy as np

import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Sequence, Box


from ocatari.ram.extract_ram_info import get_class_dict, get_max_objects
from ocatari.ram.game_objects import NoObject


class SeqSpace(Sequence):
    @property
    def shape(self):
        return self.feature_space.shape


class OCWrapper(ObservationWrapper):
    """
    A Wrapper
    """
    def __init__(self, env: gym.Env):
        self.object_types = {k: i for i, k in enumerate(get_class_dict(env.game_name).keys())}
        self.num_objects = len(self.object_types)
        super().__init__(env)

    def observation(self, observation):
        new_obs = np.swapaxes(observation, 0, 1)
        return np.reshape(new_obs, (3, -1))[:, 4:]


class EgoCentricObsWrapper(ObservationWrapper):
    """
    A Wrapper
    """
    def __init__(self, env: gym.Env, ego_pos: int):
        self.ego_pos = ego_pos
        super().__init__(env)

    def observation(self, observation):
        obs = observation - observation[self.ego_pos]
        obs[self.ego_pos] = observation[self.ego_pos]
        return obs


class ObjectLambdaWrapper(ObservationWrapper):
    def __init__(self, env, obs_lambda):
        super().__init__(env)
        self.obs_lambda = obs_lambda
        self.feature_size = obs_lambda(NoObject()).shape
        self.observation_space = SeqSpace(Box(-np.inf, np.inf, shape=self.feature_size),
                                          stack=True)
        self.max_len = sum(get_max_objects(env.game_name, env.hud).values()) # noqa: type(env) == OCAtari


    def observation(self, observation):
        state = np.zeros((self.max_len, self.feature_size))
        i = 0
        for o in self.env.objects:  # noqa: type(env) == OCAtari
            if not (o is None or o.category == "NoObject"):
                state[i] = self.obs_lambda(o)
                i += 1
        return state


class EgoCentricWrapper(ObservationWrapper):
    def __init__(self, env, player_name="Player", include_type=False,
                 use_polar_coordinates=False, relative_velocity=True):
        super().__init__(env)
        self.player_name = player_name
        self.use_polar_coordinates = use_polar_coordinates
        if relative_velocity:
            self.relative_pv_index = 4
        else:
            self.relative_pv_index = 2

        max_objs = get_max_objects(env.game_name, env.hud) # noqa: type(env) == OCAtari
        self.num_obj = len(max_objs)
        self.max_len = sum(max_objs.values())
        self.feature_size = 4 + len(max_objs) * include_type
        self.object_types =  {k: np.eye(1, self.feature_size, i) for i, k in enumerate(max_objs.keys())}

        self.observation_space = Box(-np.inf, np.inf, shape=(self.max_len, self.feature_size))

    def observation(self, observation):
        state = np.zeros((self.max_len, self.feature_size))
        i = 0
        player_idx = -1
        for o in self.env.objects:  # noqa: type(env) == OCAtari
            if not (o is None or o.category == "NoObject"):
                if player_idx == -1 and o.category == self.player_name:
                    player_pos_v = [o.dx, o.dy, *o.center]
                    player_idx = i
                state[i] = self.object_types[o.category]
                state[i, -4:-2] = o.dx, o.dy
                state[i, -2:] = o.center
                i += 1
        if player_idx != -1:  # sometimes the player disappears on termination
            # either calculate relative pos and velocity or just pos
            state[:, -self.relative_pv_index:] -= player_pos_v[-self.relative_pv_index:]  # noqa: is always set
            state[player_idx, -self.relative_pv_index:-2] = player_pos_v[-self.relative_pv_index:-2]
            if self.use_polar_coordinates:
                # position
                state[:, -2:] = get_polar_coordinates(state[:, -2:])
                # velocity
                state[:, -4:-2] = get_polar_coordinates(state[:, -4:-2])

            state[player_idx, -2:] = player_pos_v[-2:]
        return state


# xys.shape = (batch, obs, 2)
def get_polar_coordinates(xys):
    """Returns the polar coordinates of a point."""
    polar_coordinates = np.empty_like(xys)
    polar_coordinates[:, 0] = np.sqrt(np.sum(xys**2, axis=1))
    polar_coordinates[:, 1] = np.arctan2(xys[:, 0], xys[:, 1])  # angle in radians
    return polar_coordinates
