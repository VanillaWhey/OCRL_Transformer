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
    def __init__(self, env, player_name="Player", include_type=False):
        super().__init__(env)
        self.player_name = player_name

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
                if o.category == self.player_name:
                    player_pos = o.center
                    player_idx = i
                state[i] = self.object_types[o.category]
                state[i, -4:-2] = o.dx, o.dy
                state[i, -2:] = o.center
                i += 1
        if player_idx != -1:  # sometimes the player disappears on termination
            state[:, -2:] -= player_pos  # noqa: position is always set
            state[player_idx, -2:] = player_pos
        return state