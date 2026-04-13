from collections import deque

import numpy as np

import gymnasium as gym
import torch
from gymnasium import ObservationWrapper
from gymnasium.spaces import Sequence, Box

from ocatari.ram.extract_ram_info import get_class_dict, get_max_objects
from ocatari.ram.game_objects import NoObject
from torchvision.transforms import RandomCrop, functional as F

from copy import copy

import cv2


class SeqSpace(Sequence):
    @property
    def shape(self):
        return self.feature_space.shape


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
        self.brutto_feature_size = len(obs_lambda(NoObject()))
        self.observation_space = SeqSpace(Box(-np.inf, np.inf, shape=self.brutto_feature_size),
                                          stack=True)
        self.max_len = sum(get_max_objects(env.game_name, env.hud).values()) # noqa: type(env) == OCAtari


    def observation(self, observation):
        state = np.zeros((self.max_len, self.brutto_feature_size))
        i = 0
        for o in self.env.objects:  # noqa: type(env) == OCAtari
            if not (o is None or o.category == "NoObject"):
                state[i] = self.obs_lambda(o)
                i += 1
        return state


class EgoCentricWrapper(ObservationWrapper):
    def __init__(self, env, player_name="Player", type_embedding=None,
                 use_polar_coordinates=False, relative_velocity=True,
                 include_wh=False, zero_player=False):
        super().__init__(env)

        self.player_name = player_name
        self.use_polar_coordinates = use_polar_coordinates
        if relative_velocity:
            self.relative_pv_index = 4
        else:
            self.relative_pv_index = 2

        max_objs = get_max_objects(env.game_name, env.hud) # noqa: type(env) == OCAtari
        self.num_object_types = len(max_objs)
        self.max_len = sum(max_objs.values())

        if include_wh:
            self.feature_func = w_h_dx_dy_center
        else:
            self.feature_func = dx_dy_center

        self.netto_feature_size = len(self.feature_func(NoObject()))  # object features only
        self.brutto_feature_size = self.netto_feature_size  # may include type embedding etc.

        if type_embedding == "one_hot":
            self.brutto_feature_size += self.num_object_types
            self.object_types =  {k: np.eye(1, self.brutto_feature_size, i) for i, k in enumerate(max_objs.keys())}
        elif type_embedding == "additive":
            self.object_types = {k: positional_encode(self.brutto_feature_size, i) for i, k in enumerate(max_objs.keys())}
        elif type_embedding is None:
            self.object_types = {k: np.zeros((1, self.brutto_feature_size)) for k in max_objs.keys()}
        else:
            raise AttributeError(f"Type embedding {type_embedding} not supported!")

        self.observation_space = Box(-np.inf, np.inf, shape=(self.max_len, self.brutto_feature_size))

        self.zero_player = zero_player

    def observation(self, observation):
        state = np.zeros((self.max_len, self.brutto_feature_size))
        emb = np.zeros((len(self.env.objects), self.brutto_feature_size))
        i = 0
        player_idx = -1
        for o in self.env.objects:  # noqa: type(env) == OCAtari
            if not (o is None or "NoObject" in o.category):
                center_x, center_y = o.center  # only calculate once
                if o.category == self.player_name and player_idx == -1:
                    player_pos_v = [o.dx, o.dy, center_x, center_y]
                    player_idx = i
                emb[i] = self.object_types[o.category]
                state[i, -self.netto_feature_size:] = self.feature_func(o)
                i += 1
        if player_idx != -1:  # sometimes the player disappears on termination
            # either calculate relative pos and velocity or just pos
            state[:i, -self.relative_pv_index:] -= player_pos_v[-self.relative_pv_index:]  # noqa: is always set

            if not self.zero_player:  # otherwise keep player values as 0
                state[player_idx, -self.relative_pv_index:-2] = player_pos_v[-self.relative_pv_index:-2]
            if self.use_polar_coordinates:
                # position
                state[:i, -2:] = get_polar_coordinates(state[:i, -2:])
                # velocity
                state[:i, -4:-2] = get_polar_coordinates(state[:i, -4:-2])

            if not self.zero_player:  # otherwise keep player values as 0
                state[player_idx, -2:] = player_pos_v[-2:]

        # object type
        state[:i] += emb[:i]

        return state


class ObjFlatObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(-np.inf, np.inf, shape=(env.max_len * env.brutto_feature_size,))

    def observation(self, observation):
        return observation.flatten()


class RandomCropWrapper(RandomCrop, ObservationWrapper):
    def __init__(self, env, padding, obs_mode="ori"):
        super().__init__(env.observation_space.shape[:2], padding, padding_mode="edge")
        super(ObservationWrapper, self).__init__(env)

        self.objects = []
        self.offset_x, self.offset_y = 0, 0
        self.h, self.w = 0, 0

        self._state_buffer_dqn = deque([], 4)

        if type(padding) == int:
            self.pads = (padding, padding)
        else:
            self.pads = padding[:2]

        if obs_mode == "ori":
            self.obs = lambda o: o
        elif obs_mode == "dqn":
            self.obs = self.down_scale
        else:
            raise NotImplementedError(f"obs_mode {obs_mode} is not supported!")

    def down_scale(self, obs):
        dqn_obs = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)
        self._state_buffer_dqn.append(dqn_obs)

        return np.array(self._state_buffer_dqn)

    def get_params(self, img, output_size):
        return self.offset_y, self.offset_x, self.h, self.w

    def observation(self, observation):
        new_obs = self(torch.tensor(observation).moveaxis(-1, 0))

        self.objects = []
        for obj in self.env.objects:
            if not (obj is None or "NoObject" in obj.category):
                o = copy(obj)
                o.xy = (obj.x - self.offset_x + self.pads[0], obj.y - self.offset_y + self.pads[1])
                self.objects.append(o)

        return self.obs(new_obs.moveaxis(0, -1).numpy())

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        img = F.pad(torch.tensor(obs).moveaxis(-1, 0), self.padding, self.fill, self.padding_mode)
        self.offset_y, self.offset_x, self.h, self.w = super().get_params(img, self.size)

        # fill buffer
        self.observation(obs)
        self.observation(obs)
        self.observation(obs)

        return self.observation(obs), info

    def __getattr__(self, name: str):
        return super(ObservationWrapper, self).__getattr__(name)


# xys.shape = (batch, obs, 2)
def get_polar_coordinates(xys):
    """Returns the polar coordinates of a point."""
    polar_coordinates = np.empty_like(xys)
    polar_coordinates[:, 0] = np.sqrt(np.sum(xys**2, axis=1))
    polar_coordinates[:, 1] = np.arctan2(xys[:, 0], xys[:, 1])  # angle in radians
    return polar_coordinates


def positional_encode(d_model, position):
    pe = np.empty(d_model)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[0::2] = np.sin(position * div_term)
    pe[1::2] = np.cos(position * div_term)
    return pe


# object feature functions
def dx_dy_center(o):
    center_x, center_y = o.center
    return o.dx, o.dy, center_x, center_y


def w_h_dx_dy_center(o):
    center_x, center_y = o.center
    w, h = o.wh
    return w, h, o.dx, o.dy, center_x, center_y
